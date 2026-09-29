# Entity Resolution Pipeline — Amazon ML Challenge

Production-grade, modular entity-resolution pipeline that maps each Source 1
(clean reference) record to zero, one, or multiple matching records in Source 2
and Source 3 (messy vendor feeds), optimized directly for **macro F0.5**.

## Setup

```bash
pip install -r requirements.txt
```

### macOS

LightGBM needs Homebrew's OpenMP, and faiss / scikit-learn / torch each bundle
their own copy of `libomp`. With several copies loaded, LightGBM training
deadlocks (0% CPU, never finishes). Point the bundled copies at Homebrew's:

```bash
brew install libomp
pip install -r requirements.txt
for f in .venv/lib/python3.*/site-packages/{faiss/.dylibs,sklearn/.dylibs,torch/lib}/libomp.dylib; do
  mv "$f" "$f.orig" && ln -s /opt/homebrew/opt/libomp/lib/libomp.dylib "$f"
done
```

Re-run the loop after reinstalling any of those packages.

`faiss-cpu`, `sentence-transformers`, and `torch`/`transformers` are only
required if you keep embedding-based blocking and the cross-encoder enabled
(both are on by default in `configs/config.yaml`, and both can be turned off
for a lighter, faster GBM-only pipeline — see below).

## Data layout

Unzip the challenge's `student_resource` archive into `data/`:

```bash
unzip -q <archive>.zip -x '__MACOSX/*' -d data
```

giving `data/student_resource/dataset/{train,test}/*.tsv`
(`entity_id, business_name, business_address, country`; ground truth is
`source1_entity_id, matched_entity_ids`). Paths are set in
`configs/config.yaml` under `paths:`. The data has no zip/city columns, so the
exact-key blocking strategies use `country` as their geo key and skip keys
larger than `blocking.max_block_size`.

## Running

### Train

```bash
python train.py --config configs/config.yaml
```

Produces:
- `outputs/candidate_pairs_train.tsv` — every candidate pair that survived blocking
- `outputs/features_train.parquet` — full feature matrix + labels
- `models/gbm_model.txt` (+ `.meta.json`) — trained LightGBM matcher
- `models/cross_encoder/` — fine-tuned transformer (if enabled)
- `models/decision_threshold.json` — F0.5-optimized decision threshold

Flags:
- `--sample N` — train on N random source1 records, matched against the FULL
  source2/source3 (so hard negatives look like test time). 100k is plenty for
  the GBM.
- `--sample-targets` — with `--sample`: also shrink source2/source3 to the same
  fraction (keeping the true matches) for a ~1 minute smoke test.
- `--skip-cross-encoder` — force GBM-only training this run, regardless of config.

### Predict

```bash
python predict.py --config configs/config.yaml
```

Produces the two submission files in `output/`:
- `matching_results.tsv` — `source1_entity_id, matched_entity_ids`
- `candidate_pairs.tsv` — `source1_entity_id, candidate_entity_ids`

Check them with the challenge validator:
```bash
python3 data/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir data/student_resource/dataset/test
```

Flags:
- `--sample N` / `--sample-targets` — smoke test only; the output is not a valid submission.

Prediction runs on the full test set: candidates are generated for all
source1 records, then features/scores/matches are computed
`inference.predict_chunk_size` source1 records at a time to bound memory.
- `--gbm-only` — score with the GBM only, even if a cross-encoder checkpoint exists.

### Defaults

The sentence-embedding blocking strategy (`blocking.use_embeddings`) and the
DeBERTa cross-encoder (`cross_encoder.enabled`) are both off: embeddings added
<0.1% blocking recall on the training data for hours of extra compute, and the
cross-encoder's training diverged (NaN loss) on Apple MPS.

## Architecture

```
Module 1 (src/preprocessing)  Text normalization: Devanagari -> Latin
                               transliteration, legal-suffix + street-suffix
                               expansion, accent/punctuation stripping, zip
                               normalization, phonetic + blocking-key columns.

Module 2 (src/blocking)       Per-country blocking strategies unioned together:
                               (1) exact phonetic+geo key match
                               (2) exact first-token+geo key match
                               (3) TF-IDF char-ngram top-K: SVD-compressed
                                   FAISS search, re-ranked by exact cosine
                               (4) optional sentence-embedding + FAISS top-K

Module 3 (src/features,       Feature engineering: Jaro-Winkler, Levenshtein,
          src/models)         token sort/set ratio, Monge-Elkan, char n-gram
                               Jaccard — computed separately on name & address,
                               plus numeric-token overlap, acronym-match,
                               zip-match, and blocking-strategy agreement flags.
                               Two models trained on this: LightGBM (tabular
                               features) and an optional DeBERTa-v3 cross-
                               encoder (raw text pairs). Scores are ensembled.

Module 4 (src/evaluation)     Macro F0.5 computed per-source1-entity and
                               averaged. Threshold grid-searched directly
                               against this metric (not proxied via F1/AUPRC).
                               Singletons handled explicitly: absent from the
                               match dict == empty match list == correct
                               "no match" prediction, never an omitted row.
```

## Key design decisions worth knowing before you tune this

- **Blocking recall is a hard ceiling.** Nothing downstream can recover a
  true match blocking never surfaced. `train.py` logs blocking recall against
  `train_ground_truth.tsv` every run and warns if it's below
  `blocking.min_blocking_recall_target` (default 0.98) — check this first if
  end-to-end F0.5 looks low.
- **GroupKFold by `source1_id`**, not random row splits, for both GBM early
  stopping and cross-encoder train/val split. A random split would leak
  candidate-ranking information across folds for the same entity.
- **Negatives are *hard* negatives.** Every non-matching candidate that
  survived blocking is used as a negative example, not randomly sampled
  unrelated pairs — this is what makes the classifier actually precision-
  sensitive on lexically-similar-but-wrong candidates, which is exactly what
  F0.5 punishes hardest.
- **Threshold optimization targets macro F0.5 directly**, evaluated against
  the full source1 id set (including ids with zero candidate pairs at all),
  so singleton correctness is baked into the number you're optimizing, not
  bolted on afterward.
- **No external APIs / geocoding is used anywhere** in this pipeline —
  `sentence-transformers` and `faiss-cpu` both run fully offline once model
  weights are cached locally, satisfying the challenge constraint.
