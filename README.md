# Entity Resolution Pipeline — Amazon ML Challenge

Production-grade, modular entity-resolution pipeline that maps each Source 1
(clean reference) record to zero, one, or multiple matching records in Source 2
and Source 3 (messy vendor feeds), optimized directly for **macro F0.5**.

## Setup

```bash
pip install -r requirements.txt
```

`faiss-cpu`, `sentence-transformers`, and `torch`/`transformers` are only
required if you keep embedding-based blocking and the cross-encoder enabled
(both are on by default in `configs/config.yaml`, and both can be turned off
for a lighter, faster GBM-only pipeline — see below).

## Data layout

Place these under `data/` (tab-separated, header row required):

| file | columns |
|---|---|
| `source1.tsv` | `id, company_name, address, city, zip` |
| `source2.tsv` | `id, company_name, address, city, zip` |
| `source3.tsv` | `id, company_name, address, city, zip` |
| `train_ground_truth.tsv` | `source1_id, matched_ids` (matched_ids: comma-separated ids from source2 **and/or** source3, empty string for singletons) |
| `test_source1.tsv` / `test_source2.tsv` / `test_source3.tsv` | same schema as above, no ground truth |

Column names are configurable in `configs/config.yaml` under `columns:` if
your raw files use different headers.

> **Note:** `ground_truth_wide_to_long` (in `src/pipeline_utils.py`) infers
> whether each matched id belongs to source2 or source3 by set membership.
> If your source2/source3 id spaces can collide, prefix ids at load time.

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
- `--skip-cross-encoder` — force GBM-only training this run, regardless of config.

### Predict

```bash
python predict.py --config configs/config.yaml
```

Produces `outputs/matching_results.tsv` with columns `source1_id, matched_ids`
(comma-separated, empty string for no-match records).

Flags:
- `--gbm-only` — score with the GBM only, even if a cross-encoder checkpoint exists.

### Lightweight (GBM-only, no torch/faiss/sentence-transformers) mode

Set in `configs/config.yaml`:
```yaml
blocking:
  use_embeddings: false
cross_encoder:
  enabled: false
```
This runs blocking on exact-key + TF-IDF strategies only, and scores with the
GBM alone — fully functional, just skips the two heaviest/most expensive
components. Recommended as a first baseline before turning embeddings/cross-
encoder back on.

## Architecture

```
Module 1 (src/preprocessing)  Text normalization: legal-suffix + street-suffix
                               expansion, accent/punctuation stripping, zip
                               normalization, phonetic + blocking-key columns.

Module 2 (src/blocking)       Three blocking strategies unioned together:
                               (1) exact phonetic+geo key match
                               (2) TF-IDF char-ngram cosine top-K
                               (3) sentence-embedding + FAISS ANN top-K
                               -> outputs/candidate_pairs_{train,test}.tsv

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
