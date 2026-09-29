#!/usr/bin/env python3
"""
End-to-end inference CLI.

Usage
-----
    python predict.py --config configs/config.yaml
    python predict.py --config configs/config.yaml --gbm-only

Pipeline stages
---------------
1. Load test source1/source2/source3 (no labels available/used).
2. Preprocess & normalize (Module 1) -- same functions as training, so
   normalization is guaranteed consistent between train and test.
3. Generate candidate pairs via the SAME multi-strategy blocking config
   used in training (Module 2). The TF-IDF vocabulary/SVD is refit on the
   test corpus, since test entities (and a whole country, France) are unseen
   in training.
4-6. For each chunk of source1 records: build the same feature set
   (Module 3a, no labels), score with the saved GBM (+ cross-encoder if
   available), and apply the saved decision threshold -- source1 ids with no
   pair above it get an empty match list.
7. Write `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
"""

from __future__ import annotations

import argparse
import logging
import time

import pandas as pd

from src.preprocessing import preprocess_dataframe
from src.blocking import generate_candidate_pairs
from src.features import build_pair_features
from src.models import GBMMatcher, CrossEncoderMatcher
from src.evaluation.metrics import (
    predictions_to_matches, matches_to_submission_df, candidates_to_submission_df, load_threshold,
)
from src.pipeline_utils import keep_pipeline_columns, load_config, load_source_tsv, ensemble_scores, ensure_dir, sample_sources

logger = logging.getLogger("predict")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference for the entity resolution pipeline.")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--gbm-only", action="store_true",
                    help="Ignore a saved cross-encoder even if present (faster, GBM-only scoring).")
    p.add_argument("--sample", type=int, default=None, metavar="N",
                    help="Smoke test: predict for N random source1 records only (not a valid submission).")
    p.add_argument("--sample-targets", action="store_true",
                    help="With --sample: also shrink source2/source3 to the same fraction.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    cols = cfg["columns"]
    paths = cfg["paths"]
    id_col = cols["id_col"]

    # ---------------------------------------------------------------
    # 1. Load test data
    # ---------------------------------------------------------------
    logger.info("Loading test source tables...")
    source1_raw = load_source_tsv(paths["test_source1"], id_col)
    source2_raw = load_source_tsv(paths["test_source2"], id_col)
    source3_raw = load_source_tsv(paths["test_source3"], id_col)
    if args.sample:
        logger.warning("--sample is for smoke tests only: the output will NOT cover every "
                       "test source1 entity and is not a valid submission.")
        source1_raw, source2_raw, source3_raw = sample_sources(
            source1_raw, source2_raw, source3_raw, args.sample,
            sample_targets=args.sample_targets,
        )
    logger.info("Loaded test: source1=%d, source2=%d, source3=%d",
                len(source1_raw), len(source2_raw), len(source3_raw))

    # ---------------------------------------------------------------
    # 2. Preprocess (Module 1) -- identical normalization as training
    # ---------------------------------------------------------------
    logger.info("Normalizing text fields...")
    source1 = preprocess_dataframe(
        source1_raw, cols["name_col"], cols["address_col"], cols["city_col"], cols["zip_col"],
        cfg["blocking"]["zip_prefix_len"],
    )
    source2 = preprocess_dataframe(
        source2_raw, cols["name_col"], cols["address_col"], cols["city_col"], cols["zip_col"],
        cfg["blocking"]["zip_prefix_len"],
    )
    source3 = preprocess_dataframe(
        source3_raw, cols["name_col"], cols["address_col"], cols["city_col"], cols["zip_col"],
        cfg["blocking"]["zip_prefix_len"],
    )

    # Keep only what later stages use (the raw text columns of 10M+ target
    # records are several GB we don't need).
    source1, source2, source3 = (keep_pipeline_columns(df) for df in (source1, source2, source3))
    del source1_raw, source2_raw, source3_raw

    # ---------------------------------------------------------------
    # 3. Blocking / candidate generation (Module 2)
    # ---------------------------------------------------------------
    logger.info("Generating candidate pairs for test set...")
    candidate_pairs = generate_candidate_pairs(source1, source2, source3, cfg)
    logger.info("%d candidate pairs for %d source1 records", len(candidate_pairs), len(source1))

    # ---------------------------------------------------------------
    # 4-6. Features -> scores -> matches, one chunk of source1 ids at a time
    # (the full test candidate set is ~100M pairs; featurizing it in one go
    # would not fit in memory).
    # ---------------------------------------------------------------
    gbm = GBMMatcher.load(paths["gbm_model_out"])

    cross_encoder = None
    if cfg["cross_encoder"]["enabled"] and not args.gbm_only:
        try:
            logger.info("Loading saved cross-encoder model...")
            cross_encoder = CrossEncoderMatcher.load(paths["cross_encoder_model_out"])
            s1_lookup = source1.set_index("id")[["clean_name", "clean_address"]]
            t_lookups = {
                "source2": source2.set_index("id")[["clean_name", "clean_address"]],
                "source3": source3.set_index("id")[["clean_name", "clean_address"]],
            }
        except (OSError, FileNotFoundError) as e:
            logger.warning("Could not load cross-encoder model (%s); falling back to GBM-only scoring.", e)

    # min_decision_threshold is enforced again here (defense in depth) even
    # though train.py already enforces it when the threshold was selected --
    # see load_threshold's docstring.
    min_floor = cfg["evaluation"]["min_decision_threshold"]
    threshold = load_threshold(paths["threshold_out"], min_decision_threshold=min_floor)
    logger.info(
        "Decision threshold=%.4f (precision floor=%.2f) -- only candidate pairs scoring strictly "
        "above this become predicted matches; every other source1 id maps to an empty match list.",
        threshold, min_floor,
    )

    all_s1_ids = source1["id"].tolist()
    s1_pos = pd.Index(all_s1_ids).get_indexer(candidate_pairs["source1_id"])
    chunk = cfg["inference"].get("predict_chunk_size", 100_000)
    ens_cfg = cfg["ensemble"]
    matches: dict[str, set] = {}
    candidate_frames = []

    for start in range(0, len(all_s1_ids), chunk):
        t0 = time.time()
        in_chunk = (s1_pos >= start) & (s1_pos < start + chunk)
        cand = candidate_pairs[in_chunk]
        chunk_ids = all_s1_ids[start:start + chunk]
        candidate_frames.append(candidates_to_submission_df(cand, chunk_ids))
        if cand.empty:
            write_submission(matches, candidate_frames, all_s1_ids, paths)
            continue

        feature_df = build_pair_features(cand, source1, source2, source3, id_col="id")
        feature_df["gbm_score"] = gbm.predict_proba(feature_df)
        ce_score = None
        if cross_encoder is not None:
            ce_score = cross_encoder.predict_proba(feature_df, s1_lookup, t_lookups)
        scored = ensemble_scores(
            feature_df,
            gbm_proba=feature_df["gbm_score"],
            cross_encoder_proba=ce_score,
            method=ens_cfg["method"] if ce_score is not None else "gbm_only",
            gbm_weight=ens_cfg["gbm_weight"],
            cross_encoder_weight=ens_cfg["cross_encoder_weight"],
        )
        matches.update(predictions_to_matches(
            scored, threshold=threshold, score_col="final_score",
            max_matches_per_source1=cfg["inference"]["max_matches_per_source1"],
            require_min_margin=cfg["inference"]["require_min_margin"],
        ))
        logger.info("Scored source1 %d-%d of %d (%d pairs) in %.0fs",
                    start, min(start + chunk, len(all_s1_ids)), len(all_s1_ids), len(cand), time.time() - t0)
        # Write valid submission files after every chunk: source1 ids not yet
        # processed get empty lists, so a partial run is still submittable.
        write_submission(matches, candidate_frames, all_s1_ids, paths)

    n_with_match = sum(1 for v in matches.values() if v)
    logger.info("%d / %d source1 records received at least one match.", n_with_match, len(all_s1_ids))

    write_submission(matches, candidate_frames, all_s1_ids, paths)
    logger.info("Wrote final predictions -> %s and %s",
                paths["final_predictions"], paths["submission_candidate_pairs"])


def write_submission(matches: dict, candidate_frames: list, all_s1_ids: list, paths: dict) -> None:
    """Write both challenge files; ids without a candidate row yet get empty lists."""
    cands = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame(
        columns=["source1_entity_id", "candidate_entity_ids"])
    done = set(cands["source1_entity_id"])
    rest = [sid for sid in all_s1_ids if sid not in done]
    cands = pd.concat([cands, pd.DataFrame({"source1_entity_id": rest, "candidate_entity_ids": ""})],
                      ignore_index=True)
    ensure_dir(paths["submission_candidate_pairs"])
    cands.to_csv(paths["submission_candidate_pairs"], sep="\t", index=False)
    ensure_dir(paths["final_predictions"])
    matches_to_submission_df(matches, all_s1_ids).to_csv(paths["final_predictions"], sep="\t", index=False)


if __name__ == "__main__":
    main()
