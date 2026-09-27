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
   used in training (Module 2). Note: a fresh CandidateGenerator is used
   (TF-IDF vocab is refit on the test corpus) since test entities are
   disjoint from train entities and reusing a train-fit vocabulary would
   only shrink candidate coverage on unseen vocabulary, not help it.
4. Build the same feature set (Module 3a) -- NO labels attached (test has
   none).
5. Score with the saved GBM (+ cross-encoder if available) and ensemble.
6. Apply the saved decision threshold to produce final match lists,
   correctly emitting an empty string for singleton/no-match source1 ids.
7. Write `matching_results.tsv`.
"""

from __future__ import annotations

import argparse
import logging

from src.preprocessing import preprocess_dataframe
from src.blocking import generate_candidate_pairs
from src.features import build_pair_features
from src.models import GBMMatcher, CrossEncoderMatcher
from src.evaluation.metrics import predictions_to_matches, matches_to_submission_df, load_threshold
from src.pipeline_utils import load_config, load_source_tsv, ensemble_scores, ensure_dir

logger = logging.getLogger("predict")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference for the entity resolution pipeline.")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--gbm-only", action="store_true",
                    help="Ignore a saved cross-encoder even if present (faster, GBM-only scoring).")
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

    # ---------------------------------------------------------------
    # 3. Blocking / candidate generation (Module 2)
    # ---------------------------------------------------------------
    logger.info("Generating candidate pairs for test set...")
    candidate_pairs = generate_candidate_pairs(source1, source2, source3, cfg)
    ensure_dir(paths["candidate_pairs_test"])
    candidate_pairs.to_csv(paths["candidate_pairs_test"], sep="\t", index=False)
    logger.info("Wrote %d candidate pairs -> %s", len(candidate_pairs), paths["candidate_pairs_test"])

    # ---------------------------------------------------------------
    # 4. Feature engineering (Module 3a) -- no labels at inference
    # ---------------------------------------------------------------
    logger.info("Building pairwise features...")
    feature_df = build_pair_features(candidate_pairs, source1, source2, source3, id_col="id")
    ensure_dir(paths["features_test"])
    feature_df.to_parquet(paths["features_test"], index=False)
    logger.info("Feature matrix: %d rows", len(feature_df))

    if feature_df.empty:
        logger.warning("No candidate pairs survived blocking -- writing all-singleton predictions.")
        submission = matches_to_submission_df({}, source1["id"].tolist())
        ensure_dir(paths["final_predictions"])
        submission.to_csv(paths["final_predictions"], sep="\t", index=False)
        logger.info("Wrote %s", paths["final_predictions"])
        return

    # ---------------------------------------------------------------
    # 5. Score with saved models
    # ---------------------------------------------------------------
    logger.info("Loading saved GBM model and scoring...")
    gbm = GBMMatcher.load(paths["gbm_model_out"])
    feature_df["gbm_score"] = gbm.predict_proba(feature_df)

    cross_encoder_score = None
    use_ce = cfg["cross_encoder"]["enabled"] and not args.gbm_only
    if use_ce:
        try:
            logger.info("Loading saved cross-encoder model and scoring...")
            cross_encoder = CrossEncoderMatcher.load(paths["cross_encoder_model_out"])
            s1_lookup = source1.set_index("id")[["clean_name", "clean_address"]]
            s2_lookup = source2.set_index("id")[["clean_name", "clean_address"]]
            s3_lookup = source3.set_index("id")[["clean_name", "clean_address"]]
            t_lookups = {"source2": s2_lookup, "source3": s3_lookup}
            cross_encoder_score = cross_encoder.predict_proba(feature_df, s1_lookup, t_lookups)
        except (OSError, FileNotFoundError) as e:
            logger.warning("Could not load cross-encoder model (%s); falling back to GBM-only scoring.", e)

    ens_cfg = cfg["ensemble"]
    scored = ensemble_scores(
        feature_df,
        gbm_proba=feature_df["gbm_score"],
        cross_encoder_proba=cross_encoder_score,
        method=ens_cfg["method"] if cross_encoder_score is not None else "gbm_only",
        gbm_weight=ens_cfg["gbm_weight"],
        cross_encoder_weight=ens_cfg["cross_encoder_weight"],
    )

    # ---------------------------------------------------------------
    # 6. Apply saved threshold -> final match lists
    # ---------------------------------------------------------------
    # min_decision_threshold is enforced again here (defense in depth) even
    # though train.py already enforces it when the threshold was selected --
    # see load_threshold's docstring.
    min_floor = cfg["evaluation"]["min_decision_threshold"]
    threshold = load_threshold(paths["threshold_out"], min_decision_threshold=min_floor)
    logger.info(
        "Applying decision threshold=%.4f (precision floor=%.2f) -- only candidate pairs "
        "scoring strictly above this become predicted matches; every other source1 id maps "
        "to an empty match list.",
        threshold, min_floor,
    )

    matches = predictions_to_matches(
        scored, threshold=threshold, score_col="final_score",
        max_matches_per_source1=cfg["inference"]["max_matches_per_source1"],
        require_min_margin=cfg["inference"]["require_min_margin"],
    )

    n_with_match = sum(1 for v in matches.values() if v)
    logger.info("%d / %d source1 records received at least one match.",
                n_with_match, source1["id"].nunique())

    # ---------------------------------------------------------------
    # 7. Write submission
    # ---------------------------------------------------------------
    submission = matches_to_submission_df(matches, source1["id"].tolist())
    ensure_dir(paths["final_predictions"])
    submission.to_csv(paths["final_predictions"], sep="\t", index=False)
    logger.info("Wrote final predictions -> %s", paths["final_predictions"])


if __name__ == "__main__":
    main()
