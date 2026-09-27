#!/usr/bin/env python3
"""
End-to-end training CLI.

Usage
-----
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --skip-cross-encoder
    python train.py --config configs/config.yaml --log-level DEBUG

Pipeline stages
---------------
1. Load raw source1/source2/source3 + train_ground_truth.
2. Preprocess & normalize all three sources (Module 1).
3. Generate candidate pairs via multi-strategy blocking (Module 2).
   -> checks blocking recall against ground truth before continuing;
      warns loudly if recall is below `min_blocking_recall_target`.
4. Build features + attach labels from ground truth (Module 3a).
5. Train GBM matcher with GroupKFold early stopping (Module 3a).
6. Optionally train cross-encoder matcher (Module 3b).
7. Ensemble scores, optimize decision threshold for macro F0.5 (Module 4).
8. Persist: GBM model, cross-encoder model, decision threshold, and the
   intermediate candidate_pairs.tsv / features_train.parquet artifacts.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from src.preprocessing import preprocess_dataframe
from src.blocking import generate_candidate_pairs
from src.blocking.candidate_generation import evaluate_blocking_recall
from src.features import build_pair_features
from src.models import GBMMatcher, CrossEncoderMatcher
from src.models.gbm_model import build_labels
from src.evaluation import optimize_threshold
from src.evaluation.metrics import save_threshold
from src.pipeline_utils import (
    load_config, load_source_tsv, load_ground_truth_wide,
    ground_truth_wide_to_long, ground_truth_long_to_dict,
    ensemble_scores, ensure_dir,
)

logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the entity resolution pipeline.")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--skip-cross-encoder", action="store_true",
                    help="Force GBM-only training regardless of config setting.")
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
    # 1. Load raw data
    # ---------------------------------------------------------------
    logger.info("Loading raw source tables...")
    source1_raw = load_source_tsv(paths["source1"], id_col)
    source2_raw = load_source_tsv(paths["source2"], id_col)
    source3_raw = load_source_tsv(paths["source3"], id_col)
    gt_wide = load_ground_truth_wide(paths["train_ground_truth"])
    logger.info("Loaded: source1=%d, source2=%d, source3=%d, ground_truth=%d",
                len(source1_raw), len(source2_raw), len(source3_raw), len(gt_wide))

    # ---------------------------------------------------------------
    # 2. Preprocess (Module 1)
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
    logger.info("Generating candidate pairs via multi-strategy blocking...")
    candidate_pairs = generate_candidate_pairs(source1, source2, source3, cfg)
    ensure_dir(paths["candidate_pairs_train"])
    candidate_pairs.to_csv(paths["candidate_pairs_train"], sep="\t", index=False)
    logger.info("Wrote %d candidate pairs -> %s", len(candidate_pairs), paths["candidate_pairs_train"])

    gt_long = ground_truth_wide_to_long(gt_wide, set(source2["id"]), set(source3["id"]))
    recall_stats = evaluate_blocking_recall(candidate_pairs, gt_long)
    logger.info("Blocking recall on training ground truth: %.4f (%d/%d true pairs found)",
                recall_stats["recall"], recall_stats["n_found"], recall_stats["n_true_pairs"])
    target_recall = cfg["blocking"]["min_blocking_recall_target"]
    if recall_stats["recall"] < target_recall:
        logger.warning(
            "Blocking recall %.4f is BELOW target %.4f -- Stage 2/3 can never recover "
            "true matches dropped at this stage. Consider raising top_k or loosening blocking keys.",
            recall_stats["recall"], target_recall,
        )

    # ---------------------------------------------------------------
    # 4. Feature engineering + labels (Module 3a)
    # ---------------------------------------------------------------
    logger.info("Building pairwise features...")
    feature_df = build_pair_features(candidate_pairs, source1, source2, source3, id_col="id")
    labeled_df = build_labels(feature_df, gt_wide)
    ensure_dir(paths["features_train"])
    labeled_df.to_parquet(paths["features_train"], index=False)
    logger.info("Feature matrix: %d rows, %d positive labels", len(labeled_df), int(labeled_df["label"].sum()))

    non_feature_cols = {"source1_id", "target_id", "target_source", "label"}
    feature_columns = [c for c in labeled_df.columns if c not in non_feature_cols]

    # ---------------------------------------------------------------
    # 5. Train GBM (Module 3a)
    # ---------------------------------------------------------------
    logger.info("Training LightGBM matcher...")
    gbm_cfg = cfg["gbm"]
    gbm = GBMMatcher(
        feature_columns=feature_columns,
        num_boost_round=gbm_cfg["n_estimators"],
        early_stopping_rounds=gbm_cfg["early_stopping_rounds"],
        scale_pos_weight_cap=gbm_cfg["scale_pos_weight_cap"],
        random_state=gbm_cfg["random_state"],
        params={
            "num_leaves": gbm_cfg["num_leaves"],
            "learning_rate": gbm_cfg["learning_rate"],
        },
    )
    gbm_train_metrics = gbm.fit(labeled_df, n_splits=cfg["evaluation"]["cv_folds"])
    ensure_dir(paths["gbm_model_out"])
    gbm.save(paths["gbm_model_out"])
    logger.info("Saved GBM model -> %s", paths["gbm_model_out"])
    logger.info("Top 10 features by gain:\n%s", gbm.feature_importance().head(10).to_string(index=False))

    labeled_df["gbm_score"] = gbm.predict_proba(labeled_df)

    # ---------------------------------------------------------------
    # 6. Train cross-encoder (Module 3b, optional)
    # ---------------------------------------------------------------
    ce_enabled = cfg["cross_encoder"]["enabled"] and not args.skip_cross_encoder
    if ce_enabled:
        logger.info("Training transformer cross-encoder matcher...")
        ce_cfg = cfg["cross_encoder"]
        cross_encoder = CrossEncoderMatcher(
            base_model=ce_cfg["base_model"],
            max_length=ce_cfg["max_length"],
            batch_size=ce_cfg["batch_size"],
            learning_rate=ce_cfg["learning_rate"],
            epochs=ce_cfg["epochs"],
            warmup_ratio=ce_cfg["warmup_ratio"],
            output_dir=paths["cross_encoder_model_out"],
        )

        s1_lookup = source1.set_index("id")[["clean_name", "clean_address"]]
        s2_lookup = source2.set_index("id")[["clean_name", "clean_address"]]
        s3_lookup = source3.set_index("id")[["clean_name", "clean_address"]]
        t_lookups = {"source2": s2_lookup, "source3": s3_lookup}

        # Negative sampling for cross-encoder training speed: keep all
        # positives, subsample negatives at the configured ratio.
        pos = labeled_df[labeled_df["label"] == 1]
        neg = labeled_df[labeled_df["label"] == 0]
        neg_ratio = ce_cfg["negative_sampling_ratio"]
        n_neg_keep = min(len(neg), len(pos) * neg_ratio) if len(pos) > 0 else len(neg)
        neg_sampled = neg.sample(n=n_neg_keep, random_state=42) if n_neg_keep < len(neg) else neg
        ce_train_df = pd.concat([pos, neg_sampled], ignore_index=True)
        logger.info("Cross-encoder training set: %d positive, %d negative (sampled from %d)",
                    len(pos), len(neg_sampled), len(neg))

        ce_metrics = cross_encoder.fit(ce_train_df, s1_lookup, t_lookups)
        logger.info("Cross-encoder validation metrics: %s", ce_metrics)

        ensure_dir(paths["cross_encoder_model_out"] + "/model.safetensors")
        cross_encoder.save(paths["cross_encoder_model_out"])
        logger.info("Saved cross-encoder -> %s", paths["cross_encoder_model_out"])

        labeled_df["cross_encoder_score"] = cross_encoder.predict_proba(labeled_df, s1_lookup, t_lookups)
    else:
        logger.info("Cross-encoder training skipped (disabled in config or --skip-cross-encoder passed).")

    # ---------------------------------------------------------------
    # 7. Ensemble + threshold optimization (Module 4)
    # ---------------------------------------------------------------
    logger.info("Computing ensemble scores and optimizing decision threshold for macro F%.1f...",
                cfg["evaluation"]["beta"])
    ens_cfg = cfg["ensemble"]
    scored = ensemble_scores(
        labeled_df,
        gbm_proba=labeled_df["gbm_score"],
        cross_encoder_proba=labeled_df.get("cross_encoder_score"),
        method=ens_cfg["method"] if ce_enabled else "gbm_only",
        gbm_weight=ens_cfg["gbm_weight"],
        cross_encoder_weight=ens_cfg["cross_encoder_weight"],
    )

    gt_dict = ground_truth_long_to_dict(gt_long)
    all_s1_ids = source1["id"].tolist()
    eval_cfg = cfg["evaluation"]
    threshold_result = optimize_threshold(
        scored, gt_dict, all_s1_ids,
        beta=eval_cfg["beta"],
        search_space=tuple(eval_cfg["threshold_search_space"]),
        steps=eval_cfg["threshold_search_steps"],
        max_matches_per_source1=cfg["inference"]["max_matches_per_source1"],
        min_decision_threshold=eval_cfg["min_decision_threshold"],
    )
    best = threshold_result["best"]
    unrestricted = threshold_result["best_unrestricted"]
    logger.info(
        "Selected threshold=%.4f (floor=%.2f) -> macro P=%.4f, R=%.4f, F%.1f=%.4f (n_entities=%d)",
        best["threshold"], eval_cfg["min_decision_threshold"],
        best["precision"], best["recall"], eval_cfg["beta"], best["f_beta"], best["n_entities"],
    )
    if unrestricted["threshold"] < eval_cfg["min_decision_threshold"]:
        logger.warning(
            "Unrestricted validation optimum was threshold=%.4f (F%.1f=%.4f), BELOW the %.2f "
            "precision floor -- %.4f F%.1f was traded away to enforce it. This usually means the "
            "model isn't confidently separating true/false matches yet; consider more/better "
            "features, more labeled data, or reviewing feature_importance() before trusting this run.",
            unrestricted["threshold"], eval_cfg["beta"], unrestricted["f_beta"],
            eval_cfg["min_decision_threshold"], unrestricted["f_beta"] - best["f_beta"], eval_cfg["beta"],
        )

    ensure_dir(paths["threshold_out"])
    save_threshold(threshold_result, paths["threshold_out"])
    logger.info("Saved decision threshold -> %s", paths["threshold_out"])

    logger.info("Training pipeline complete.")


if __name__ == "__main__":
    main()
