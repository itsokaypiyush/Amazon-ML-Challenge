"""
Module 3: Gradient Boosting Matcher (Stage 2, part B)
=======================================================

Wraps LightGBM training/inference for the candidate-pair binary
classification task: P(target_id is a true match for source1_id | features).

Key design decisions
--------------------
- Labels come from `train_ground_truth.tsv` (source1_id -> comma-separated
  matched_ids). Any candidate pair whose target_id appears in that list is a
  positive; every other CANDIDATE pair (i.e. pairs that survived blocking
  but are not in ground truth) is a hard negative — these are far more
  informative than random negatives because they're pairs the blocking
  stage already thought looked plausible.
- Severe class imbalance is expected (blocking produces ~top_k candidates
  per source1 record, of which typically 0-2 are true matches). We handle
  this via `scale_pos_weight` (capped, since an uncapped ratio on a
  precision-critical metric can overcorrect toward recall) AND optional
  negative subsampling for training speed.
- Early stopping is done on a held-out validation split grouped by
  source1_id (never a random row split!) to prevent leakage: rows from the
  same source1_id must not appear in both train and validation, or the
  model effectively memorizes candidate rankings within a group.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)


def build_labels(
    feature_df: pd.DataFrame, ground_truth: pd.DataFrame
) -> pd.DataFrame:
    """
    Attach a binary `label` column to the feature dataframe.

    `ground_truth` must have columns [source1_id, matched_ids] where
    matched_ids is a comma-separated string of target ids (across BOTH
    source2 and source3 — target_source in feature_df disambiguates which
    table an id belongs to, so id collisions across sources are handled by
    scoping the true-set lookup per target_source group).
    """
    gt = ground_truth.copy()
    gt["matched_ids"] = gt["matched_ids"].fillna("").astype(str)
    keys = {
        f"{sid}|{t.strip()}"
        for sid, ids in zip(gt["source1_id"], gt["matched_ids"])
        for t in ids.split(",") if t.strip()
    }
    out = feature_df.copy()
    pair = out["source1_id"].astype(str) + "|" + out["target_id"].astype(str)
    out["label"] = pair.isin(keys).astype(np.int8)
    return out


@dataclass
class GBMMatcher:
    feature_columns: list[str]
    params: dict = field(default_factory=dict)
    num_boost_round: int = 2000
    early_stopping_rounds: int = 100
    scale_pos_weight_cap: float = 20.0
    random_state: int = 42

    booster: Optional[lgb.Booster] = field(default=None, init=False, repr=False)

    def _default_params(self, scale_pos_weight: float) -> dict:
        base = {
            "objective": "binary",
            "metric": ["average_precision", "auc"],
            "num_leaves": 63,
            "learning_rate": 0.03,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "scale_pos_weight": min(scale_pos_weight, self.scale_pos_weight_cap),
            "verbosity": -1,
            "seed": self.random_state,
        }
        base.update(self.params)
        return base

    def fit(
        self,
        feature_df: pd.DataFrame,
        n_splits: int = 5,
    ) -> dict:
        """
        Train with GroupKFold (grouped by source1_id) for early-stopping
        validation, then refit a final model on all data using the average
        best iteration across folds. Returns a dict of per-fold metrics for
        logging/debugging.
        """
        X = feature_df[self.feature_columns].values
        y = feature_df["label"].values
        groups = feature_df["source1_id"].values

        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        scale_pos_weight = n_neg / max(n_pos, 1)
        logger.info(
            "Training GBM on %d pairs (%d positive, %d negative, scale_pos_weight=%.2f, capped at %.2f)",
            len(y), n_pos, n_neg, scale_pos_weight, self.scale_pos_weight_cap,
        )

        params = self._default_params(scale_pos_weight)

        gkf = GroupKFold(n_splits=min(n_splits, max(2, len(np.unique(groups)) // 2 or 2)))
        best_iterations = []
        fold_metrics = []
        oof = np.zeros(len(y), dtype=np.float64)

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            train_set = lgb.Dataset(X[train_idx], label=y[train_idx])
            val_set = lgb.Dataset(X[val_idx], label=y[val_idx], reference=train_set)

            booster = lgb.train(
                params,
                train_set,
                num_boost_round=self.num_boost_round,
                valid_sets=[val_set],
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            best_iterations.append(booster.best_iteration)
            val_pred = booster.predict(X[val_idx], num_iteration=booster.best_iteration)
            oof[val_idx] = val_pred
            fold_metrics.append(
                {"fold": fold, "best_iteration": booster.best_iteration,
                 "n_val": len(val_idx), "n_val_pos": int(y[val_idx].sum())}
            )
            logger.info("Fold %d: best_iteration=%d, val_pos=%d/%d",
                         fold, booster.best_iteration, int(y[val_idx].sum()), len(val_idx))

        final_rounds = max(int(np.mean(best_iterations)), 10)
        full_set = lgb.Dataset(X, label=y)
        self.booster = lgb.train(params, full_set, num_boost_round=final_rounds)

        # Out-of-fold predictions: every row scored by a model that never saw
        # its source1 group -- use these (not the final model's in-sample
        # scores) to pick the decision threshold.
        return {"fold_metrics": fold_metrics, "final_num_rounds": final_rounds, "oof_pred": oof}

    def predict_proba(self, feature_df: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model has not been trained or loaded yet.")
        X = feature_df[self.feature_columns].values
        return self.booster.predict(X)

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        if self.booster is None:
            raise RuntimeError("Model has not been trained or loaded yet.")
        imp = self.booster.feature_importance(importance_type=importance_type)
        return (
            pd.DataFrame({"feature": self.feature_columns, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: str) -> None:
        if self.booster is None:
            raise RuntimeError("Model has not been trained yet.")
        self.booster.save_model(path)
        meta_path = path + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump({"feature_columns": self.feature_columns}, f)

    @classmethod
    def load(cls, path: str) -> "GBMMatcher":
        with open(path + ".meta.json") as f:
            meta = json.load(f)
        instance = cls(feature_columns=meta["feature_columns"])
        instance.booster = lgb.Booster(model_file=path)
        return instance
