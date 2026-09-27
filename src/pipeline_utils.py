"""
Shared utilities used by both train.py and predict.py: config/data loading,
ground-truth reshaping (wide comma-separated -> long per-target-source
format), and the final ensemble scoring function that combines GBM and
cross-encoder probabilities into one `final_score` per candidate pair.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _resolve_column(df: pd.DataFrame, preferred: str, aliases: tuple[str, ...] = ()) -> str | None:
    candidates = [preferred, *aliases]
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _rename_common_columns(df: pd.DataFrame, *, id_col: str = "id") -> pd.DataFrame:
    rename_map = {}
    resolved_id = _resolve_column(df, id_col, ("entity_id",))
    if resolved_id and resolved_id != id_col:
        rename_map[resolved_id] = id_col

    for preferred, aliases in {
        "company_name": ("business_name",),
        "address": ("business_address",),
        "city": ("city_name",),
        "zip": ("postal_code", "zipcode"),
    }.items():
        resolved = _resolve_column(df, preferred, aliases)
        if resolved and resolved != preferred:
            rename_map[resolved] = preferred

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_source_tsv(path: str, id_col: str = "id") -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = _rename_common_columns(df, id_col=id_col)
    if id_col not in df.columns:
        raise ValueError(f"Expected id column '{id_col}' not found in {path}. Found: {list(df.columns)}")
    return df


def load_ground_truth_wide(path: str) -> pd.DataFrame:
    """Load train_ground_truth.tsv as-is: columns [source1_id, matched_ids]."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    rename_map = {}
    src_id = _resolve_column(df, "source1_id", ("source1_entity_id",))
    if src_id and src_id != "source1_id":
        rename_map[src_id] = "source1_id"
    matched = _resolve_column(df, "matched_ids", ("matched_entity_ids",))
    if matched and matched != "matched_ids":
        rename_map[matched] = "matched_ids"
    if rename_map:
        df = df.rename(columns=rename_map)
    if "matched_ids" not in df.columns:
        raise ValueError(f"Ground truth file {path} missing 'matched_ids' column. Found: {list(df.columns)}")
    df["matched_ids"] = df["matched_ids"].fillna("")
    return df


def ground_truth_wide_to_long(
    gt_wide: pd.DataFrame,
    source2_ids: set,
    source3_ids: set,
) -> pd.DataFrame:
    """
    Explode the wide ground-truth format (one row per source1_id, comma
    separated matched ids possibly spanning BOTH source2 and source3) into
    long format: one row per (source1_id, target_id, target_source).

    `target_source` is inferred by checking id membership against each
    source's known id set — this assumes source2_id and source3_id spaces
    don't collide. If they can collide in your actual data, prefix ids at
    load time (e.g. "s2_"+id) before calling this function so membership is
    unambiguous.
    """
    rows = []
    for _, row in gt_wide.iterrows():
        sid = row["source1_id"]
        ids = [x.strip() for x in row["matched_ids"].split(",") if x.strip()]
        for tid in ids:
            if tid in source2_ids:
                rows.append((sid, tid, "source2"))
            elif tid in source3_ids:
                rows.append((sid, tid, "source3"))
            else:
                logger.warning(
                    "Ground truth id %s (for source1_id=%s) not found in either "
                    "source2 or source3 id sets; skipping.", tid, sid
                )
    return pd.DataFrame(rows, columns=["source1_id", "target_id", "target_source"])


def ground_truth_long_to_dict(gt_long: pd.DataFrame) -> dict:
    """source1_id -> set of ALL matched target ids (across both target
    sources) — the format `macro_f_beta` / `optimize_threshold` expect."""
    out: dict[str, set] = {}
    for sid, group in gt_long.groupby("source1_id"):
        out[sid] = set(group["target_id"].astype(str).tolist())
    return out


def ensemble_scores(
    feature_df: pd.DataFrame,
    gbm_proba: "pd.Series | None",
    cross_encoder_proba: "pd.Series | None",
    method: str = "weighted_average",
    gbm_weight: float = 0.5,
    cross_encoder_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Combine GBM and (optional) cross-encoder probabilities into a single
    `final_score` column on a copy of `feature_df`. Falls back gracefully
    when the cross-encoder was disabled (GBM-only pipeline).
    """
    out = feature_df.copy()

    if gbm_proba is not None:
        out["gbm_score"] = gbm_proba
    if cross_encoder_proba is not None:
        out["cross_encoder_score"] = cross_encoder_proba

    has_gbm = "gbm_score" in out.columns
    has_ce = "cross_encoder_score" in out.columns

    if method == "gbm_only" or (has_gbm and not has_ce):
        out["final_score"] = out["gbm_score"]
    elif method == "cross_encoder_only" or (has_ce and not has_gbm):
        out["final_score"] = out["cross_encoder_score"]
    elif method == "weighted_average" and has_gbm and has_ce:
        total_w = gbm_weight + cross_encoder_weight
        out["final_score"] = (
            out["gbm_score"] * gbm_weight + out["cross_encoder_score"] * cross_encoder_weight
        ) / total_w
    elif method == "stacked" and has_gbm and has_ce:
        # Simple, dependency-free "stacking": geometric mean, which penalizes
        # cases where the two models strongly disagree more than a plain
        # average would -- a useful extra precision guard.
        out["final_score"] = (out["gbm_score"] * out["cross_encoder_score"]) ** 0.5
    else:
        raise ValueError(
            f"Cannot compute ensemble with method='{method}': "
            f"has_gbm={has_gbm}, has_ce={has_ce}"
        )

    return out


def ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
