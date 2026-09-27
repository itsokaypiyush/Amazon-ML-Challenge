"""
Module 4: Precision-Weighted Post-Processing & Evaluation (Stage 3)
=====================================================================

Implements:
  1. Macro F-beta (beta=0.5) scoring, computed PER source1 entity and then
     averaged — this is what "macro" means here: an entity with 1 true
     match is weighted the same as an entity with 5, so the metric doesn't
     let performance on a few high-fan-out entities dominate the score.
  2. Threshold search that directly optimizes this macro metric on a
     held-out validation split (NOT the standard "optimal F1 threshold"
     trick, which under-weights precision relative to what F0.5 demands).
  3. Safe conversion of per-pair probabilities into the final match-list
     format, including correct handling of singletons (source1 entities
     with zero matches above threshold map to an empty list, never to
     omission or to an arbitrary low-confidence guess).

Why precision weighting matters mechanically: with beta=0.5,
F_beta = (1+beta^2) * P * R / (beta^2 * P + R). At beta=0.5, a false
positive that drops precision from 1.0 to 0.5 costs far more F-score than
a missed true match that drops recall by the same absolute amount. The
threshold search below is calibrated directly against this, not proxied
through F1 or AUPRC.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd


def _f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta2 = beta ** 2
    denom = beta2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def macro_f_beta(
    predicted: dict[str, set],
    ground_truth: dict[str, set],
    beta: float = 0.5,
) -> dict:
    """
    Compute macro-averaged F-beta over the union of source1 ids appearing in
    either `predicted` or `ground_truth`.

    Both dicts map source1_id -> set of matched target ids. A source1_id
    missing from either dict is treated as predicting/truthing an empty set
    (a legitimate singleton), so callers do not need to pre-fill every id.

    Returns per-entity-averaged precision, recall, f_beta, plus the count of
    entities scored, for transparency/debugging.
    """
    all_ids = set(predicted.keys()) | set(ground_truth.keys())
    if not all_ids:
        return {"precision": 0.0, "recall": 0.0, "f_beta": 0.0, "n_entities": 0}

    precisions, recalls, f_betas = [], [], []
    for sid in all_ids:
        pred_set = predicted.get(sid, set())
        true_set = ground_truth.get(sid, set())

        if not pred_set and not true_set:
            # Correctly predicted singleton: perfect score for this entity.
            precisions.append(1.0)
            recalls.append(1.0)
            f_betas.append(1.0)
            continue

        tp = len(pred_set & true_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(true_set) if true_set else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f_betas.append(_f_beta(precision, recall, beta))

    return {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f_beta": float(np.mean(f_betas)),
        "n_entities": len(all_ids),
    }


def predictions_to_matches(
    scored_pairs: pd.DataFrame,
    threshold: float,
    score_col: str = "final_score",
    max_matches_per_source1: int = 5,
    require_min_margin: float = 0.0,
) -> dict[str, set]:
    """
    Convert a scored candidate-pairs dataframe into the final
    source1_id -> {matched target_ids} mapping.

    Decision rule:
      - Keep only pairs with score > threshold.
      - Cap at `max_matches_per_source1` highest-scoring survivors per
        source1_id (protects against a rare pathological case where many
        near-duplicate candidates all clear the threshold; true fan-out
        beyond this is uncommon in vendor-matching problems but the cap is
        configurable).
      - `require_min_margin` (optional): if set > 0, any candidate beyond
        the top-1 must be within `margin` of the top score to be kept, as
        an extra precision guard for ambiguous multi-match cases.
      - source1_ids with zero surviving pairs are simply absent from the
        output dict (equivalent to an empty match list == correct
        singleton handling); callers must fill these back in explicitly
        when building the full submission (see `matches_to_submission_df`).
    """
    kept = scored_pairs[scored_pairs[score_col] > threshold].copy()
    if kept.empty:
        return {}

    kept = kept.sort_values(["source1_id", score_col], ascending=[True, False])

    matches: dict[str, set] = {}
    for sid, group in kept.groupby("source1_id"):
        top_score = group[score_col].iloc[0]
        rows = group
        if require_min_margin > 0.0:
            rows = group[group[score_col] >= top_score - require_min_margin]
        rows = rows.head(max_matches_per_source1)
        matches[sid] = set(rows["target_id"].astype(str).tolist())

    return matches


def optimize_threshold(
    scored_pairs: pd.DataFrame,
    ground_truth: dict[str, set],
    all_source1_ids: list[str],
    score_col: str = "final_score",
    beta: float = 0.5,
    search_space: tuple[float, float] = (0.5, 0.99),
    steps: int = 200,
    max_matches_per_source1: int = 5,
    min_decision_threshold: float = 0.85,
) -> dict:
    """
    Grid-search a decision threshold over `search_space` that maximizes
    macro F-beta on a validation set, evaluated against ALL source1 ids
    (including those with zero candidate pairs at all, who must count as
    correct singletons if ground truth also has them empty).

    `min_decision_threshold` is a HARD FLOOR on the returned threshold: the
    full range in `search_space` is still scored (so the returned `curve`
    is complete for diagnostics/plotting), but `best` is selected only from
    among thresholds >= `min_decision_threshold`. This is deliberate and not
    just a search-space restriction — with beta=0.5, precision dominates the
    metric, and false merges carry a heavy penalty per the challenge spec, so
    we refuse to let a threshold that merely scored best on a (possibly
    small, possibly unrepresentative) validation split silently authorize
    low-confidence matches in production. If every point above the floor
    scores worse than points below it, that is itself a signal the model
    needs more/better features or more positive training examples — not a
    reason to lower the floor.

    Returns `best`, `curve` (full, unrestricted, for diagnostics), and
    `best_unrestricted` (the true argmax ignoring the floor) so callers can
    see how much macro F-beta was traded away to enforce precision safety.
    """
    thresholds = np.linspace(search_space[0], search_space[1], steps)
    curve = []
    best_unrestricted = {"threshold": search_space[0], "f_beta": -1.0}
    best = {"threshold": max(search_space[0], min_decision_threshold), "f_beta": -1.0}

    # Precompute ground truth for every source1 id, including ones with an
    # empty true set, so the metric penalizes any false positives created
    # for entities that should be singletons.
    gt_full = {sid: ground_truth.get(sid, set()) for sid in all_source1_ids}

    for t in thresholds:
        matches = predictions_to_matches(
            scored_pairs, threshold=t, score_col=score_col,
            max_matches_per_source1=max_matches_per_source1,
        )
        pred_full = {sid: matches.get(sid, set()) for sid in all_source1_ids}
        metrics = macro_f_beta(pred_full, gt_full, beta=beta)
        metrics["threshold"] = float(t)
        curve.append(metrics)

        if metrics["f_beta"] > best_unrestricted["f_beta"]:
            best_unrestricted = metrics
        if t >= min_decision_threshold and metrics["f_beta"] > best["f_beta"]:
            best = metrics

    if best["f_beta"] < 0:
        # No grid point fell at/above the floor (e.g. search_space's upper
        # bound is itself below min_decision_threshold) -- fall back to
        # scoring exactly at the floor so we still return a valid, floor-
        # respecting result rather than a sentinel.
        matches = predictions_to_matches(
            scored_pairs, threshold=min_decision_threshold, score_col=score_col,
            max_matches_per_source1=max_matches_per_source1,
        )
        pred_full = {sid: matches.get(sid, set()) for sid in all_source1_ids}
        best = macro_f_beta(pred_full, gt_full, beta=beta)
        best["threshold"] = float(min_decision_threshold)

    return {"best": best, "best_unrestricted": best_unrestricted, "curve": curve,
            "min_decision_threshold": min_decision_threshold}


def save_threshold(threshold_result: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(
            {
                "threshold": threshold_result["best"]["threshold"],
                "validation_metrics": threshold_result["best"],
                "min_decision_threshold_floor": threshold_result.get("min_decision_threshold"),
                "unrestricted_optimum": threshold_result.get("best_unrestricted"),
            },
            f, indent=2,
        )


def load_threshold(path: str, min_decision_threshold: Optional[float] = None) -> float:
    """
    Load the persisted decision threshold. If `min_decision_threshold` is
    passed, it is enforced here too (defense in depth): even if
    `decision_threshold.json` were ever hand-edited or came from an older
    training run made before the floor existed, inference will never apply
    a threshold below the caller-specified precision floor.
    """
    with open(path) as f:
        data = json.load(f)
    threshold = float(data["threshold"])
    if min_decision_threshold is not None and threshold < min_decision_threshold:
        threshold = min_decision_threshold
    return threshold


def matches_to_submission_df(
    matches: dict[str, set], all_source1_ids: list[str]
) -> pd.DataFrame:
    """
    Build the final `matching_results.tsv`-ready dataframe. Every source1_id
    in `all_source1_ids` gets exactly one row; ids with no surviving matches
    get an empty string (never omitted, never NaN) in `matched_ids`, per the
    competition's required singleton-handling format.
    """
    rows = []
    for sid in all_source1_ids:
        matched = sorted(matches.get(sid, set()))
        rows.append({"source1_id": sid, "matched_ids": ",".join(matched)})
    return pd.DataFrame(rows, columns=["source1_id", "matched_ids"])
