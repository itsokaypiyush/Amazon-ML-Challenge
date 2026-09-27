"""
Module 3: Feature Engineering (Stage 2, part A)
=================================================

Turns each (source1_id, target_id) candidate pair into a fixed-width feature
vector for the LightGBM classifier. Features are computed SEPARATELY on name
and address fields (never concatenated first) because a good name-match with
a bad address-match is a very different signal than the reverse, and the GBM
needs both signals to learn that distinction.

All string-distance functions come from `rapidfuzz` (C++ backed, fast enough
for candidate-pair-scale computation, unlike pure-python `python-Levenshtein`
alternatives) except Monge-Elkan, which rapidfuzz doesn't ship natively, so
we implement it directly on top of rapidfuzz's token-level primitives.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance

FEATURE_COLUMNS: list[str] = []  # populated at import time below


def _safe_str(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x)


# -----------------------------------------------------------------------------
# Core string-distance metrics (each returns a 0-100 or 0-1 similarity score)
# -----------------------------------------------------------------------------

def jaro_winkler_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return distance.JaroWinkler.similarity(a, b)


def levenshtein_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def token_sort_ratio(a: str, b: str) -> float:
    """Robust to word-order differences: 'Jones Smith Co' vs 'Smith Jones Co'."""
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(a, b) / 100.0


def token_set_ratio(a: str, b: str) -> float:
    """Robust to extra/missing tokens: 'Acme Corp' vs 'Acme Corp of Delaware'."""
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0


def monge_elkan(a: str, b: str) -> float:
    """
    Monge-Elkan similarity: for each token in `a`, take its best similarity
    against any token in `b`, then average. Asymmetric and good at catching
    partial-name containment (e.g. one source dropping a "Global" or
    "Holdings" qualifier that the other keeps).
    """
    tokens_a = a.split()
    tokens_b = b.split()
    if not tokens_a or not tokens_b:
        return 0.0
    scores = []
    for ta in tokens_a:
        best = max(fuzz.ratio(ta, tb) / 100.0 for tb in tokens_b)
        scores.append(best)
    return float(np.mean(scores))


def char_ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    """Character n-gram Jaccard overlap — catches abbreviation-style
    similarity (e.g. 'intl' vs 'international') that word-level metrics miss."""
    def ngrams(s: str) -> set:
        s = f"  {s}  "  # pad so edge grams are still informative
        return {s[i:i + n] for i in range(len(s) - n + 1)}

    set_a, set_b = ngrams(a), ngrams(b)
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def numeric_token_overlap(a: str, b: str) -> float:
    """Jaccard overlap of numeric tokens only (street numbers, unit numbers,
    zip fragments). Two addresses can be textually similar but refer to
    different buildings if the numbers don't match — this feature lets the
    GBM penalize that case specifically."""
    nums_a = set(re.findall(r"\d+", a))
    nums_b = set(re.findall(r"\d+", b))
    if not nums_a and not nums_b:
        return 1.0  # neither has numbers -> not a distinguishing signal, don't penalize
    if not nums_a or not nums_b:
        return 0.0
    inter = len(nums_a & nums_b)
    union = len(nums_a | nums_b)
    return inter / union


def acronym_match(a: str, b: str) -> float:
    """
    Checks whether one string's initials match the other string verbatim,
    e.g. 'ibm' vs 'international business machines'. Returns 1.0 on match,
    0.0 otherwise. Directional check is applied both ways.
    """
    def initials(s: str) -> str:
        return "".join(tok[0] for tok in s.split() if tok)

    a_clean = a.replace(" ", "")
    b_clean = b.replace(" ", "")
    if not a_clean or not b_clean:
        return 0.0
    if initials(b) == a_clean or initials(a) == b_clean:
        return 1.0
    return 0.0


def length_diff_ratio(a: str, b: str) -> float:
    """Normalized absolute length difference (0 = identical length,
    1 = one string is empty and the other is not)."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 0.0
    return abs(la - lb) / max(la, lb, 1)


# -----------------------------------------------------------------------------
# Row-level feature builder
# -----------------------------------------------------------------------------

def _field_features(a: str, b: str, prefix: str) -> dict:
    return {
        f"{prefix}_jaro_winkler": jaro_winkler_sim(a, b),
        f"{prefix}_levenshtein_ratio": levenshtein_ratio(a, b),
        f"{prefix}_token_sort_ratio": token_sort_ratio(a, b),
        f"{prefix}_token_set_ratio": token_set_ratio(a, b),
        f"{prefix}_monge_elkan": monge_elkan(a, b),
        f"{prefix}_char_ngram_jaccard": char_ngram_jaccard(a, b),
        f"{prefix}_length_diff_ratio": length_diff_ratio(a, b),
    }


def compute_pair_features(
    s1_name: str, s1_addr: str, s1_zip: str,
    t_name: str, t_addr: str, t_zip: str,
) -> dict:
    """Compute the full feature dict for a single candidate pair. Pure
    function — safe to parallelize with multiprocessing/joblib if needed."""
    s1_name, s1_addr = _safe_str(s1_name), _safe_str(s1_addr)
    t_name, t_addr = _safe_str(t_name), _safe_str(t_addr)
    s1_zip, t_zip = _safe_str(s1_zip), _safe_str(t_zip)

    feats = {}
    feats.update(_field_features(s1_name, t_name, "name"))
    feats.update(_field_features(s1_addr, t_addr, "address"))

    feats["address_numeric_token_overlap"] = numeric_token_overlap(s1_addr, t_addr)
    feats["name_acronym_match"] = acronym_match(s1_name, t_name)
    feats["zip_exact_match"] = float(bool(s1_zip) and bool(t_zip) and s1_zip == t_zip)
    feats["zip_prefix3_match"] = float(
        bool(s1_zip) and bool(t_zip) and s1_zip[:3] == t_zip[:3]
    )

    # Composite convenience features: often the single strongest signal is
    # "how similar is name AND address simultaneously" rather than either alone.
    feats["name_address_avg_jw"] = (
        feats["name_jaro_winkler"] + feats["address_jaro_winkler"]
    ) / 2.0
    feats["name_address_min_token_set"] = min(
        feats["name_token_set_ratio"], feats["address_token_set_ratio"]
    )

    return feats


def build_pair_features(
    candidate_pairs: pd.DataFrame,
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    id_col: str = "id",
) -> pd.DataFrame:
    """
    Join candidate pairs against the cleaned Source 1 / Source 2 / Source 3
    tables and compute the full feature matrix.

    `candidate_pairs` must have: source1_id, target_id, target_source, plus
    the blocking strategy indicator/score columns from Module 2 (those are
    carried through as features directly, since strategy agreement itself
    is predictive of a true match).

    Returns a dataframe with one row per candidate pair: identifying columns
    (source1_id, target_id, target_source) + all engineered feature columns.
    """
    global FEATURE_COLUMNS

    s1_lookup = source1.set_index(id_col)[["clean_name", "clean_address", "clean_zip"]]
    s2_lookup = source2.set_index(id_col)[["clean_name", "clean_address", "clean_zip"]]
    s3_lookup = source3.set_index(id_col)[["clean_name", "clean_address", "clean_zip"]]
    target_lookups = {"source2": s2_lookup, "source3": s3_lookup}

    records = []
    blocking_score_cols = [
        c for c in candidate_pairs.columns
        if c.startswith("strategy_") or c.endswith("_score") or c == "num_strategies_agree"
    ]

    for target_source, group in candidate_pairs.groupby("target_source"):
        t_lookup = target_lookups[target_source]
        s1_ids = group["source1_id"].values
        t_ids = group["target_id"].values

        s1_rows = s1_lookup.reindex(s1_ids)
        t_rows = t_lookup.reindex(t_ids)

        for i in range(len(group)):
            feats = compute_pair_features(
                s1_rows["clean_name"].iloc[i], s1_rows["clean_address"].iloc[i], s1_rows["clean_zip"].iloc[i],
                t_rows["clean_name"].iloc[i], t_rows["clean_address"].iloc[i], t_rows["clean_zip"].iloc[i],
            )
            row = {
                "source1_id": s1_ids[i],
                "target_id": t_ids[i],
                "target_source": target_source,
            }
            for c in blocking_score_cols:
                row[c] = group[c].iloc[i]
            row.update(feats)
            records.append(row)

    feature_df = pd.DataFrame.from_records(records)

    id_cols = {"source1_id", "target_id", "target_source"}
    FEATURE_COLUMNS = [c for c in feature_df.columns if c not in id_cols]

    return feature_df
