"""
Module 2: Blocking & Candidate Generation (Stage 1)
=====================================================

Goal: reduce the O(N_source1 * N_source2) / O(N_source1 * N_source3) pairwise
comparison space down to a manageable shortlist per Source 1 record, without
losing true matches (blocking recall is the metric that matters here — Stage
2/3 can only fix precision, never recover candidates blocking drops).

Everything runs per country: every true match in the training ground truth
is between records of the same country, so partitioning is free recall-wise
and shrinks each search space several-fold.

Strategies (unioned, each catching different failure modes of the others):

  1. Phonetic+geo exact key (NYSIIS code of first name token + geo key,
     built in Module 1's `blocking_key` column) — catches spelling variants
     and minor typos within the same region.
  2. First-raw-token+geo exact key — an INDEPENDENT exact key that doesn't
     depend on phonetic encoding at all (short/numeric/foreign names).
  2b. Street-number + street-name exact key — catches renamed records
     (DBA names, web domains) at an unchanged address.
  3. TF-IDF char n-gram nearest neighbours — catches partial-token overlaps,
     abbreviations, reordering. To scale to millions of records, TF-IDF
     vectors are compressed with TruncatedSVD and searched with a FAISS IVF
     index; the retrieved shortlist is then re-ranked by the EXACT sparse
     TF-IDF cosine and cut to `top_k`.
  4. (Optional, off by default) sentence-embedding + FAISS nearest
     neighbours. On the training data it adds <0.1% blocking recall over
     1-3 while being by far the slowest strategy.

Exact-key blocks with more than `max_block_size` target records are skipped
(the data only has `country` as a geo field, so keys like "SHR|india" would
otherwise explode into huge joins); those records still get TF-IDF candidates.

Every distinct (source1_id, target_id) pair produced by ANY strategy is kept
(union, not intersection) and tagged with which strategy found it — these
tags, the exact TF-IDF cosine, the candidate's TF-IDF rank and its gap to the
best candidate of the same source1 record become features in Module 3.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None

STRATEGY_COLUMNS = [
    "strategy_blocking_key", "strategy_first_token_key", "strategy_address_key",
    "strategy_tfidf", "strategy_embedding",
]


def _combined_text(df: pd.DataFrame) -> list[str]:
    """Normalized name + address as one string per record, for vectorization.
    Name is weighted implicitly by appearing once; address adds disambiguating
    geo context without dominating (names tend to be the stronger match signal)."""
    return (df["clean_name"].fillna("") + " " + df["clean_address"].fillna("")).str.strip().tolist()


def _first_token_key(df: pd.DataFrame) -> np.ndarray:
    """Raw first name token + geo key. Deliberately does NOT reuse the
    phonetic code, so a name whose NYSIIS encoding is unstable (very short,
    numeric, or non-English tokens) still gets a real shot at an exact match.
    Empty first token or empty geo gives "" (never matched)."""
    first = df["clean_name"].fillna("").str.split(" ", n=1).str[0]
    geo = df["geo_key"].fillna("")
    return np.where((first != "") & (geo != ""), first + "|" + geo, "")


def _address_key(df: pd.DataFrame) -> np.ndarray:
    """First street number + the token after it + country ("1718 mowry|us").
    Catches records whose names were rewritten (DBA names, domains, missing
    legal suffix) but whose street address is intact. "" when the address
    has no number followed by a word (never matched)."""
    out = np.full(len(df), "", dtype=object)
    for i, (addr, country) in enumerate(zip(df["clean_address"].fillna(""), df["country_key"].fillna(""))):
        toks = addr.split()
        for j in range(len(toks) - 1):
            if toks[j].isdigit():
                out[i] = f"{toks[j]} {toks[j + 1]}|{country}"
                break
    return out


def _pair_cosine(A: sp.csr_matrix, B: sp.csr_matrix, qa: np.ndarray, tb: np.ndarray,
                 chunk: int = 500_000) -> np.ndarray:
    """Exact cosine of row qa[i] of A and row tb[i] of B (rows are L2-normalized)."""
    out = np.empty(len(qa), dtype=np.float32)
    for s in range(0, len(qa), chunk):
        e = s + chunk
        out[s:e] = np.asarray(A[qa[s:e]].multiply(B[tb[s:e]]).sum(axis=1)).ravel()
    return out


@dataclass
class CandidateGenerator:
    """
    Holds the blocking configuration and the fitted TF-IDF vectorizer + SVD,
    so both target sources are searched in the same vector space.
    """

    top_k: int = 20
    ngram_range: tuple = (2, 3)
    tfidf_max_features: int = 50_000
    use_zip_prefix_block: bool = True
    use_address_block: bool = True
    max_block_size: int = 50
    svd_dim: int = 256
    fit_sample_size: int = 300_000
    ann_nprobe: int = 256
    rerank_factor: int = 5
    use_embeddings: bool = False
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 256
    n_jobs: int = -1

    _vectorizer: Optional[TfidfVectorizer] = field(default=None, init=False, repr=False)
    _svd: Optional[TruncatedSVD] = field(default=None, init=False, repr=False)
    _embedder: Optional["SentenceTransformer"] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Text models
    # ------------------------------------------------------------------
    def fit(self, texts: list[str]) -> "CandidateGenerator":
        """Fit TF-IDF vocabulary and SVD projection on a random sample of texts."""
        rng = np.random.default_rng(0)
        if len(texts) > self.fit_sample_size:
            idx = rng.choice(len(texts), self.fit_sample_size, replace=False)
            texts = [texts[i] for i in idx]
        t0 = time.time()
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=self.ngram_range,
            max_features=self.tfidf_max_features, sublinear_tf=True, dtype=np.float32,
        ).fit(texts)
        X = self._vectorizer.transform(texts)
        self._svd = TruncatedSVD(n_components=self.svd_dim, random_state=0).fit(X)
        logger.info("Fitted TF-IDF (%d features) + SVD(%d) on %d texts in %.0fs",
                    len(self._vectorizer.vocabulary_), self.svd_dim, len(texts), time.time() - t0)
        return self

    def _tfidf(self, texts: list[str]) -> sp.csr_matrix:
        """Parallel TF-IDF transform (sklearn's own transform is single-threaded)."""
        step = 200_000
        if len(texts) <= step:
            return self._vectorizer.transform(texts).tocsr()
        parts = Parallel(n_jobs=self.n_jobs)(
            delayed(self._vectorizer.transform)(texts[i:i + step]) for i in range(0, len(texts), step)
        )
        return sp.vstack(parts).tocsr()

    def _dense(self, X: sp.csr_matrix) -> np.ndarray:
        D = np.ascontiguousarray(self._svd.transform(X), dtype=np.float32)
        faiss.normalize_L2(D)
        return D

    @staticmethod
    def _build_index(vecs: np.ndarray, nprobe: int):
        n, d = vecs.shape
        if n < 50_000:
            index = faiss.IndexFlatIP(d)
            index.add(vecs)
            return index
        nlist = int(min(16_384, 4 * np.sqrt(n)))
        index = faiss.index_factory(d, f"IVF{nlist},Flat", faiss.METRIC_INNER_PRODUCT)
        rng = np.random.default_rng(0)
        index.train(vecs[rng.choice(n, min(n, nlist * 64), replace=False)])
        index.add(vecs)
        faiss.extract_index_ivf(index).nprobe = min(nprobe, nlist)
        return index

    # ------------------------------------------------------------------
    # Strategies (all return int64 pair codes q * n_target + t).
    # On the training data (US, full source2 pool): the SVD/FAISS settings
    # svd_dim=128/nprobe=32 kept 75% of true pairs, 256/256 with 100
    # retrieved kept 91%, and adding the address key brought it to 95.5%.
    # ------------------------------------------------------------------
    def _key_codes(self, s1_keys: np.ndarray, t_keys: np.ndarray, n_t: int) -> np.ndarray:
        s1 = pd.DataFrame({"k": s1_keys, "q": np.arange(len(s1_keys))})
        t = pd.DataFrame({"k": t_keys, "t": np.arange(len(t_keys))})
        s1, t = s1[s1["k"] != ""], t[t["k"] != ""]
        sizes = t["k"].value_counts()
        ok = sizes.index[sizes <= self.max_block_size]
        t = t[t["k"].isin(ok)]
        s1 = s1[s1["k"].isin(ok)]
        m = s1.merge(t, on="k")
        return m["q"].to_numpy(np.int64) * n_t + m["t"].to_numpy(np.int64)

    def _tfidf_codes(self, S1: sp.csr_matrix, T: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
        """ANN shortlist in SVD space, re-ranked by exact TF-IDF cosine, top_k kept."""
        n_q, n_t = S1.shape[0], T.shape[0]
        k = min(self.top_k * self.rerank_factor, n_t)
        index = self._build_index(self._dense(T), self.ann_nprobe)
        codes, scores = [], []
        step = 100_000
        for s in range(0, n_q, step):
            S1c = S1[s:s + step]
            _, I = index.search(self._dense(S1c), k)
            nq = I.shape[0]
            q = np.repeat(np.arange(nq), k)
            t = I.ravel()
            valid = t >= 0
            sim = np.full(len(t), -1.0, dtype=np.float32)
            sim[valid] = _pair_cosine(S1c, T, q[valid], t[valid])
            sim = sim.reshape(nq, k)
            top = np.argsort(-sim, axis=1)[:, : self.top_k]
            rows = np.arange(nq)[:, None]
            top_sim, top_t = sim[rows, top].ravel(), I[rows, top].ravel()
            top_q = np.repeat(np.arange(s, s + nq), top.shape[1])
            keep = (top_t >= 0) & (top_sim > 0)
            codes.append(top_q[keep].astype(np.int64) * n_t + top_t[keep])
            scores.append(top_sim[keep])
        return np.concatenate(codes), np.concatenate(scores)

    def _embedding_codes(self, s1_texts: list[str], t_texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if self._embedder is None:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers is not installed but use_embeddings=True.")
            self._embedder = SentenceTransformer(self.embedding_model_name)
        enc = lambda x: self._embedder.encode(  # noqa: E731
            x, batch_size=self.embedding_batch_size, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        T = enc(t_texts)
        index = self._build_index(T, self.ann_nprobe)
        k = min(self.top_k, len(t_texts))
        D, I = index.search(enc(s1_texts), k)
        q = np.repeat(np.arange(len(s1_texts)), k)
        t, d = I.ravel(), D.ravel()
        keep = t >= 0
        return q[keep].astype(np.int64) * len(t_texts) + t[keep], d[keep].astype(np.float32)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def _partition_candidates(self, s1: pd.DataFrame, t: pd.DataFrame, target_name: str) -> pd.DataFrame:
        n_t = len(t)
        s1_texts, t_texts = _combined_text(s1), _combined_text(t)
        S1, T = self._tfidf(s1_texts), self._tfidf(t_texts)

        found = {}
        tfidf_codes, tfidf_scores = self._tfidf_codes(S1, T)
        found["strategy_tfidf"] = tfidf_codes
        found["strategy_blocking_key"] = self._key_codes(
            s1["blocking_key"].to_numpy(), t["blocking_key"].to_numpy(), n_t)
        if self.use_zip_prefix_block:
            found["strategy_first_token_key"] = self._key_codes(
                _first_token_key(s1), _first_token_key(t), n_t)
        if self.use_address_block:
            found["strategy_address_key"] = self._key_codes(_address_key(s1), _address_key(t), n_t)
        emb = None
        if self.use_embeddings:
            emb = self._embedding_codes(s1_texts, t_texts)
            found["strategy_embedding"] = emb[0]

        codes = np.unique(np.concatenate(list(found.values())))
        q, ti = codes // n_t, codes % n_t

        # Exact TF-IDF cosine for every pair (reuse what the ANN step computed).
        score = np.empty(len(codes), dtype=np.float32)
        pos = np.searchsorted(codes, tfidf_codes)
        score[:] = np.nan
        score[pos] = tfidf_scores
        missing = np.isnan(score)
        if missing.any():
            score[missing] = _pair_cosine(S1, T, q[missing], ti[missing])

        out = pd.DataFrame({
            "source1_id": s1["id"].to_numpy()[q],
            "target_id": t["id"].to_numpy()[ti],
            "target_source": target_name,
        })
        for col in STRATEGY_COLUMNS:
            if col in found:
                out[col] = np.isin(codes, found[col]).astype(np.int8)
        out["tfidf_score"] = score
        if emb is not None:
            e = np.zeros(len(codes), dtype=np.float32)
            e[np.searchsorted(codes, emb[0])] = emb[1]
            out["embedding_score"] = e

        # Rank of each candidate within its source1 record (by exact TF-IDF,
        # 1 = best) and its gap to that record's best candidate.
        order = np.lexsort((-score, q))
        q_sorted = q[order]
        starts = np.r_[0, np.flatnonzero(np.diff(q_sorted)) + 1]
        group_start = np.repeat(starts, np.diff(np.r_[starts, len(q_sorted)]))
        rank = np.empty(len(codes), dtype=np.int16)
        rank[order] = (np.arange(len(codes)) - group_start + 1).astype(np.int16)
        best = np.empty(len(codes), dtype=np.float32)
        best[order] = score[order][group_start]
        out["tfidf_rank"] = rank
        out["tfidf_gap"] = score - best
        out["num_strategies_agree"] = out[[c for c in STRATEGY_COLUMNS if c in out]].sum(axis=1).astype(np.int8)
        return out

    def generate_for_target(self, source1: pd.DataFrame, target: pd.DataFrame, target_name: str) -> pd.DataFrame:
        """Run all enabled strategies against one target source, country by country."""
        t0 = time.time()
        s1_country = source1["country_key"] if "country_key" in source1 else pd.Series("", index=source1.index)
        t_country = target["country_key"] if "country_key" in target else pd.Series("", index=target.index)
        frames = []
        for country in sorted(set(s1_country.unique())):
            s1c = source1[s1_country.to_numpy() == country]
            tc = target[t_country.to_numpy() == country]
            if len(tc) == 0 or len(s1c) == 0:
                continue
            tp = time.time()
            frames.append(self._partition_candidates(s1c, tc, target_name))
            logger.info("[%s/%s] %d source1 x %d target -> %d pairs (%.0fs)",
                        target_name, country or "-", len(s1c), len(tc), len(frames[-1]), time.time() - tp)
        merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["source1_id", "target_id", "target_source"])
        n_s1 = source1["id"].nunique()
        logger.info("[%s] %d source1 records -> %d candidate pairs (avg %.1f/record) in %.0fs",
                    target_name, n_s1, len(merged), len(merged) / max(n_s1, 1), time.time() - t0)
        return merged


def generate_candidate_pairs(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Top-level entry point: fits one TF-IDF+SVD space on a sample of all three
    sources and returns one long-format dataframe of candidate pairs across
    source2 and source3, ready for Module 3.

    Output columns:
        source1_id, target_id, target_source, strategy_* flags,
        tfidf_score, tfidf_rank, tfidf_gap, [embedding_score],
        num_strategies_agree
    """
    b = config["blocking"]
    gen = CandidateGenerator(
        top_k=b["top_k"],
        ngram_range=tuple(b["ngram_range"]),
        tfidf_max_features=b["tfidf_max_features"],
        use_zip_prefix_block=b["use_zip_prefix_block"],
        use_address_block=b.get("use_address_block", True),
        max_block_size=b.get("max_block_size", 50),
        svd_dim=b.get("svd_dim", 256),
        fit_sample_size=b.get("fit_sample_size", 300_000),
        ann_nprobe=b.get("ann_nprobe", 256),
        rerank_factor=b.get("rerank_factor", 5),
        use_embeddings=b.get("use_embeddings", False),
        embedding_model_name=b["embedding_model"],
        embedding_batch_size=b["embedding_batch_size"],
    )
    per_source = max(1, gen.fit_sample_size // 3)
    fit_texts = []
    for df in (source1, source2, source3):
        fit_texts += _combined_text(df.sample(n=min(len(df), per_source), random_state=0))
    gen.fit(fit_texts)

    cand_s2 = gen.generate_for_target(source1, source2, "source2")
    cand_s3 = gen.generate_for_target(source1, source3, "source3")
    out = pd.concat([cand_s2, cand_s3], ignore_index=True)
    out["target_source"] = out["target_source"].astype("category")
    return out


def evaluate_blocking_recall(
    candidate_pairs: pd.DataFrame, ground_truth_long: pd.DataFrame
) -> dict:
    """
    Sanity-check utility: what fraction of TRUE (source1_id, target_id) pairs
    from ground truth actually survived into the candidate set? Run during
    development (never at inference, since test labels don't exist).

    `ground_truth_long` must have columns [source1_id, target_id, target_source].
    """
    if ground_truth_long.empty:
        return {"recall": float("nan"), "n_true_pairs": 0, "n_found": 0, "n_missed": 0}
    cand_keys = pd.Index(candidate_pairs["source1_id"].astype(str) + "|" + candidate_pairs["target_id"].astype(str))
    gt_keys = ground_truth_long["source1_id"].astype(str) + "|" + ground_truth_long["target_id"].astype(str)
    found = int(gt_keys.isin(cand_keys).sum())
    return {
        "recall": found / len(gt_keys),
        "n_true_pairs": len(gt_keys),
        "n_found": found,
        "n_missed": len(gt_keys) - found,
    }
