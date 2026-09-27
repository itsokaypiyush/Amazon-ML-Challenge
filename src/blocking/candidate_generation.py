"""
Module 2: Blocking & Candidate Generation (Stage 1)
=====================================================

Goal: reduce the O(N_source1 * N_source2) / O(N_source1 * N_source3) pairwise
comparison space down to a manageable shortlist per Source 1 record, without
losing true matches (blocking recall is the metric that matters here — Stage
2/3 can only fix precision, never recover candidates blocking drops).

Strategy (union of FOUR independent blocking signals, each catching
different failure modes of the others — this redundancy is deliberate:
candidate recall is a hard ceiling on the whole pipeline, so no single key
is trusted alone):

  1. Phonetic+geo exact key (NYSIIS code of first name token + zip-prefix/
     city, built in Module 1's `blocking_key` column) — catches spelling
     variants and minor typos within the same region.
  2. First-raw-token+zip exact key — an INDEPENDENT exact key that doesn't
     depend on phonetic encoding at all. Catches cases where NYSIIS
     collapses two genuinely-different-sounding names together (raising
     block size without helping recall) or, more importantly, where a
     short/numeric/foreign name produces an unstable phonetic code but the
     literal first token still agrees.
  3. TF-IDF char n-gram + cosine similarity top-K — catches partial-token
     overlaps, abbreviations, reordering (does not require any exact key
     match at all, so it fixes recall gaps from (1) and (2) together).
  4. Sentence-embedding + FAISS approximate nearest neighbor top-K —
     catches semantically similar but lexically dissimilar variants
     (different abbreviation styles, translated legal suffixes, etc).

Every distinct (source1_id, target_id) pair produced by ANY strategy is kept
(union, not intersection) and tagged with which strategy found it — this
tag becomes a feature in Module 3, since agreement across strategies is
itself a strong precision signal. `evaluate_blocking_recall` below should be
run against `train_ground_truth.tsv` any time a blocking parameter changes,
since this stage is the only place true matches can be lost for good.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None


def _combined_text(row: pd.Series) -> str:
    """Concatenate normalized name + address into one string for vectorization.
    Name is weighted implicitly by appearing once; address adds disambiguating
    geo context without dominating (names tend to be the stronger match signal)."""
    return f"{row.get('clean_name', '')} {row.get('clean_address', '')}".strip()


@dataclass
class CandidateGenerator:
    """
    Encapsulates all blocking strategies and their fitted artifacts so the
    exact same vectorizer/index can be reused for train and test inference
    (critical: fitting a fresh TF-IDF vocab on test data would silently
    change feature semantics between train and inference).
    """

    top_k: int = 20
    ngram_range: tuple = (2, 3)
    tfidf_max_features: int = 50_000
    use_zip_prefix_block: bool = True
    zip_prefix_len: int = 3
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 256
    faiss_nlist: int = 100
    use_embeddings: bool = True

    _tfidf_vectorizer: Optional[TfidfVectorizer] = field(default=None, init=False, repr=False)
    _embedder: Optional["SentenceTransformer"] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Strategy 1: exact blocking-key match
    # ------------------------------------------------------------------
    def _blocking_key_candidates(
        self, source1: pd.DataFrame, target: pd.DataFrame, target_name: str
    ) -> pd.DataFrame:
        """Strategy 1: exact match on Module 1's phonetic+geo `blocking_key`."""
        merged = source1[["id", "blocking_key"]].merge(
            target[["id", "blocking_key"]],
            on="blocking_key",
            suffixes=("_s1", f"_{target_name}"),
        )
        merged = merged.rename(
            columns={"id_s1": "source1_id", f"id_{target_name}": "target_id"}
        )
        merged["strategy_blocking_key"] = 1
        return merged[["source1_id", "target_id", "strategy_blocking_key"]]

    @staticmethod
    def _first_token_zip_key(name: str, zip_code: str, zip_prefix_len: int) -> str:
        """Second, independent exact key: raw first name token (not phonetic-
        encoded) + zip prefix. Deliberately does NOT reuse the phonetic code,
        so a name whose NYSIIS encoding is unstable (very short, numeric, or
        non-English tokens) still gets a real shot at an exact-key match."""
        first_tok = name.split(" ")[0] if name else ""
        geo = (zip_code or "")[:zip_prefix_len]
        return f"{first_tok}|{geo}"

    def _first_token_key_candidates(
        self, source1: pd.DataFrame, target: pd.DataFrame, target_name: str
    ) -> pd.DataFrame:
        """Strategy 2: exact match on the first-raw-token+zip key. Only
        applied when `use_zip_prefix_block` is enabled in config, since it
        is meaningless without a geo field to disambiguate common first
        tokens (e.g. "The", "Global") from colliding across regions."""
        if not self.use_zip_prefix_block:
            return pd.DataFrame(columns=["source1_id", "target_id", "strategy_first_token_key"])

        s1 = source1.copy()
        t = target.copy()
        s1["_ft_key"] = [
            self._first_token_zip_key(n, z, self.zip_prefix_len)
            for n, z in zip(s1["clean_name"], s1.get("clean_zip", ""))
        ]
        t["_ft_key"] = [
            self._first_token_zip_key(n, z, self.zip_prefix_len)
            for n, z in zip(t["clean_name"], t.get("clean_zip", ""))
        ]
        # Empty first-token or empty geo is not a meaningful match signal —
        # drop those rows so blank keys don't fan out into a huge false block.
        s1 = s1[s1["_ft_key"].str.len() > 1]
        t = t[t["_ft_key"].str.len() > 1]

        merged = s1[["id", "_ft_key"]].merge(
            t[["id", "_ft_key"]], on="_ft_key", suffixes=("_s1", f"_{target_name}")
        )
        merged = merged.rename(
            columns={"id_s1": "source1_id", f"id_{target_name}": "target_id"}
        )
        merged["strategy_first_token_key"] = 1
        return merged[["source1_id", "target_id", "strategy_first_token_key"]]

    # ------------------------------------------------------------------
    # Strategy 2: TF-IDF char n-gram cosine top-K
    # ------------------------------------------------------------------
    def _fit_tfidf(self, corpus: list[str]) -> TfidfVectorizer:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            max_features=self.tfidf_max_features,
            sublinear_tf=True,
        )
        vectorizer.fit(corpus)
        return vectorizer

    def _tfidf_topk_candidates(
        self, source1: pd.DataFrame, target: pd.DataFrame, target_name: str
    ) -> pd.DataFrame:
        s1_text = source1.apply(_combined_text, axis=1).tolist()
        t_text = target.apply(_combined_text, axis=1).tolist()

        if self._tfidf_vectorizer is None:
            # Fit once on the union of all reference-side text seen so far
            # (Source 1 is the anchor and is always available at fit time).
            self._tfidf_vectorizer = self._fit_tfidf(s1_text + t_text)

        s1_vecs = self._tfidf_vectorizer.transform(s1_text)
        t_vecs = self._tfidf_vectorizer.transform(t_text)

        k = min(self.top_k, len(target))
        if k == 0:
            return pd.DataFrame(columns=["source1_id", "target_id", "strategy_tfidf", "tfidf_score"])

        nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
        nn.fit(t_vecs)
        distances, indices = nn.kneighbors(s1_vecs)

        rows = []
        s1_ids = source1["id"].values
        t_ids = target["id"].values
        for i, s1_id in enumerate(s1_ids):
            for rank in range(k):
                sim = 1.0 - distances[i, rank]
                if sim <= 0:
                    continue
                rows.append((s1_id, t_ids[indices[i, rank]], 1, sim))

        return pd.DataFrame(
            rows, columns=["source1_id", "target_id", "strategy_tfidf", "tfidf_score"]
        )

    # ------------------------------------------------------------------
    # Strategy 3: sentence-embedding + FAISS ANN top-K
    # ------------------------------------------------------------------
    def _get_embedder(self):
        if self._embedder is None:
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers is not installed but use_embeddings=True. "
                    "Install it or set use_embeddings=False in the config."
                )
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def _embed(self, texts: list[str]) -> np.ndarray:
        embedder = self._get_embedder()
        vecs = embedder.encode(
            texts,
            batch_size=self.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # so inner product == cosine similarity
            convert_to_numpy=True,
        )
        return vecs.astype("float32")

    def _faiss_topk_candidates(
        self, source1: pd.DataFrame, target: pd.DataFrame, target_name: str
    ) -> pd.DataFrame:
        s1_text = source1.apply(_combined_text, axis=1).tolist()
        t_text = target.apply(_combined_text, axis=1).tolist()

        if len(t_text) == 0:
            return pd.DataFrame(
                columns=["source1_id", "target_id", "strategy_embedding", "embedding_score"]
            )

        s1_vecs = self._embed(s1_text)
        t_vecs = self._embed(t_text)

        dim = t_vecs.shape[1]
        n_target = t_vecs.shape[0]
        k = min(self.top_k, n_target)

        if faiss is not None:
            use_ivf = n_target >= max(self.faiss_nlist * 10, 1000)
            if use_ivf:
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFFlat(quantizer, dim, self.faiss_nlist, faiss.METRIC_INNER_PRODUCT)
                index.train(t_vecs)
                index.add(t_vecs)
                index.nprobe = min(10, self.faiss_nlist)
            else:
                # Small candidate pools: exact flat search is fast enough and
                # avoids IVF's under-trained-cluster degenerate behavior.
                index = faiss.IndexFlatIP(dim)
                index.add(t_vecs)
            scores, indices = index.search(s1_vecs, k)
        else:
            logger.warning("faiss not installed; falling back to sklearn brute-force ANN.")
            nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
            nn.fit(t_vecs)
            dist, indices = nn.kneighbors(s1_vecs)
            scores = 1.0 - dist

        rows = []
        s1_ids = source1["id"].values
        t_ids = target["id"].values
        for i, s1_id in enumerate(s1_ids):
            for rank in range(k):
                idx = indices[i, rank]
                if idx < 0:
                    continue
                rows.append((s1_id, t_ids[idx], 1, float(scores[i, rank])))

        return pd.DataFrame(
            rows, columns=["source1_id", "target_id", "strategy_embedding", "embedding_score"]
        )

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def generate_for_target(
        self, source1: pd.DataFrame, target: pd.DataFrame, target_name: str
    ) -> pd.DataFrame:
        """Run all enabled strategies against one target source and union
        the results, keeping per-strategy indicator/score columns."""
        frames = [self._blocking_key_candidates(source1, target, target_name)]
        frames.append(self._first_token_key_candidates(source1, target, target_name))
        frames.append(self._tfidf_topk_candidates(source1, target, target_name))
        if self.use_embeddings:
            try:
                frames.append(self._faiss_topk_candidates(source1, target, target_name))
            except ImportError as e:
                logger.warning("Skipping embedding-based blocking: %s", e)

        merged = frames[0]
        for f in frames[1:]:
            merged = merged.merge(f, on=["source1_id", "target_id"], how="outer")

        for col in ["strategy_blocking_key", "strategy_first_token_key", "strategy_tfidf", "strategy_embedding"]:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0).astype(int)
        for col in ["tfidf_score", "embedding_score"]:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0)

        merged["target_source"] = target_name
        merged["num_strategies_agree"] = merged[
            [c for c in merged.columns if c.startswith("strategy_")]
        ].sum(axis=1)

        n_s1 = source1["id"].nunique()
        n_candidates = len(merged)
        logger.info(
            "[%s] %d source1 records -> %d candidate pairs (avg %.1f candidates/record)",
            target_name, n_s1, n_candidates, n_candidates / max(n_s1, 1),
        )
        return merged


def generate_candidate_pairs(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Top-level entry point: builds a single generator (so TF-IDF vocabulary is
    consistent across both target sources) and returns one long-format
    dataframe of candidate pairs across source2 and source3, ready to be
    written to `candidate_pairs.tsv` and consumed by Module 3.

    Output columns:
        source1_id, target_id, target_source,
        strategy_blocking_key, strategy_tfidf, strategy_embedding,
        tfidf_score, embedding_score, num_strategies_agree
    """
    gen = CandidateGenerator(
        top_k=config["blocking"]["top_k"],
        ngram_range=tuple(config["blocking"]["ngram_range"]),
        tfidf_max_features=config["blocking"]["tfidf_max_features"],
        use_zip_prefix_block=config["blocking"]["use_zip_prefix_block"],
        zip_prefix_len=config["blocking"]["zip_prefix_len"],
        embedding_model_name=config["blocking"]["embedding_model"],
        embedding_batch_size=config["blocking"]["embedding_batch_size"],
        faiss_nlist=config["blocking"]["faiss_nlist"],
        use_embeddings=config["blocking"].get("use_embeddings", True),
    )

    cand_s2 = gen.generate_for_target(source1, source2, "source2")
    cand_s3 = gen.generate_for_target(source1, source3, "source3")

    all_candidates = pd.concat([cand_s2, cand_s3], ignore_index=True)
    return all_candidates


def evaluate_blocking_recall(
    candidate_pairs: pd.DataFrame, ground_truth_long: pd.DataFrame
) -> dict:
    """
    Sanity-check utility: what fraction of TRUE (source1_id, target_id) pairs
    from ground truth actually survived into the candidate set? This should
    be run during development (never at inference, since test labels don't
    exist) to make sure top_k / thresholds aren't silently dropping true
    matches before Stage 2 ever sees them.

    `ground_truth_long` must have columns [source1_id, target_id, target_source].
    """
    cand_keys = set(
        zip(candidate_pairs["source1_id"], candidate_pairs["target_id"], candidate_pairs["target_source"])
    )
    gt_keys = list(
        zip(ground_truth_long["source1_id"], ground_truth_long["target_id"], ground_truth_long["target_source"])
    )
    if not gt_keys:
        return {"recall": float("nan"), "n_true_pairs": 0, "n_found": 0}

    found = sum(1 for k in gt_keys if k in cand_keys)
    return {
        "recall": found / len(gt_keys),
        "n_true_pairs": len(gt_keys),
        "n_found": found,
        "n_missed": len(gt_keys) - found,
    }
