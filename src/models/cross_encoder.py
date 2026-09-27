"""
Module 3: Transformer Cross-Encoder Matcher (Stage 2, part C)
================================================================

Fine-tunes a small transformer (default: DeBERTa-v3-small) as a binary
cross-encoder over formatted pair text:

    [CLS] <s1 name> <s1 address> [SEP] <cand name> <cand address> [SEP]

Why alongside the GBM (not instead of): the GBM's hand-engineered string
metrics are strong on lexical variation but blind to compositional
semantics (e.g. "Global Freight Solutions" vs "Worldwide Cargo Services" as
a possible DBA/rename pair, or word-order-independent legal-entity
paraphrases the abbreviation map didn't anticipate). The cross-encoder
learns those directly from data. The two model scores are combined at
inference (Module 4 / ensemble).

This module is optional and gated by `cross_encoder.enabled` in the config
since it is far more expensive to train than the GBM; for smaller
datasets or tight compute budgets, GBM-only is a fully functional pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

logger = logging.getLogger(__name__)


def format_pair_text(name: str, address: str) -> str:
    """Format one side of a pair as a single text segment. Name first (it
    carries most of the matching signal); address appended for context."""
    name = name or ""
    address = address or ""
    return f"{name} , {address}".strip(" ,")


class PairDataset(Dataset):
    """Tokenizes (text_a, text_b) pairs lazily via the HF tokenizer's
    native pair-encoding (`tokenizer(text_a, text_b, ...)`), which correctly
    inserts the [SEP] token and builds token_type_ids for encoder models
    like DeBERTa/RoBERTa/BERT."""

    def __init__(self, texts_a: list[str], texts_b: list[str],
                 labels: Optional[list[int]], tokenizer, max_length: int):
        self.texts_a = texts_a
        self.texts_b = texts_b
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts_a)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts_a[idx],
            self.texts_b[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _compute_metrics(eval_pred):
    from sklearn.metrics import average_precision_score, roc_auc_score

    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    out = {}
    try:
        out["auprc"] = average_precision_score(labels, probs)
        out["auroc"] = roc_auc_score(labels, probs)
    except ValueError:
        # Can happen on a degenerate eval split with a single class present.
        out["auprc"] = float("nan")
        out["auroc"] = float("nan")
    return out


@dataclass
class CrossEncoderMatcher:
    base_model: str = "microsoft/deberta-v3-small"
    max_length: int = 96
    batch_size: int = 32
    learning_rate: float = 2e-5
    epochs: int = 3
    warmup_ratio: float = 0.1
    output_dir: str = "models/cross_encoder"
    random_state: int = 42

    tokenizer: Optional[object] = field(default=None, init=False, repr=False)
    model: Optional[object] = field(default=None, init=False, repr=False)

    def _load_base(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.base_model, num_labels=2
            )

    def _build_texts(self, df: pd.DataFrame, s1_lookup: pd.DataFrame, t_lookups: dict):
        texts_a, texts_b = [], []
        for _, row in df.iterrows():
            s1 = s1_lookup.loc[row["source1_id"]]
            t = t_lookups[row["target_source"]].loc[row["target_id"]]
            texts_a.append(format_pair_text(s1["clean_name"], s1["clean_address"]))
            texts_b.append(format_pair_text(t["clean_name"], t["clean_address"]))
        return texts_a, texts_b

    def fit(
        self,
        labeled_pairs: pd.DataFrame,
        s1_lookup: pd.DataFrame,
        t_lookups: dict,
        val_fraction: float = 0.15,
    ) -> dict:
        """
        `labeled_pairs` must have: source1_id, target_id, target_source, label.
        `s1_lookup` / each of `t_lookups` values: indexed by id, columns
        clean_name, clean_address (same lookups built in Module 3a).

        Splits by source1_id group (not random rows) to avoid leakage
        between train/val, same rationale as the GBM's GroupKFold.
        """
        self._load_base()

        rng = np.random.RandomState(self.random_state)
        unique_groups = labeled_pairs["source1_id"].unique()
        rng.shuffle(unique_groups)
        n_val_groups = max(1, int(len(unique_groups) * val_fraction))
        val_groups = set(unique_groups[:n_val_groups])

        is_val = labeled_pairs["source1_id"].isin(val_groups)
        train_df = labeled_pairs[~is_val].reset_index(drop=True)
        val_df = labeled_pairs[is_val].reset_index(drop=True)

        logger.info(
            "Cross-encoder train/val split: %d train pairs, %d val pairs (%d val groups)",
            len(train_df), len(val_df), n_val_groups,
        )

        train_a, train_b = self._build_texts(train_df, s1_lookup, t_lookups)
        val_a, val_b = self._build_texts(val_df, s1_lookup, t_lookups)

        train_ds = PairDataset(train_a, train_b, train_df["label"].tolist(),
                                self.tokenizer, self.max_length)
        val_ds = PairDataset(val_a, val_b, val_df["label"].tolist(),
                              self.tokenizer, self.max_length)

        args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            learning_rate=self.learning_rate,
            num_train_epochs=self.epochs,
            warmup_ratio=self.warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="auprc",
            greater_is_better=True,
            logging_steps=50,
            save_total_limit=2,
            report_to=[],
            seed=self.random_state,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=_compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        trainer.train()
        eval_metrics = trainer.evaluate()

        self.model = trainer.model
        return eval_metrics

    def predict_proba(
        self, df: pd.DataFrame, s1_lookup: pd.DataFrame, t_lookups: dict
    ) -> np.ndarray:
        """Returns P(match=1) for each row of df, in the same row order."""
        self._load_base()
        self.model.eval()

        texts_a, texts_b = self._build_texts(df, s1_lookup, t_lookups)
        ds = PairDataset(texts_a, texts_b, None, self.tokenizer, self.max_length)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        probs = []
        batch_size = self.batch_size * 2
        with torch.no_grad():
            for start in range(0, len(ds), batch_size):
                batch = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
                input_ids = torch.stack([b["input_ids"] for b in batch]).to(device)
                attention_mask = torch.stack([b["attention_mask"] for b in batch]).to(device)
                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if "token_type_ids" in batch[0]:
                    kwargs["token_type_ids"] = torch.stack(
                        [b["token_type_ids"] for b in batch]
                    ).to(device)
                logits = self.model(**kwargs).logits
                batch_probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                probs.append(batch_probs)

        return np.concatenate(probs) if probs else np.array([])

    def save(self, path: str) -> None:
        self._load_base()
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "CrossEncoderMatcher":
        instance = cls(base_model=path, **kwargs)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForSequenceClassification.from_pretrained(path)
        return instance
