"""Fake news transformer training (DistilBERT)."""

from __future__ import annotations

import inspect
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from taime_api.db.engine import get_session
from taime_api.db.models import Job
from taime_api.training.cancel import TrainingCancelled
from taime_api.training.device import hf_device_kwargs, resolve_device
from taime_api.training.hf_common import decide_model_source, resolve_hf_hparams
from taime_api.training.split import safe_train_val_test_split

logger = logging.getLogger(__name__)

# Where the DistilBERT-cased base model is loaded from for offline training.
# Defaults under DATA_DIR so it matches where scripts/fetch_models.py writes
# (DATA_DIR/hf/<model>); the Docker images bake the model outside the /data
# volume and point FAKE_NEWS_MODEL_DIR at it so the bind mount cannot shadow it.
_DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DEFAULT_FAKE_NEWS_BASE_DIR = Path(
    os.getenv("FAKE_NEWS_MODEL_DIR") or str(_DATA_DIR / "hf" / "distilbert-base-cased")
)


def _is_cancelled(job_id: int) -> bool:
    with get_session() as session:
        job = session.get(Job, job_id)
        return bool(job and job.status == "cancelled")


class CancelTrainingCallback(TrainerCallback):
    def __init__(self, job_id: int) -> None:
        self.job_id = job_id

    def _check_cancelled(self) -> None:
        if _is_cancelled(self.job_id):
            raise TrainingCancelled()

    def on_train_begin(self, args, state, control, **kwargs):
        self._check_cancelled()
        return control

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._check_cancelled()
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self._check_cancelled()
        return control

    def on_prediction_step(self, args, state, control, **kwargs):
        self._check_cancelled()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if _is_cancelled(self.job_id):
            control.should_training_stop = True
            control.should_save = False
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        self._check_cancelled()
        return control


class TextDataset(Dataset):
    def __init__(self, encodings: dict[str, Any], labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


def _find_file(dataset_path: Path, name: str) -> Path | None:
    if dataset_path.is_dir():
        matches = list(dataset_path.rglob(name))
        if matches:
            return matches[0]
        return None
    if dataset_path.name == name:
        return dataset_path
    if dataset_path.parent:
        matches = list(dataset_path.parent.rglob(name))
        if matches:
            return matches[0]
    return None


def _clean_reuters_prefix(text: str) -> str:
    pattern = r"^[A-Z]+\\s\\[[^\\]]+\\]\\s*"
    return re.sub(pattern, "", text).strip()


def _load_fake_news_dataframe(dataset_path: Path) -> pd.DataFrame:
    fake_path = _find_file(dataset_path, "Fake.csv")
    true_path = _find_file(dataset_path, "True.csv")
    if not fake_path or not true_path:
        raise ValueError("Fake.csv/True.csv not found for fake news training")

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 0
    true_df["label"] = 1

    for df in (fake_df, true_df):
        for col in ("title", "text"):
            if col not in df.columns:
                df[col] = ""
        df["title"] = df["title"].fillna("").astype(str)
        df["text"] = df["text"].fillna("").astype(str)

    true_df["text"] = true_df["text"].apply(_clean_reuters_prefix)

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df["combined_text"] = (df["title"].str.strip() + " " + df["text"].str.strip()).str.strip()
    df = df[["combined_text", "label"]].dropna()
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def train_fake_news_model(
    dataset_path: Path,
    config: dict[str, Any],
    models_dir: Path,
    job_id: int,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    logger.info("Loading fake news dataset from %s", dataset_path)
    df = _load_fake_news_dataframe(dataset_path)

    max_samples = config.get("max_train_samples")
    if max_samples:
        df = df.head(int(max_samples))

    train_df, val_df, test_df, split_notes = safe_train_val_test_split(
        df,
        label_col="label",
        test_split=float(config.get("test_split", 0.2)),
    )

    hparams = resolve_hf_hparams(
        config,
        default_epochs=3,
        default_batch_size=16,
        default_lr=5e-6,
        default_weight_decay=0.1,
    )
    device = resolve_device(config)
    set_seed(hparams["seed"])

    # DistilBERT-cased is the partner-approved base. It loads from a local asset
    # dir (offline by default); see decide_model_source for the download fallback.
    model_dir = Path(config.get("model_dir") or DEFAULT_FAKE_NEWS_BASE_DIR)
    base_model = str(config.get("base_model", "distilbert-base-cased"))
    model_src, local_only = decide_model_source(
        model_dir, base_model, bool(os.getenv("TAIME_ALLOW_HF_DOWNLOAD"))
    )
    tokenizer = AutoTokenizer.from_pretrained(model_src, local_files_only=local_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_src,
        num_labels=2,
        local_files_only=local_only,
    )

    max_length = int(config.get("max_length", 128))
    train_encodings = tokenizer(
        train_df["combined_text"].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    val_encodings = tokenizer(
        val_df["combined_text"].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    test_encodings = tokenizer(
        test_df["combined_text"].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    train_dataset = TextDataset(train_encodings, train_df["label"].tolist())
    val_dataset = TextDataset(val_encodings, val_df["label"].tolist())
    test_dataset = TextDataset(test_encodings, test_df["label"].tolist())

    if progress_callback:
        progress_callback(10.0)

    output_dir = models_dir / f"fake_news_training_{job_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if _is_cancelled(job_id):
        raise TrainingCancelled()

    eval_key = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments).parameters
        else "evaluation_strategy"
    )
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        **hparams,
        **{eval_key: "epoch"},
        **hf_device_kwargs(device),
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to="none",
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[CancelTrainingCallback(job_id)],
    )

    trainer.train()

    if _is_cancelled(job_id):
        raise TrainingCancelled()

    if progress_callback:
        progress_callback(70.0)

    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    report = classification_report(
        labels,
        preds,
        labels=[0, 1],
        target_names=["Fake", "True"],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n_train_samples": len(train_dataset),
        "n_val_samples": len(val_dataset),
        "n_test_samples": len(test_dataset),
        "model_name": model_src,
        "max_length": max_length,
        "train_sample_limit": max_samples,
        "hyperparameters": {**hparams, "device": device},
    }
    if split_notes:
        metrics["split_notes"] = split_notes

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"fake_news_model_{job_id}"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    if progress_callback:
        progress_callback(100.0)

    return model_path, metrics
