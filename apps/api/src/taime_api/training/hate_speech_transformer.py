"""Hate speech transformer training utilities."""

from __future__ import annotations

import inspect
import json
import logging
import os
import shutil
import tempfile
import zipfile
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
from taime_api.training.hf_common import resolve_hf_hparams
from taime_api.training.split import safe_train_val_test_split

logger = logging.getLogger(__name__)


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


DEFAULT_HATE_SPEECH_ZIP = Path(
    os.getenv(
        "HATE_SPEECH_MODEL_ZIP",
        "models_datasets/FakeNews_HateSpeech_model.zip",
    )
)
DEFAULT_ASSETS_DIR = Path(os.getenv("HATE_SPEECH_ASSETS_DIR", "data/hf/hate_speech"))


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


def _ensure_assets(dataset_path: Path) -> dict[str, Path]:
    tokenizer_dir = DEFAULT_ASSETS_DIR / "tokenizer"
    binary_dir = DEFAULT_ASSETS_DIR / "binary"
    multilabel_dir = DEFAULT_ASSETS_DIR / "multilabel"

    def _extract_assets_from_archive(archive: zipfile.ZipFile) -> bool:
        extracted = False
        for member in archive.infolist():
            name = member.filename
            if name.endswith("/"):
                continue
            if "/models/tokenizer/" in name:
                target = tokenizer_dir / Path(name).name
            elif "/models/ROBERTA_hate_speech/" in name:
                target = binary_dir / Path(name).name
            elif "/models/ROBERTA_hate_speech_multilabel/" in name:
                target = multilabel_dir / Path(name).name
            else:
                continue
            target.write_bytes(archive.read(name))
            extracted = True

        # Fallback: pick tokenizer files by filename if paths differ.
        for member in archive.infolist():
            name = member.filename
            if name.endswith("/"):
                continue
            base = Path(name).name
            if base in {
                "vocab.json",
                "merges.txt",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            }:
                target = tokenizer_dir / base
                if not target.exists():
                    target.write_bytes(archive.read(name))
                    extracted = True
        return extracted

    def _extract_assets_from_zip(path: Path) -> bool:
        try:
            with zipfile.ZipFile(path) as archive:
                if _extract_assets_from_archive(archive):
                    return True
                for member in archive.infolist():
                    name = member.filename.lower()
                    if not name.endswith(".zip"):
                        continue
                    with (
                        archive.open(member) as source,
                        tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp,
                    ):
                        shutil.copyfileobj(source, tmp)
                        tmp_path = Path(tmp.name)
                    try:
                        with zipfile.ZipFile(tmp_path) as nested:
                            if _extract_assets_from_archive(nested):
                                return True
                    except zipfile.BadZipFile:
                        continue
                    finally:
                        tmp_path.unlink(missing_ok=True)
        except zipfile.BadZipFile:
            return False
        return False

    def _ensure_tokenizer_config() -> None:
        config_path = tokenizer_dir / "config.json"
        if not config_path.exists():
            for source_dir in (binary_dir, multilabel_dir):
                source = source_dir / "config.json"
                if source.exists():
                    config_path.write_bytes(source.read_bytes())
                    break
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        if "model_type" not in payload:
            payload["model_type"] = "roberta"
            config_path.write_text(json.dumps(payload), encoding="utf-8")

    def _ensure_tokenizer_files() -> None:
        required = {"vocab.json", "merges.txt"}
        missing = [name for name in required if not (tokenizer_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                "Tokenizer files missing in hate speech assets: "
                + ", ".join(missing)
                + ". Include tokenizer files in the model zip."
            )

    if tokenizer_dir.exists() and binary_dir.exists() and multilabel_dir.exists():
        _ensure_tokenizer_config()
        try:
            _ensure_tokenizer_files()
        except FileNotFoundError:
            logger.warning(
                "Hate speech assets are incomplete. Attempting to rehydrate from model zip."
            )
        else:
            return {
                "tokenizer": tokenizer_dir,
                "binary": binary_dir,
                "multilabel": multilabel_dir,
            }

    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    binary_dir.mkdir(parents=True, exist_ok=True)
    multilabel_dir.mkdir(parents=True, exist_ok=True)

    zip_candidates: list[Path] = []
    if dataset_path.exists():
        if dataset_path.is_file() and dataset_path.suffix.lower() == ".zip":
            zip_candidates.append(dataset_path)
        elif dataset_path.is_dir():
            zip_candidates.extend(sorted(dataset_path.rglob("*.zip")))

    if DEFAULT_HATE_SPEECH_ZIP.exists():
        zip_candidates.append(DEFAULT_HATE_SPEECH_ZIP)

    archive_path = None
    for candidate in zip_candidates:
        if not candidate.exists():
            continue
        if _extract_assets_from_zip(candidate):
            archive_path = candidate
            break

    if archive_path is None:
        raise FileNotFoundError(
            "Hate speech model zip not found. Provide it in the dataset package or set "
            "HATE_SPEECH_MODEL_ZIP."
        )

    _ensure_tokenizer_config()
    _ensure_tokenizer_files()
    return {
        "tokenizer": tokenizer_dir,
        "binary": binary_dir,
        "multilabel": multilabel_dir,
    }


def _map_target_to_label(target: str) -> str:
    if target in {"MIGRANTS", "DISABLED", "disability", "POC", "origin"}:
        return "Racism"
    if target in {"WOMEN", "gender"}:
        return "Sexism"
    if target in {"sexual_orientation", "LGBT+"}:
        return "Sexual orientation"
    if target in {"MUSLIMS", "JEWS", "religion"}:
        return "Religious"
    return "Unknown"


def _load_hate_speech_dataframe(dataset_path: Path, classification_type: str) -> pd.DataFrame:
    mlma_path = _find_file(dataset_path, "en_dataset.csv")
    conan_path = _find_file(dataset_path, "Multitarget-CONAN.csv")
    if not mlma_path or not conan_path:
        raise ValueError("en_dataset.csv or Multitarget-CONAN.csv not found")

    df_mlma = pd.read_csv(mlma_path)
    df_conan = pd.read_csv(conan_path)

    df_mlma["label"] = df_mlma["sentiment"].apply(lambda x: 0 if x == "normal" else 1)
    df_mlma["text"] = df_mlma.pop("tweet")
    df_mlma["target"] = df_mlma.get("target")
    df_mlma = df_mlma[["label", "text", "target"]]

    df_con_non_hate = df_conan[["COUNTER_NARRATIVE", "TARGET"]].copy()
    df_con_hate = df_conan[["HATE_SPEECH", "TARGET"]].copy()
    df_con_non_hate["text"] = df_con_non_hate.pop("COUNTER_NARRATIVE")
    df_con_hate["text"] = df_con_hate.pop("HATE_SPEECH")
    df_con_non_hate["label"] = 0
    df_con_hate["label"] = 1
    df_con_non_hate["target"] = df_con_non_hate.pop("TARGET")
    df_con_hate["target"] = df_con_hate.pop("TARGET")

    df_con = pd.concat([df_con_non_hate, df_con_hate], ignore_index=True)

    hs_df = pd.concat([df_con, df_mlma], ignore_index=True)
    hs_df = hs_df.dropna(subset=["text"]).drop_duplicates(subset=["text", "label"])

    if classification_type == "binary":
        hs_df["label"] = hs_df["label"].astype(int)
        return hs_df[["text", "label"]]

    # Multilabel mapping per provided preprocessing
    hs_df["mapped_target"] = hs_df["target"].astype(str).apply(_map_target_to_label)
    hs_df["label"] = hs_df.apply(
        lambda row: (
            0
            if row["label"] == 0
            else (
                1
                if row["mapped_target"] == "Racism"
                else 2
                if row["mapped_target"] == "Sexism"
                else 3
                if row["mapped_target"] == "Sexual orientation"
                else 4
                if row["mapped_target"] == "Religious"
                else -1
            )
        ),
        axis=1,
    )
    hs_df = hs_df[hs_df["label"] >= 0]
    return hs_df[["text", "label"]]


def train_hate_speech_model(
    dataset_path: Path,
    config: dict[str, Any],
    models_dir: Path,
    job_id: int,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    classification_type = str(config.get("classification_type", "binary")).lower()
    if classification_type not in {"binary", "multilabel"}:
        classification_type = "binary"

    df = _load_hate_speech_dataframe(dataset_path, classification_type)

    # The assembled frame is label-ordered (CONAN counter-narratives = label 0,
    # then CONAN hate = label 1, then MLMA). Shuffle with a seeded order BEFORE
    # truncating so a `max_train_samples` demo subset stays class-mixed instead
    # of collapsing to a single class. Mirrors the fake-news trainer, which
    # shuffles before head().
    seed = int(config.get("seed", 42))
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

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
        default_epochs=5,
        default_batch_size=16,
        default_lr=5e-5,
        default_weight_decay=0.01,
        default_warmup_steps=500,
    )
    device = resolve_device(config)
    set_seed(hparams["seed"])

    assets = _ensure_assets(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(assets["tokenizer"], local_files_only=True)
    model_dir = assets["binary"] if classification_type == "binary" else assets["multilabel"]
    num_labels = 2 if classification_type == "binary" else 5
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=num_labels,
        local_files_only=True,
    )

    max_length = int(config.get("max_length", 128))
    train_encodings = tokenizer(
        train_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    val_encodings = tokenizer(
        val_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    test_encodings = tokenizer(
        test_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    train_dataset = TextDataset(train_encodings, train_df["label"].tolist())
    val_dataset = TextDataset(val_encodings, val_df["label"].tolist())
    test_dataset = TextDataset(test_encodings, test_df["label"].tolist())

    if progress_callback:
        progress_callback(10.0)

    output_dir = models_dir / f"hate_speech_training_{job_id}"
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
    cm = confusion_matrix(labels, preds, labels=list(range(num_labels)))
    label_names = (
        ["Non Hatefull", "Hatefull"]
        if classification_type == "binary"
        else ["Normal", "Racism", "Sexism", "Sexual orientation", "Religious"]
    )
    report = classification_report(
        labels,
        preds,
        labels=list(range(num_labels)),
        target_names=label_names,
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
        "model_name": str(model_dir),
        "max_length": max_length,
        "classification_type": classification_type,
        "train_sample_limit": max_samples,
        "hyperparameters": {**hparams, "device": device},
    }
    if split_notes:
        metrics["split_notes"] = split_notes

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"hate_speech_model_{classification_type}_{job_id}"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    if progress_callback:
        progress_callback(100.0)

    return model_path, metrics
