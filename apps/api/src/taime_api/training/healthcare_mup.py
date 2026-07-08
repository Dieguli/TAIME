"""Healthcare (MUP) training utilities."""

from __future__ import annotations

import logging
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from taime_api.db.engine import get_session
from taime_api.db.models import Job
from taime_api.training.cancel import TrainingCancelled
from taime_api.training.xgb_common import resolve_xgb_params

logger = logging.getLogger(__name__)

EXPECTED_FEATURE_ORDER = [
    "age_1",
    "age_2",
    "age_3",
    "age_4",
    "age_5",
    "age_6",
    "age_7",
    "comorbidities_1",
    "comorbidities_2",
    "comorbidities_3",
    "comorbidities_4",
    "comorbidities_5",
    "comorbidities_6",
    "comorbidities_7",
    "comorbidities_8",
    "comorbidities_9",
    "comorbidities_11",
    "comorbidities_12",
    "comorbidities_13",
    "comorbidities_14",
    "comorbidities_15",
    "comorbidities_16",
    "comorbidities_17",
    "comorbidities_18",
    "gender_1",
    "gender_2",
    "morbidity_history_1",
    "morbidity_history_2",
    "morbidity_history_3",
    "morbidity_history_4",
    "morbidity_history_5",
    "morbidity_history_6",
    "morbidity_history_8",
    "morbidity_history_9",
    "negative_habits_1",
    "negative_habits_2",
    "negative_habits_3",
    "negative_habits_4",
    "negative_habits_5",
    "negative_habits_6",
    "negative_habits_7",
    "negative_habits_8",
    "negative_habits_9",
    "negative_habits_10",
    "physical_finding_1",
    "physical_finding_2",
    "physical_finding_3",
    "physical_finding_4",
    "physical_finding_5",
    "physical_finding_7",
    "physical_finding_8",
    "physical_finding_9",
    "physical_finding_10",
    "physical_finding_11",
    "physical_finding_12",
    "symptoms_1",
    "symptoms_2",
    "symptoms_3",
    "symptoms_4",
    "symptoms_5",
    "symptoms_6",
    "symptoms_7",
    "symptoms_8",
    "symptoms_9",
    "symptoms_10",
    "symptoms_11",
    "symptoms_12",
    "symptoms_13",
    "symptoms_14",
    "symptoms_15",
    "symptoms_16",
    "symptoms_17",
    "symptoms_18",
    "symptoms_19",
    "symptoms_20",
    "symptoms_21",
    "symptoms_22",
    "symptoms_23",
    "symptoms_24",
]

HEALTHCARE_FEATURE_FILES = ("mup_risk_model_data.csv", "MUP_data_processed.csv")
HEALTHCARE_LABEL_FILES = ("y_train_model_data.csv",)
LABEL_MAPPING = {
    "low": 0,
    "moderate": 1,
    "high": 2,
    "0": 0,
    "1": 1,
    "2": 2,
}


def _read_csv_flexible(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if len(df.columns) == 1 and ";" in df.columns[0]:
        df = pd.read_csv(file_path, sep=";")
    return df


def _find_first(dataset_dir: Path, filenames: tuple[str, ...]) -> Path | None:
    for name in filenames:
        matches = list(dataset_dir.rglob(name))
        if matches:
            return matches[0]
    return None


def _encode_labels(labels: pd.Series) -> np.ndarray:
    encoded: list[int] = []
    for value in labels:
        value_str = str(value).strip().lower()
        if value_str in LABEL_MAPPING:
            encoded.append(LABEL_MAPPING[value_str])
        else:
            try:
                num_val = int(float(value_str))
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Unknown risk value: {value}") from exc
            if num_val not in (0, 1, 2):
                raise ValueError(f"Unknown risk value: {value}")
            encoded.append(num_val)
    return np.array(encoded)


def _resolve_healthcare_paths(dataset_path: Path) -> tuple[Path, Path | None]:
    if dataset_path.is_dir():
        feature_path = _find_first(dataset_path, HEALTHCARE_FEATURE_FILES)
        label_path = _find_first(dataset_path, HEALTHCARE_LABEL_FILES)
    else:
        feature_path = dataset_path
        label_path = _find_first(dataset_path.parent, HEALTHCARE_LABEL_FILES)
    if feature_path is None:
        raise ValueError("Healthcare feature file not found")
    return feature_path, label_path


def load_healthcare_data(dataset_path: Path) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    feature_path, label_path = _resolve_healthcare_paths(dataset_path)
    df = _read_csv_flexible(feature_path)
    label_column = None

    if label_path is not None and label_path.exists():
        labels_df = pd.read_csv(label_path)
        label_column = "risk" if "risk" in labels_df.columns else labels_df.columns[0]
        labels = _encode_labels(labels_df[label_column])
    else:
        for candidate in ("risk_ground_truth_decoded", "risk", "risk_prediction"):
            if candidate in df.columns:
                label_column = candidate
                labels = _encode_labels(df[candidate])
                df = df.drop(columns=[candidate])
                break
        else:  # pragma: no cover - defensive
            raise ValueError("Healthcare label column not found")

    if list(df.columns) != EXPECTED_FEATURE_ORDER:
        raise ValueError("Healthcare feature order does not match expected 79-column schema")

    meta = {
        "label_column": label_column,
        "feature_count": len(df.columns),
        "row_count": len(df),
    }
    return df, labels, meta


def train_healthcare_model(
    dataset_path: Path,
    config: dict[str, Any],
    models_dir: Path,
    job_id: int,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    logger.info("Loading healthcare dataset from %s", dataset_path)
    features, labels, meta = load_healthcare_data(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(
        features.values,
        labels,
        test_size=float(config.get("test_split", 0.2)),
        random_state=int(config.get("random_state", 42)),
        stratify=labels,
    )

    with get_session() as session:
        job = session.get(Job, job_id)
        if job and job.status == "cancelled":
            raise TrainingCancelled()

    if progress_callback:
        progress_callback(10.0)

    xgb_params = resolve_xgb_params(config)
    model = XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        **xgb_params,
    )
    model.fit(X_train, y_train)

    with get_session() as session:
        job = session.get(Job, job_id)
        if job and job.status == "cancelled":
            raise TrainingCancelled()

    if progress_callback:
        progress_callback(70.0)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Low", "Moderate", "High"],
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
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "n_features": meta["feature_count"],
        "label_column": meta["label_column"],
        "device": xgb_params["device"],
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"healthcare_pc_model_{job_id}.pkl"
    with open(model_path, "wb") as handle:
        pickle.dump(
            {
                "model": model,
                "feature_order": EXPECTED_FEATURE_ORDER,
                "label_column": meta["label_column"],
                "config": config,
            },
            handle,
        )

    if progress_callback:
        progress_callback(100.0)

    return model_path, metrics
