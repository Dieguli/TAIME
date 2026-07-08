"""Dataset splitting helpers with small-sample safeguards."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split


def _can_stratify(labels: Iterable[int], train_size: int, test_size: int) -> bool:
    counts = pd.Series(labels).value_counts()
    if len(counts) < 2:
        return False
    if counts.min() < 2:
        return False
    return not (train_size < len(counts) or test_size < len(counts))


def safe_train_val_test_split(
    df: pd.DataFrame,
    label_col: str,
    test_split: float,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Split into train/val/test with guardrails for tiny datasets."""
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")
    total = len(df)
    if total < 3:
        raise ValueError(
            "Dataset too small to split into train/val/test. Provide at least 3 samples."
        )

    test_size = int(round(total * test_split))
    test_size = max(2, min(total - 1, test_size))
    train_size = total - test_size
    if train_size < 1:
        raise ValueError(
            "Training split would be empty. Increase dataset size or reduce test_split."
        )

    notes: list[str] = []
    stratify = df[label_col] if _can_stratify(df[label_col], train_size, test_size) else None
    if stratify is None:
        notes.append("Stratified split disabled for train/test due to small class counts.")

    train_df, temp_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    if len(temp_df) < 2:
        raise ValueError("Validation/test split would be empty. Increase dataset size.")

    val_size = max(1, len(temp_df) // 2)
    test_size_2 = len(temp_df) - val_size
    if test_size_2 < 1:
        test_size_2 = 1
        val_size = len(temp_df) - 1

    stratify_temp = (
        temp_df[label_col] if _can_stratify(temp_df[label_col], val_size, test_size_2) else None
    )
    if stratify_temp is None:
        notes.append("Stratified split disabled for val/test due to small class counts.")

    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size_2,
        random_state=seed,
        stratify=stratify_temp,
    )

    return train_df, val_df, test_df, notes
