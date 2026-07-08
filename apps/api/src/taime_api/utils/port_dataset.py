"""Helpers for the SINTEF port (TSMixer) dataset package."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PortDatasetError(ValueError):
    """Raised when the port dataset package is invalid."""


SERIES_META_NAME = "series_meta.json"
PAST_COVS_META_NAME = "past_covs_meta.json"


@dataclass(frozen=True)
class PortDatasetSummary:
    """Lightweight summary of the port dataset package."""

    series_count: int
    past_covariates_count: int
    series_columns: set[int]
    past_covariates_columns: set[int]
    index_names: set[str]
    static_covariate_keys: set[str]
    start_values: list[str]
    end_values: list[str]


def resolve_port_dataset_root(base_path: Path) -> Path:
    """Resolve the dataset root directory that contains meta JSON files."""
    base_path = base_path.resolve()
    if _has_meta(base_path):
        return base_path

    subdirs = [item for item in base_path.iterdir() if item.is_dir()]
    candidates = [item for item in subdirs if _has_meta(item)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise PortDatasetError("Multiple port dataset roots found in uploaded package")

    raise PortDatasetError("Port dataset meta files not found in uploaded package")


def load_meta(meta_path: Path) -> list[dict[str, Any]]:
    """Load a meta JSON file and ensure it is a list of entries."""
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        raise PortDatasetError(f"Failed to read meta file: {meta_path.name}") from exc

    if not isinstance(data, list):
        raise PortDatasetError(f"Meta file {meta_path.name} must contain a list of entries")

    return data


def summarize_meta(
    series_meta: list[dict[str, Any]],
    past_covs_meta: list[dict[str, Any]],
) -> PortDatasetSummary:
    """Summarize meta information for the port dataset."""
    series_columns = set()
    past_covs_columns = set()
    index_names: set[str] = set()
    static_covariate_keys: set[str] = set()
    start_values: list[str] = []
    end_values: list[str] = []

    for entry in series_meta:
        n_cols = entry.get("n_columns")
        if isinstance(n_cols, int):
            series_columns.add(n_cols)

        idx_name = entry.get("index_name")
        if isinstance(idx_name, str):
            index_names.add(idx_name)

        sc = entry.get("static_covariates")
        if isinstance(sc, dict):
            static_covariate_keys.update(sc.keys())

        start = entry.get("start")
        end = entry.get("end")
        if isinstance(start, str):
            start_values.append(start)
        if isinstance(end, str):
            end_values.append(end)

    for entry in past_covs_meta:
        n_cols = entry.get("n_columns")
        if isinstance(n_cols, int):
            past_covs_columns.add(n_cols)

        idx_name = entry.get("index_name")
        if isinstance(idx_name, str):
            index_names.add(idx_name)

        sc = entry.get("static_covariates")
        if isinstance(sc, dict):
            static_covariate_keys.update(sc.keys())

        start = entry.get("start")
        end = entry.get("end")
        if isinstance(start, str):
            start_values.append(start)
        if isinstance(end, str):
            end_values.append(end)

    return PortDatasetSummary(
        series_count=len(series_meta),
        past_covariates_count=len(past_covs_meta),
        series_columns=series_columns,
        past_covariates_columns=past_covs_columns,
        index_names=index_names,
        static_covariate_keys=static_covariate_keys,
        start_values=start_values,
        end_values=end_values,
    )


def _has_meta(path: Path) -> bool:
    return (path / SERIES_META_NAME).exists() and (path / PAST_COVS_META_NAME).exists()
