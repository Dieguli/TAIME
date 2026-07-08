"""Validation for the SINTEF port time-series dataset package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from taime_api.utils.port_dataset import (
    PortDatasetError,
    PortDatasetSummary,
    load_meta,
    resolve_port_dataset_root,
    summarize_meta,
)


@dataclass
class PortValidationResult:
    """Validation result for port datasets."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: PortDatasetSummary | None = None


def validate_port_dataset(path: Path) -> PortValidationResult:
    """Validate the port dataset folder structure and metadata."""
    try:
        root = resolve_port_dataset_root(path)
        series_meta = load_meta(root / "series_meta.json")
        past_meta = load_meta(root / "past_covs_meta.json")
    except PortDatasetError as exc:
        return PortValidationResult(valid=False, errors=[str(exc)])

    errors: list[str] = []
    warnings: list[str] = []
    summary = summarize_meta(series_meta, past_meta)

    if summary.series_count == 0:
        errors.append("Series metadata is empty")
    if summary.past_covariates_count == 0:
        errors.append("Past covariates metadata is empty")
    if summary.series_count != summary.past_covariates_count:
        errors.append("Series and past covariates counts do not match")

    if summary.series_columns and summary.series_columns != {1}:
        errors.append(f"Unexpected series column counts: {sorted(summary.series_columns)}")
    if summary.past_covariates_columns and summary.past_covariates_columns != {121}:
        warnings.append(
            "Past covariates column count differs from expected 121: "
            f"{sorted(summary.past_covariates_columns)}"
        )

    if summary.index_names and summary.index_names != {"date"}:
        warnings.append(f"Unexpected index names: {sorted(summary.index_names)}")

    if summary.static_covariate_keys and "NumEscala" not in summary.static_covariate_keys:
        warnings.append(
            f"Static covariate key 'NumEscala' not found: {sorted(summary.static_covariate_keys)}"
        )

    # Verify referenced parquet files exist
    missing_files = []
    for entry in series_meta:
        file_name = entry.get("file")
        if isinstance(file_name, str) and not (root / file_name).exists():
            missing_files.append(file_name)
    for entry in past_meta:
        file_name = entry.get("file")
        if isinstance(file_name, str) and not (root / file_name).exists():
            missing_files.append(file_name)

    if missing_files:
        errors.append(f"Missing parquet files: {missing_files[:5]}")

    # Verify model artifacts exist anywhere in the package
    pt_files = [p for p in path.rglob("*.pt") if not p.name.endswith(".pt.ckpt")]
    ckpt_files = list(path.rglob("*.ckpt"))
    if not pt_files:
        errors.append("Missing model file (.pt)")
    elif len(pt_files) > 1:
        errors.append("Multiple .pt model files found")
    if not ckpt_files:
        errors.append("Missing model checkpoint (.ckpt)")

    return PortValidationResult(
        valid=len(errors) == 0, errors=errors, warnings=warnings, summary=summary
    )
