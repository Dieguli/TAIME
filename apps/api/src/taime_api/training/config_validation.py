"""Per-SUT validation of training-job hyperparameters.

Dependency-free (no FastAPI / torch) so it is unit-testable and reusable. The
service layer catches :class:`ConfigValidationError` and turns it into an
HTTP 422 response. Only values that are *present* are validated — omitted keys
fall back to the trainer defaults — so the partner-approved sets/ranges are
enforced without forcing the caller to send every field.
"""

from __future__ import annotations

from typing import Any

from taime_api.training.port_common import (
    BATCH_SIZES,
    DROPOUT_RANGE,
    LR_FACTOR_RANGE,
    LR_PATIENCE_RANGE,
    LR_RANGE,
    NORM_TYPES,
)

ALLOWED_DEVICES = {"auto", "cpu", "cuda", "gpu"}
_BOOL_STRINGS = {"true", "false", "1", "0", "yes", "no", "y", "n"}

PORT_SUT = "infra_port"
TRANSFORMER_SUTS = {"disinfo_fake", "disinfo_hate"}


class ConfigValidationError(ValueError):
    """Raised when a job config violates the allowed hyperparameter sets/ranges."""


def validate_job_config(sut_type: str, config: dict[str, Any] | None) -> None:
    """Validate a job config for the given SUT type, raising on the first batch of errors."""
    config = config or {}
    errors: list[str] = []

    device = config.get("device")
    if device is None:
        device = config.get("accelerator")
    if device is not None and str(device).lower() not in ALLOWED_DEVICES:
        errors.append(f"device must be one of {sorted(ALLOWED_DEVICES)}; got {device!r}")

    if sut_type == PORT_SUT:
        _validate_port(config, errors)
    else:
        _validate_common_training(config, errors)
        if sut_type in TRANSFORMER_SUTS:
            _validate_transformer(config, errors)

    if errors:
        raise ConfigValidationError("Invalid hyperparameters: " + "; ".join(errors))


# --- helpers ----------------------------------------------------------------
def _present(config: dict[str, Any], key: str) -> bool:
    return key in config and config[key] is not None


def _check_range(
    config: dict[str, Any],
    key: str,
    lo: float,
    hi: float,
    errors: list[str],
    *,
    integer: bool = False,
) -> None:
    if not _present(config, key):
        return
    try:
        value = float(config[key])
    except (TypeError, ValueError):
        errors.append(f"{key} must be a number")
        return
    if integer and value != int(value):
        errors.append(f"{key} must be an integer")
        return
    if not (lo <= value <= hi):
        errors.append(f"{key} must be within [{lo}, {hi}]; got {config[key]}")


def _check_min_int(config: dict[str, Any], key: str, minimum: int, errors: list[str]) -> None:
    if not _present(config, key):
        return
    try:
        value = int(config[key])
    except (TypeError, ValueError):
        errors.append(f"{key} must be an integer")
        return
    if value < minimum:
        errors.append(f"{key} must be >= {minimum}; got {value}")


def _check_bool(config: dict[str, Any], key: str, errors: list[str]) -> None:
    if not _present(config, key):
        return
    value = config[key]
    if isinstance(value, bool):
        return
    if str(value).lower() not in _BOOL_STRINGS:
        errors.append(f"{key} must be a boolean")


def _validate_port(config: dict[str, Any], errors: list[str]) -> None:
    if _present(config, "batch_size"):
        try:
            batch = int(config["batch_size"])
            if batch not in BATCH_SIZES:
                errors.append(f"batch_size must be one of {list(BATCH_SIZES)}; got {batch}")
        except (TypeError, ValueError):
            errors.append("batch_size must be an integer")

    _check_range(config, "dropout", *DROPOUT_RANGE, errors)
    _check_range(config, "learning_rate", *LR_RANGE, errors)
    _check_range(config, "lr_scheduler_factor", *LR_FACTOR_RANGE, errors)
    _check_range(config, "lr_scheduler_patience", *LR_PATIENCE_RANGE, errors, integer=True)

    if _present(config, "norm_type") and str(config["norm_type"]) not in NORM_TYPES:
        errors.append(f"norm_type must be one of {list(NORM_TYPES)}; got {config['norm_type']!r}")

    _check_bool(config, "use_reversible_instance_norm", errors)
    _check_bool(config, "normalize_before", errors)
    _check_min_int(config, "epochs", 1, errors)
    _check_min_int(config, "n_epochs", 1, errors)


def _validate_common_training(config: dict[str, Any], errors: list[str]) -> None:
    _check_min_int(config, "epochs", 1, errors)
    _check_min_int(config, "batch_size", 1, errors)
    if _present(config, "learning_rate"):
        try:
            lr = float(config["learning_rate"])
            if lr <= 0:
                errors.append("learning_rate must be a positive number")
        except (TypeError, ValueError):
            errors.append("learning_rate must be a number")


def _validate_transformer(config: dict[str, Any], errors: list[str]) -> None:
    if _present(config, "weight_decay"):
        try:
            if float(config["weight_decay"]) < 0:
                errors.append("weight_decay must be >= 0")
        except (TypeError, ValueError):
            errors.append("weight_decay must be a number")
    _check_min_int(config, "warmup_steps", 0, errors)
    if _present(config, "seed"):
        try:
            int(config["seed"])
        except (TypeError, ValueError):
            errors.append("seed must be an integer")
