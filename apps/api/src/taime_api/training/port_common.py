"""Dependency-free helpers and allowed ranges for the port (Darts TSMixer) retraining config.

Free of ``torch``/``darts`` imports so the hyperparameter resolution and the
validation ranges can be unit-tested (and reused by the job-config validator)
without the heavy ML stack installed.

Allowed sets/ranges come from the partner specification (Daniel Diosdado):
    batch_size: [64, 128, 256]
    dropout: [0.2, 0.5]
    lr: [1e-5, 1e-3]
    lr_scheduler_factor: [0.1, 0.9]
    lr_scheduler_patience: [1, 10]
    use_reversible_instance_norm / normalize_before: {True, False}
    norm_type: ['LayerNorm', 'LayerNormNoBias', 'TimeBatchNorm2d']
Architecture params (hidden_size/ff_size/num_blocks) are NOT tuned on retrain.
"""

from __future__ import annotations

from typing import Any

# --- Partner-approved allowed sets / ranges (inclusive) ---------------------
NORM_TYPES: tuple[str, ...] = ("LayerNorm", "LayerNormNoBias", "TimeBatchNorm2d")
BATCH_SIZES: tuple[int, ...] = (64, 128, 256)
DROPOUT_RANGE: tuple[float, float] = (0.2, 0.5)
LR_RANGE: tuple[float, float] = (1e-5, 1e-3)
LR_FACTOR_RANGE: tuple[float, float] = (0.1, 0.9)
LR_PATIENCE_RANGE: tuple[int, int] = (1, 10)

# Structural defaults (used only when they cannot be recovered from the seed model).
# They mirror the partner's production seed (TSMixer_Study_V7_NumEscala) so a failed
# seed read still trains the same architecture instead of an unrelated one.
PORT_STRUCTURAL_DEFAULTS: dict[str, Any] = {
    "input_chunk_length": 48,
    "output_chunk_length": 48,
    "hidden_size": 32,
    "ff_size": 256,
    "num_blocks": 1,
    "activation": "Sigmoid",
}
_STRUCTURAL_INT_KEYS = (
    "input_chunk_length",
    "output_chunk_length",
    "hidden_size",
    "ff_size",
    "num_blocks",
)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes", "y"}


def resolve_port_hparams(config: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve Daniel's tunable TSMixer hyperparameters from job config.

    ``epochs`` (the generic UI field) maps to Darts' ``n_epochs``.
    """
    config = config or {}
    epochs = config.get("epochs", config.get("n_epochs"))
    return {
        "batch_size": _to_int(config.get("batch_size"), 64),
        "n_epochs": max(1, _to_int(epochs, 10)),
        "dropout": _to_float(config.get("dropout"), 0.2),
        "learning_rate": _to_float(config.get("learning_rate"), 1e-3),
        "lr_scheduler_factor": _to_float(config.get("lr_scheduler_factor"), 0.5),
        "lr_scheduler_patience": _to_int(config.get("lr_scheduler_patience"), 5),
        # Off by default: on a binary target RIN de-normalizes a flat (all-0/all-1)
        # input window to a logit ~= the window mean, so sigmoid >= 0.5 predicts 1
        # everywhere unless trained with a far larger lr than the approved range.
        "use_reversible_instance_norm": _to_bool(config.get("use_reversible_instance_norm"), False),
        "normalize_before": _to_bool(config.get("normalize_before"), False),
        "norm_type": str(config.get("norm_type") or "LayerNorm"),
    }


def structural_overrides_from_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Pull any explicitly-pinned structural params from config (optional overrides)."""
    config = config or {}
    overrides: dict[str, Any] = {}
    for key in _STRUCTURAL_INT_KEYS:
        if config.get(key) is not None:
            overrides[key] = _to_int(config.get(key), PORT_STRUCTURAL_DEFAULTS[key])
    if config.get("activation"):
        overrides["activation"] = str(config["activation"])
    return overrides
