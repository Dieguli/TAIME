"""Shared, dependency-free helpers for the HuggingFace transformer trainers.

Kept free of ``torch``/``transformers`` imports so the hyperparameter resolution
can be unit-tested without the heavy ML stack installed. The trainers feed the
returned dict straight into ``TrainingArguments(**kwargs, ...)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_hf_hparams(
    config: dict[str, Any] | None,
    *,
    default_epochs: int,
    default_batch_size: int,
    default_lr: float,
    default_weight_decay: float,
    default_warmup_steps: int = 0,
    default_seed: int = 42,
) -> dict[str, Any]:
    """Resolve the partner-approved transformer hyperparameters from job config.

    Returns the numeric ``TrainingArguments`` kwargs (epochs, batch size, learning
    rate, weight decay, warmup steps, seed). The optimizer is HuggingFace's default
    ``adamw_torch`` (AdamW), so it is intentionally not overridden here.
    """
    config = config or {}
    return {
        "num_train_epochs": int(config.get("epochs", default_epochs)),
        "per_device_train_batch_size": int(config.get("batch_size", default_batch_size)),
        "per_device_eval_batch_size": int(config.get("batch_size", default_batch_size)),
        "learning_rate": float(config.get("learning_rate", default_lr)),
        "weight_decay": float(config.get("weight_decay", default_weight_decay)),
        "warmup_steps": int(config.get("warmup_steps", default_warmup_steps)),
        "seed": int(config.get("seed", default_seed)),
    }


def decide_model_source(
    model_dir: Path,
    base_model: str,
    allow_download: bool,
) -> tuple[str, bool]:
    """Decide where to load a base model from, returning ``(name_or_path, local_files_only)``.

    * A populated local directory wins -> load it offline (``local_files_only=True``).
    * Otherwise, if downloads are allowed, load the Hub repo id (``local_files_only=False``).
    * Otherwise raise a clear :class:`FileNotFoundError` that names the expected path —
      never let ``local_files_only=True`` raise an opaque error deep in transformers.
    """
    model_dir = Path(model_dir)
    if model_dir.exists() and model_dir.is_dir() and any(model_dir.iterdir()):
        return str(model_dir), True
    if allow_download:
        return base_model, False
    raise FileNotFoundError(
        f"Base model assets not found at '{model_dir}'. Provide them there "
        f"(e.g. run scripts/fetch_models.py to fetch '{base_model}'), or set "
        "TAIME_ALLOW_HF_DOWNLOAD=1 to download from the HuggingFace Hub at runtime."
    )
