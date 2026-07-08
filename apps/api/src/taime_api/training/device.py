"""Shared device (CPU/GPU) resolution for training jobs.

A single place to decide whether a retraining job runs on CPU or CUDA, and to
translate that decision into the framework-specific knobs used by the four SUT
trainers:

* HuggingFace ``Trainer``  -> ``TrainingArguments`` device kwargs (fake news, hate speech)
* XGBoost ``XGBClassifier`` -> ``device`` + ``tree_method`` (healthcare)
* Darts / PyTorch-Lightning -> ``pl_trainer_kwargs={"accelerator": ...}`` (port TSMixer)

The canonical config key is ``device`` (``"auto"`` | ``"cpu"`` | ``"cuda"``). The
legacy ``accelerator`` key (used by the existing port config) is also accepted.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)

Device = Literal["cpu", "cuda"]

_CPU_ALIASES = {"cpu"}
_CUDA_ALIASES = {"cuda", "gpu"}
_AUTO_ALIASES = {"auto", ""}


class DeviceUnavailableError(RuntimeError):
    """Raised when CUDA is explicitly requested but no CUDA device is available."""


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch is a hard dependency here
        return False


def _read_request(config: dict[str, Any] | None) -> str:
    """Read the requested device from config, defaulting to 'auto'."""
    config = config or {}
    raw = config.get("device")
    if raw is None:
        raw = config.get("accelerator")  # back-compat with existing port config
    if raw is None:
        return "auto"
    return str(raw).strip().lower()


def resolve_device(config: dict[str, Any] | None) -> Device:
    """Resolve a job config into a canonical device string ('cpu' or 'cuda').

    * ``cpu`` -> always CPU.
    * ``cuda`` / ``gpu`` -> CUDA; raises :class:`DeviceUnavailableError` if no
      CUDA device is present (mirrors the existing port accelerator validation).
    * ``auto`` (default) / unknown -> CUDA if available, otherwise CPU.
    """
    request = _read_request(config)

    if request in _CPU_ALIASES:
        return "cpu"

    if request in _CUDA_ALIASES:
        if not _cuda_available():
            raise DeviceUnavailableError(
                "GPU/CUDA device requested but none is available. Use device='cpu' "
                "or device='auto', or run on a CUDA-enabled host with a CUDA-enabled "
                "torch build."
            )
        return "cuda"

    if request not in _AUTO_ALIASES:
        logger.warning("Unknown device request %r; falling back to auto-detect.", request)

    if _cuda_available():
        logger.info("Device resolved to CUDA (auto).")
        return "cuda"
    logger.info("Device resolved to CPU (auto).")
    return "cpu"


def hf_device_kwargs(device: Device) -> dict[str, Any]:
    """``TrainingArguments`` kwargs for the requested device.

    On CUDA, enables fp16 mixed precision. On CPU, forces the Trainer off any
    auto-detected accelerator using the correct kwarg for the installed
    transformers version (``use_cpu`` is modern; ``no_cuda`` is the older name) —
    detected via :func:`inspect.signature`, the same approach the trainers use
    for ``eval_strategy`` vs ``evaluation_strategy``.
    """
    if device == "cuda":
        return {"fp16": True}

    try:
        from transformers import TrainingArguments

        params = inspect.signature(TrainingArguments.__init__).parameters
    except Exception:  # pragma: no cover - transformers is a hard dependency here
        params = {}
    cpu_key = "use_cpu" if "use_cpu" in params else "no_cuda"
    return {cpu_key: True}


def xgb_device_kwargs(device: Device) -> dict[str, Any]:
    """XGBoost (>=2.0) kwargs: ``device`` + ``tree_method`` (``gpu_hist`` is deprecated)."""
    return {"device": device, "tree_method": "hist"}


def darts_accelerator(device: Device) -> str:
    """PyTorch-Lightning accelerator string for Darts ``pl_trainer_kwargs``."""
    return "gpu" if device == "cuda" else "cpu"
