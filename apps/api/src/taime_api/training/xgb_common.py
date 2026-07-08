"""Dependency-free XGBoost hyperparameter resolution for the healthcare (MUP) SUT.

Free of xgboost imports so the resolution (including cpu/gpu device selection) is
unit-testable without the ML stack installed.
"""

from __future__ import annotations

from typing import Any

from taime_api.training.device import resolve_device, xgb_device_kwargs


def resolve_xgb_params(config: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve XGBClassifier kwargs (including cpu/gpu device) from job config.

    Adds XGBoost>=2.0 device selection: ``device='cuda'`` + ``tree_method='hist'``
    on GPU, ``device='cpu'`` otherwise (the deprecated ``gpu_hist`` is avoided).
    """
    config = config or {}
    params: dict[str, Any] = {
        "n_estimators": int(config.get("n_estimators", config.get("epochs", 120))),
        "max_depth": int(config.get("max_depth", 6)),
        "learning_rate": float(config.get("learning_rate", 0.1)),
        "random_state": int(config.get("random_state", config.get("seed", 42))),
    }
    params.update(xgb_device_kwargs(resolve_device(config)))
    return params
