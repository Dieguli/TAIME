"""Runtime capability API endpoint."""

from __future__ import annotations

import importlib
import os
from typing import Any

from fastapi import APIRouter

from taime_api.api.v1.schemas import RuntimeResponse

router = APIRouter()


def _torch_capabilities() -> tuple[bool, str | None, str | None]:
    """Return CUDA capability details without making torch a hard import."""
    try:
        torch: Any = importlib.import_module("torch")
    except Exception:
        return False, None, None

    torch_version = str(getattr(torch, "__version__", "")) or None

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        return False, None, torch_version

    device_name = None
    if cuda_available:
        try:
            device_name = str(torch.cuda.get_device_name(0))
        except Exception:
            device_name = None

    return cuda_available, device_name, torch_version


@router.get("", response_model=RuntimeResponse)
async def get_runtime() -> RuntimeResponse:
    """Expose runtime capabilities needed by the demo operator UI."""
    cuda_available, device_name, torch_version = _torch_capabilities()
    return RuntimeResponse(
        version="2.0.0",
        image=os.getenv("TAIME_IMAGE", "unknown"),
        cuda_available=cuda_available,
        device_name=device_name,
        torch_version=torch_version,
    )
