"""Tests for the shared CPU/GPU device resolver (no torch required)."""

import pytest

from taime_api.training.device import (
    DeviceUnavailableError,
    darts_accelerator,
    hf_device_kwargs,
    resolve_device,
    xgb_device_kwargs,
)


def test_resolve_cpu_explicit():
    assert resolve_device({"device": "cpu"}) == "cpu"
    assert resolve_device({"device": "CPU"}) == "cpu"


def test_resolve_auto_without_cuda(monkeypatch):
    monkeypatch.setattr("taime_api.training.device._cuda_available", lambda: False)
    assert resolve_device({"device": "auto"}) == "cpu"
    assert resolve_device({}) == "cpu"
    assert resolve_device(None) == "cpu"


def test_resolve_auto_with_cuda(monkeypatch):
    monkeypatch.setattr("taime_api.training.device._cuda_available", lambda: True)
    assert resolve_device({"device": "auto"}) == "cuda"


def test_resolve_cuda_unavailable_raises(monkeypatch):
    monkeypatch.setattr("taime_api.training.device._cuda_available", lambda: False)
    with pytest.raises(DeviceUnavailableError):
        resolve_device({"device": "cuda"})
    # legacy accelerator key is honored too
    with pytest.raises(DeviceUnavailableError):
        resolve_device({"accelerator": "gpu"})


def test_resolve_cuda_available(monkeypatch):
    monkeypatch.setattr("taime_api.training.device._cuda_available", lambda: True)
    assert resolve_device({"device": "cuda"}) == "cuda"
    assert resolve_device({"accelerator": "gpu"}) == "cuda"


def test_xgb_device_kwargs():
    assert xgb_device_kwargs("cuda") == {"device": "cuda", "tree_method": "hist"}
    assert xgb_device_kwargs("cpu") == {"device": "cpu", "tree_method": "hist"}


def test_darts_accelerator():
    assert darts_accelerator("cuda") == "gpu"
    assert darts_accelerator("cpu") == "cpu"


def test_hf_device_kwargs_cuda():
    assert hf_device_kwargs("cuda") == {"fp16": True}


def test_hf_device_kwargs_cpu():
    kwargs = hf_device_kwargs("cpu")
    # Exactly one CPU-forcing flag, named for the installed transformers version.
    assert list(kwargs.values()) == [True]
    assert next(iter(kwargs)) in {"use_cpu", "no_cuda"}
