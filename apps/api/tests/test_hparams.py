"""Tests for hyperparameter resolution + model-source decisions (no torch required)."""

import pytest

from taime_api.training.hf_common import decide_model_source, resolve_hf_hparams
from taime_api.training.port_common import resolve_port_hparams, structural_overrides_from_config
from taime_api.training.xgb_common import resolve_xgb_params


def test_hf_defaults_match_fake_news_spec():
    h = resolve_hf_hparams(
        {}, default_epochs=3, default_batch_size=16, default_lr=5e-6, default_weight_decay=0.1
    )
    assert h["num_train_epochs"] == 3
    assert h["per_device_train_batch_size"] == 16
    assert h["per_device_eval_batch_size"] == 16
    assert h["learning_rate"] == 5e-6
    assert h["weight_decay"] == 0.1
    assert h["warmup_steps"] == 0
    assert h["seed"] == 42


def test_hf_overrides_match_hate_speech_spec():
    h = resolve_hf_hparams(
        {
            "epochs": 5,
            "batch_size": 16,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "warmup_steps": 500,
            "seed": 7,
        },
        default_epochs=2,
        default_batch_size=8,
        default_lr=2e-5,
        default_weight_decay=0.0,
        default_warmup_steps=0,
    )
    assert h["num_train_epochs"] == 5
    assert h["per_device_train_batch_size"] == 16
    assert h["learning_rate"] == 5e-5
    assert h["weight_decay"] == 0.01
    assert h["warmup_steps"] == 500
    assert h["seed"] == 7


def test_decide_model_source_prefers_local(tmp_path):
    model_dir = tmp_path / "distilbert-base-cased"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    src, local_only = decide_model_source(model_dir, "distilbert-base-cased", allow_download=False)
    assert src == str(model_dir)
    assert local_only is True


def test_decide_model_source_downloads_when_allowed(tmp_path):
    src, local_only = decide_model_source(
        tmp_path / "missing", "distilbert-base-cased", allow_download=True
    )
    assert src == "distilbert-base-cased"
    assert local_only is False


def test_decide_model_source_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        decide_model_source(tmp_path / "missing", "distilbert-base-cased", allow_download=False)


def test_port_hparams_defaults():
    h = resolve_port_hparams({})
    assert h["batch_size"] == 64
    assert h["n_epochs"] == 10
    assert h["norm_type"] == "LayerNorm"
    assert h["use_reversible_instance_norm"] is False
    assert h["normalize_before"] is False


def test_port_hparams_overrides():
    h = resolve_port_hparams(
        {
            "epochs": 12,
            "batch_size": 256,
            "dropout": 0.5,
            "learning_rate": 1e-5,
            "lr_scheduler_factor": 0.1,
            "lr_scheduler_patience": 10,
            "use_reversible_instance_norm": True,
            "normalize_before": True,
            "norm_type": "TimeBatchNorm2d",
        }
    )
    assert h["n_epochs"] == 12
    assert h["batch_size"] == 256
    assert h["dropout"] == 0.5
    assert h["learning_rate"] == 1e-5
    assert h["lr_scheduler_factor"] == 0.1
    assert h["lr_scheduler_patience"] == 10
    assert h["norm_type"] == "TimeBatchNorm2d"
    assert h["normalize_before"] is True
    assert h["use_reversible_instance_norm"] is True


def test_structural_overrides():
    assert structural_overrides_from_config({"batch_size": 128}) == {}
    overrides = structural_overrides_from_config(
        {"hidden_size": 128, "num_blocks": 3, "activation": "GELU"}
    )
    assert overrides == {"hidden_size": 128, "num_blocks": 3, "activation": "GELU"}


def test_xgb_params_defaults_cpu(monkeypatch):
    monkeypatch.setattr("taime_api.training.device._cuda_available", lambda: False)
    params = resolve_xgb_params({})
    assert params["n_estimators"] == 120
    assert params["max_depth"] == 6
    assert params["learning_rate"] == 0.1
    assert params["random_state"] == 42
    assert params["device"] == "cpu"
    assert params["tree_method"] == "hist"


def test_xgb_params_maps_epochs_and_gpu(monkeypatch):
    monkeypatch.setattr("taime_api.training.device._cuda_available", lambda: True)
    params = resolve_xgb_params({"epochs": 200, "learning_rate": 0.05, "device": "cuda"})
    assert params["n_estimators"] == 200  # epochs -> n_estimators for XGBoost
    assert params["learning_rate"] == 0.05
    assert params["device"] == "cuda"
    assert params["tree_method"] == "hist"
