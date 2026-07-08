"""Tests for per-SUT job-config validation (no torch required)."""

import pytest

from taime_api.training.config_validation import ConfigValidationError, validate_job_config


@pytest.mark.parametrize(
    "config",
    [
        {},  # empty -> defaults apply, valid
        {
            "batch_size": 128,
            "dropout": 0.3,
            "learning_rate": 1e-3,
            "lr_scheduler_factor": 0.5,
            "lr_scheduler_patience": 5,
            "norm_type": "TimeBatchNorm2d",
            "use_reversible_instance_norm": True,
            "normalize_before": False,
        },
        {"batch_size": 64, "norm_type": "LayerNorm", "device": "cuda"},
    ],
)
def test_port_valid(config):
    validate_job_config("infra_port", config)


@pytest.mark.parametrize(
    "config",
    [
        {"batch_size": 32},  # not in {64,128,256}
        {"dropout": 0.9},  # outside [0.2, 0.5]
        {"learning_rate": 1e-9},  # below 1e-5
        {"learning_rate": 0.5},  # above 1e-3
        {"lr_scheduler_factor": 1.5},  # outside [0.1, 0.9]
        {"lr_scheduler_patience": 99},  # outside [1, 10]
        {"norm_type": "Nope"},  # not an allowed norm
    ],
)
def test_port_invalid(config):
    with pytest.raises(ConfigValidationError):
        validate_job_config("infra_port", config)


def test_transformer_valid():
    validate_job_config(
        "disinfo_fake",
        {"epochs": 3, "batch_size": 16, "learning_rate": 5e-6, "weight_decay": 0.1, "seed": 42},
    )
    validate_job_config(
        "disinfo_hate",
        {
            "epochs": 5,
            "batch_size": 16,
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "learning_rate": 5e-5,
        },
    )


@pytest.mark.parametrize(
    "sut,config",
    [
        ("disinfo_fake", {"learning_rate": -1}),
        ("disinfo_fake", {"epochs": 0}),
        ("disinfo_hate", {"warmup_steps": -5}),
        ("disinfo_fake", {"weight_decay": -0.1}),
        ("disinfo_fake", {"device": "tpu"}),
        ("healthcare_pc", {"epochs": 0}),
        ("healthcare_pc", {"batch_size": 0}),
    ],
)
def test_invalid(sut, config):
    with pytest.raises(ConfigValidationError):
        validate_job_config(sut, config)


def test_all_devices_accepted():
    for device in ("auto", "cpu", "cuda", "gpu"):
        validate_job_config("disinfo_fake", {"device": device})
