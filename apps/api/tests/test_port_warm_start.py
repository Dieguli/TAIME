"""Warm start: fine-tune the port TSMixer from the seed's weights (opt-in).

From scratch, RIN on the binary target pins every logit near the window mean
unless trained with the seed's out-of-range lr for its full schedule; the seed's
own weights have already escaped that, so fine-tuning from them is the cheap
retrain. These tests pin the guardrails: exact weight transfer, structure taken
from the seed, and a recorded fall back to scratch on any mismatch.
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

SEED_SHAPE = [(48, 1), (48, 121), None, None, (1, 1), (48, 1)]


def _mismatch(**overrides):
    from taime_api.training.port_tsmixer import seed_shape_mismatch

    kwargs = {
        "target_width": 1,
        "covariate_width": 121,
        "static_size": 1,
        "input_chunk_length": 48,
        "output_chunk_length": 48,
    }
    kwargs.update(overrides)
    return seed_shape_mismatch(SEED_SHAPE, **kwargs)


def test_seed_shape_matches_the_partner_data():
    assert _mismatch() is None


@pytest.mark.parametrize(
    ("override", "word"),
    [
        ({"covariate_width": 120}, "covariate"),
        ({"target_width": 2}, "target"),
        ({"static_size": 0}, "static"),
        ({"input_chunk_length": 96}, "input_chunk_length"),
        ({"output_chunk_length": 24}, "output_chunk_length"),
    ],
)
def test_seed_shape_mismatch_is_reported(override, word):
    reason = _mismatch(**override)
    assert reason is not None and word in reason


def test_seed_shape_mismatch_without_recorded_shape():
    from taime_api.training.port_tsmixer import seed_shape_mismatch

    reason = seed_shape_mismatch(
        None,
        target_width=1,
        covariate_width=121,
        static_size=1,
        input_chunk_length=48,
        output_chunk_length=48,
    )
    assert reason is not None


STRUCTURAL = {
    "input_chunk_length": 12,
    "output_chunk_length": 6,
    "hidden_size": 8,
    "ff_size": 8,
    "num_blocks": 1,
    "activation": "ReLU",
}


def _synthetic(n_covariates: int = 2):
    import pandas as pd
    from darts import TimeSeries

    rng = np.random.default_rng(0)
    index = pd.date_range("2024-01-01", periods=60, freq="10min")
    series = [
        TimeSeries.from_times_and_values(index, (rng.random(60) > 0.5).astype(np.float32))
        for _ in range(3)
    ]
    covariates = [
        TimeSeries.from_times_and_values(index, rng.random((60, n_covariates)).astype(np.float32))
        for _ in range(3)
    ]
    return series, covariates


@pytest.fixture
def partner_style_seed(tmp_path, monkeypatch):
    """A tiny seed trained like the partner's: RIN on, a non-default norm, BCE pos_weight."""
    pytest.importorskip("darts")
    from darts.models import TSMixerModel

    monkeypatch.chdir(tmp_path)  # keep Lightning logs out of the repo
    series, covariates = _synthetic()
    seed = TSMixerModel(
        **{k: v for k, v in STRUCTURAL.items()},
        use_reversible_instance_norm=True,
        norm_type="LayerNormNoBias",
        loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.3])),
        n_epochs=1,
        batch_size=8,
        random_state=1,
        pl_trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": False, "logger": False},
    )
    seed.fit(series, past_covariates=covariates, verbose=False)
    path = tmp_path / "seed.pt"
    seed.save(str(path))
    return path, series, covariates


def _network_state(state_dict):
    return {
        k: v
        for k, v in state_dict.items()
        if not k.startswith(("criterion", "train_criterion", "val_criterion"))
    }


def test_warm_start_loads_every_seed_weight_and_takes_seed_structure(partner_style_seed):
    from darts.models import TSMixerModel

    from taime_api.training.port_common import resolve_port_hparams
    from taime_api.training.port_tsmixer import warm_start_from_seed

    seed_path, series, covariates = partner_style_seed
    hparams = resolve_port_hparams({"epochs": 1})  # form defaults: RIN off, LayerNorm

    model, info = warm_start_from_seed(
        seed_path, STRUCTURAL, hparams, "cpu", None, series, covariates
    )

    assert model is not None, info
    assert info["init_weights"] == "seed"
    assert info["warm_start_structure"]["use_reversible_instance_norm"] is True
    assert info["warm_start_structure"]["norm_type"] == "LayerNormNoBias"
    seed_weights = _network_state(
        torch.load(str(seed_path) + ".ckpt", map_location="cpu", weights_only=False)["state_dict"]
    )
    loaded = model.model.state_dict()
    assert seed_weights
    for key, value in seed_weights.items():
        assert torch.equal(loaded[key], value), key

    model.fit(series=series, past_covariates=covariates, verbose=False)
    assert isinstance(model.model.train_criterion, torch.nn.BCEWithLogitsLoss)
    assert model.model.train_criterion.state_dict() == {}

    out = seed_path.parent / "retrained.pt"
    model.save(str(out), clean=True)
    reloaded = TSMixerModel.load(str(out), map_location="cpu")
    forecast = reloaded.predict(n=6, series=series[0], past_covariates=covariates[0], verbose=False)
    assert forecast.n_timesteps == 6


def test_load_tsmixer_model_reads_seeds_saved_with_a_weighted_loss(partner_style_seed):
    """Darts 0.39 cannot reload its own save when the loss has pos_weight; TAIME must."""
    from darts.models import TSMixerModel

    from taime_api.training.port_tsmixer import load_tsmixer_model

    seed_path, series, covariates = partner_style_seed
    with pytest.raises(RuntimeError, match="criterion.pos_weight"):
        TSMixerModel.load(str(seed_path), map_location="cpu")

    model = load_tsmixer_model(seed_path)

    seed_weights = _network_state(
        torch.load(str(seed_path) + ".ckpt", map_location="cpu", weights_only=False)["state_dict"]
    )
    loaded = model.model.state_dict()
    for key, value in seed_weights.items():
        assert torch.equal(loaded[key], value), key
    forecast = model.predict(n=6, series=series[0], past_covariates=covariates[0], verbose=False)
    assert forecast.n_timesteps == 6


def test_warm_start_falls_back_on_covariate_mismatch(partner_style_seed):
    from taime_api.training.port_common import resolve_port_hparams
    from taime_api.training.port_tsmixer import warm_start_from_seed

    seed_path, series, _ = partner_style_seed
    _, wider_covariates = _synthetic(n_covariates=3)

    model, info = warm_start_from_seed(
        seed_path, STRUCTURAL, resolve_port_hparams({}), "cpu", None, series, wider_covariates
    )

    assert model is None
    assert "covariate" in info["warm_start_fallback_reason"]


def test_warm_start_falls_back_without_seed_weights(partner_style_seed):
    from taime_api.training.port_common import resolve_port_hparams
    from taime_api.training.port_tsmixer import warm_start_from_seed

    seed_path, series, covariates = partner_style_seed
    Path(str(seed_path) + ".ckpt").unlink()

    model, info = warm_start_from_seed(
        seed_path, STRUCTURAL, resolve_port_hparams({}), "cpu", None, series, covariates
    )

    assert model is None
    assert ".ckpt" in info["warm_start_fallback_reason"]


def _patch_retrain(monkeypatch, port_tsmixer, calls, warm_result):
    from taime_api.training.port_common import PORT_STRUCTURAL_DEFAULTS

    series = [[0.0] * 200 + [i] for i in range(30)]
    covs = [f"cov-{i}" for i in range(30)]

    class FakePortModel:
        def __init__(self, origin):
            self.origin = origin

        def fit(self, *, series, past_covariates, **kwargs):
            calls["fit_kwargs"] = kwargs
            calls["fitted_origin"] = self.origin

        def save(self, path, clean=False):
            Path(path).write_bytes(b"model")

    def fake_recipe(d, c, h, apply_overrides=True):
        calls["apply_overrides"] = apply_overrides
        return dict(PORT_STRUCTURAL_DEFAULTS), {"architecture_source": "seed"}

    def fake_warm(seed_path, structural, hparams, accelerator, callbacks, s, c):
        calls["warm_train_size"] = len(s)
        model, info = warm_result
        return (FakePortModel("seed") if model else None), info

    monkeypatch.setattr(port_tsmixer, "load_port_timeseries", lambda d, m: (series, covs))
    monkeypatch.setattr(port_tsmixer, "_resolve_seed_recipe", fake_recipe)
    monkeypatch.setattr(
        port_tsmixer, "resolve_model_paths", lambda d, c: type("P", (), {"model_path": d})()
    )
    monkeypatch.setattr(port_tsmixer, "warm_start_from_seed", fake_warm)
    monkeypatch.setattr(
        port_tsmixer, "build_tsmixer_model", lambda *a, **k: FakePortModel("scratch")
    )
    monkeypatch.setattr(port_tsmixer, "run_backtest", lambda *a, **k: {})
    monkeypatch.setattr(port_tsmixer, "run_horizon_backtest", lambda *a, **k: {})
    monkeypatch.setattr(port_tsmixer, "_make_port_progress_callback", lambda *_: None)
    monkeypatch.setattr(port_tsmixer, "_is_port_job_cancelled", lambda job_id: False)


def test_retrain_warm_starts_when_requested(monkeypatch, tmp_path):
    from taime_api.training import port_tsmixer

    calls: dict = {}
    info = {"init_weights": "seed", "warm_start_structure": {"use_reversible_instance_norm": True}}
    _patch_retrain(monkeypatch, port_tsmixer, calls, (True, info))

    result = port_tsmixer.retrain_port_model(
        dataset_dir=tmp_path,
        config={"warm_start": True, "max_samples_per_ts": 300},
        models_dir=tmp_path,
        job_id=1,
    )

    assert calls["fitted_origin"] == "seed"
    assert calls["apply_overrides"] is False
    assert calls["fit_kwargs"] == {"max_samples_per_ts": 300}
    assert result.metrics["init_weights"] == "seed"
    assert result.metrics["hyperparameters"]["use_reversible_instance_norm"] is True


def test_retrain_falls_back_to_scratch_and_records_why(monkeypatch, tmp_path):
    from taime_api.training import port_tsmixer

    calls: dict = {}
    reason = {"warm_start_fallback_reason": "seed covariate width 121 != data 120"}
    _patch_retrain(monkeypatch, port_tsmixer, calls, (False, reason))

    result = port_tsmixer.retrain_port_model(
        dataset_dir=tmp_path, config={"warm_start": True}, models_dir=tmp_path, job_id=1
    )

    assert calls["fitted_origin"] == "scratch"
    assert result.metrics["init_weights"] == "scratch"
    assert "121" in result.metrics["warm_start_fallback_reason"]


def test_retrain_warm_starts_by_default(monkeypatch, tmp_path):
    from taime_api.training import port_tsmixer

    calls: dict = {}
    _patch_retrain(monkeypatch, port_tsmixer, calls, (True, {"init_weights": "seed"}))

    result = port_tsmixer.retrain_port_model(
        dataset_dir=tmp_path, config={}, models_dir=tmp_path, job_id=1
    )

    assert calls["fitted_origin"] == "seed"
    assert calls["apply_overrides"] is False
    assert calls["fit_kwargs"] == {}
    assert result.metrics["init_weights"] == "seed"


def test_retrain_from_scratch_when_warm_start_is_off(monkeypatch, tmp_path):
    from taime_api.training import port_tsmixer

    calls: dict = {}
    _patch_retrain(monkeypatch, port_tsmixer, calls, (True, {"init_weights": "seed"}))

    result = port_tsmixer.retrain_port_model(
        dataset_dir=tmp_path, config={"warm_start": False}, models_dir=tmp_path, job_id=1
    )

    assert calls["fitted_origin"] == "scratch"
    assert "warm_train_size" not in calls
    assert calls["apply_overrides"] is True
    assert result.metrics["init_weights"] == "scratch"


def test_warm_start_fallback_restores_the_users_structural_overrides(monkeypatch, tmp_path):
    from taime_api.training import port_tsmixer
    from taime_api.training.port_common import PORT_STRUCTURAL_DEFAULTS

    calls: dict = {}
    _patch_retrain(monkeypatch, port_tsmixer, calls, (False, {"warm_start_fallback_reason": "x"}))

    def fake_recipe(d, c, h, apply_overrides=True):
        structural = dict(PORT_STRUCTURAL_DEFAULTS)
        provenance = {"architecture_source": "seed"}
        if apply_overrides:
            structural["hidden_size"] = 128
        else:
            provenance["ignored_structural_overrides"] = {"hidden_size": 128}
        return structural, provenance

    built: dict = {}

    class Scratch:
        def fit(self, *, series, past_covariates, **kwargs):
            pass

        def save(self, path, clean=False):
            Path(path).write_bytes(b"model")

    def fake_build(structural, hparams, accelerator, extra_callbacks=None):
        built["structural"] = dict(structural)
        return Scratch()

    monkeypatch.setattr(port_tsmixer, "_resolve_seed_recipe", fake_recipe)
    monkeypatch.setattr(port_tsmixer, "build_tsmixer_model", fake_build)

    result = port_tsmixer.retrain_port_model(
        dataset_dir=tmp_path,
        config={"warm_start": True, "hidden_size": 128},
        models_dir=tmp_path,
        job_id=1,
    )

    assert built["structural"]["hidden_size"] == 128
    assert result.metrics["init_weights"] == "scratch"
    assert result.metrics["hyperparameters"]["hidden_size"] == 128


def test_report_warns_that_the_seed_may_have_seen_the_holdout(monkeypatch, tmp_path):
    from taime_api.training import port_tsmixer

    calls: dict = {}
    _patch_retrain(monkeypatch, port_tsmixer, calls, (True, {"init_weights": "seed"}))

    result = port_tsmixer.retrain_port_model(
        dataset_dir=tmp_path, config={"warm_start": True}, models_dir=tmp_path, job_id=1
    )

    assert result.metrics["metrics_scope"] == "held_out_series"
    assert "seed" in result.metrics["holdout_note"]
