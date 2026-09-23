"""Port (Darts TSMixer) retraining recipe regression tests.

Covers the partner-reported failures: retrain silently trained on 20 series,
built an MSE regressor instead of a logit classifier and kept reversible
instance norm on (both yield "only produces 1s"), and fell back to an unrelated
architecture when the seed pickle referenced a class from the partner's
``__main__``.
"""

import sys
from pathlib import Path

import pytest

from taime_api.services.jobs import _normalize_port_config
from taime_api.training.port_common import PORT_STRUCTURAL_DEFAULTS

torch = pytest.importorskip("torch")


def test_normalize_port_config_trains_on_all_series_by_default():
    assert _normalize_port_config({})["max_series"] is None
    assert _normalize_port_config({"max_series": 0})["max_series"] is None
    assert _normalize_port_config({"max_series": 50})["max_series"] == 50
    assert _normalize_port_config({})["backtest_max_series"] == 20


def test_parse_port_config_keeps_all_series():
    from taime_api.training.port_tsmixer import parse_port_config

    assert parse_port_config({}).max_series is None
    assert parse_port_config({"max_series": None}).max_series is None
    assert parse_port_config({"max_series": -1}).max_series is None
    assert parse_port_config({"max_series": 30}).max_series == 30
    assert parse_port_config({}).backtest_max_series == 20
    assert parse_port_config({"backtest_max_series": 0}).backtest_max_series is None
    # The job service stores None for "all"; the parser must not turn it back into 20.
    for raw in (0, -1):
        normalized = _normalize_port_config({"max_series": raw, "backtest_max_series": raw})
        parsed = parse_port_config(normalized)
        assert parsed.max_series is None
        assert parsed.backtest_max_series is None
    assert parse_port_config(_normalize_port_config({})).max_series is None
    assert parse_port_config(_normalize_port_config({})).backtest_max_series == 20


def test_structural_defaults_match_partner_production_architecture():
    assert PORT_STRUCTURAL_DEFAULTS == {
        "input_chunk_length": 48,
        "output_chunk_length": 48,
        "hidden_size": 32,
        "ff_size": 256,
        "num_blocks": 1,
        "activation": "Sigmoid",
    }


class FakeSeedModel:
    """Minimal stand-in for a pickled TSMixerModel (only what the reader uses)."""

    def __init__(self, params):
        self.model_params = params
        self.input_chunk_length = params["input_chunk_length"]
        self.output_chunk_length = params["output_chunk_length"]


def _save_seed_with_main_only_metric(path: Path) -> None:
    """Pickle a seed whose metric class only exists in the trainer's ``__main__``."""

    class PrimerSecuenciaUnosDistance:
        pass

    PrimerSecuenciaUnosDistance.__module__ = "__main__"
    PrimerSecuenciaUnosDistance.__qualname__ = "PrimerSecuenciaUnosDistance"
    main = sys.modules["__main__"]
    main.PrimerSecuenciaUnosDistance = PrimerSecuenciaUnosDistance
    try:
        seed = FakeSeedModel(
            {
                "input_chunk_length": 48,
                "output_chunk_length": 48,
                "hidden_size": 32,
                "ff_size": 256,
                "num_blocks": 1,
                "activation": "Sigmoid",
                "loss_fn": torch.nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([2.3078], dtype=torch.float64)
                ),
                "torch_metrics": PrimerSecuenciaUnosDistance(),
            }
        )
        torch.save(seed, path)
    finally:
        delattr(main, "PrimerSecuenciaUnosDistance")


def test_plain_torch_load_fails_on_main_only_class(tmp_path):
    """Baseline: the partner's error, reproduced without darts."""
    path = tmp_path / "seed.pt"
    _save_seed_with_main_only_metric(path)
    with pytest.raises(AttributeError, match="PrimerSecuenciaUnosDistance"):
        torch.load(path, weights_only=False)


def test_seed_reader_tolerates_main_only_class(tmp_path):
    from taime_api.training.port_tsmixer import extract_structural_params, read_seed_model

    path = tmp_path / "seed.pt"
    _save_seed_with_main_only_metric(path)

    seed, unresolved = read_seed_model(path)

    assert unresolved == ["__main__.PrimerSecuenciaUnosDistance"]
    assert extract_structural_params(seed) == {
        "input_chunk_length": 48,
        "output_chunk_length": 48,
        "hidden_size": 32,
        "ff_size": 256,
        "num_blocks": 1,
        "activation": "Sigmoid",
    }
    assert isinstance(seed.model_params["loss_fn"], torch.nn.BCEWithLogitsLoss)


def test_seed_loss_info_reports_class_weight():
    from taime_api.training.port_tsmixer import seed_loss_info

    seed_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.3078], dtype=torch.float64))
    info = seed_loss_info(seed_loss)

    assert info["seed_loss_fn"] == "BCEWithLogitsLoss"
    assert info["seed_pos_weight"] == pytest.approx(2.3078)
    assert seed_loss_info(None) == {"seed_loss_fn": None}
    assert seed_loss_info(torch.nn.MSELoss()) == {"seed_loss_fn": "MSELoss"}


def test_build_tsmixer_model_trains_logits_with_unweighted_bce():
    pytest.importorskip("darts")
    from taime_api.training.port_common import resolve_port_hparams
    from taime_api.training.port_tsmixer import build_tsmixer_model

    model = build_tsmixer_model(dict(PORT_STRUCTURAL_DEFAULTS), resolve_port_hparams({}), "cpu")
    loss = model.model_params["loss_fn"]

    assert isinstance(loss, torch.nn.BCEWithLogitsLoss)
    # Loss buffers (e.g. pos_weight) land in the checkpoint and break Darts' reload.
    assert loss.state_dict() == {}
    # RIN pins logits of flat binary windows near 0/1 -> sigmoid >= 0.5 everywhere.
    assert model.model_params["use_reversible_instance_norm"] is False


def test_retrained_artifact_round_trips_through_plain_darts_load(monkeypatch, tmp_path):
    """fit -> save(clean=True) -> TSMixerModel.load -> predict, as the partner's TAI-LA does."""
    pytest.importorskip("darts")
    import numpy as np
    import pandas as pd
    from darts import TimeSeries
    from darts.models import TSMixerModel

    from taime_api.training.port_common import resolve_port_hparams
    from taime_api.training.port_tsmixer import build_tsmixer_model

    monkeypatch.chdir(tmp_path)  # keep Lightning logs out of the repo
    rng = np.random.default_rng(0)
    index = pd.date_range("2024-01-01", periods=60, freq="10min")
    series = [
        TimeSeries.from_times_and_values(index, (rng.random(60) > 0.5).astype(np.float32))
        for _ in range(3)
    ]
    covariates = [
        TimeSeries.from_times_and_values(index, rng.random((60, 2)).astype(np.float32))
        for _ in range(3)
    ]
    structural = {
        "input_chunk_length": 12,
        "output_chunk_length": 6,
        "hidden_size": 8,
        "ff_size": 8,
        "num_blocks": 1,
        "activation": "Sigmoid",
    }
    model = build_tsmixer_model(structural, resolve_port_hparams({"epochs": 1}), "cpu")
    model.fit(series=series, past_covariates=covariates, verbose=False)
    path = tmp_path / "port_model.pt"
    model.save(str(path), clean=True)

    loaded = TSMixerModel.load(str(path), map_location="cpu")
    forecast = loaded.predict(n=6, series=series[0], past_covariates=covariates[0], verbose=False)

    assert forecast.n_timesteps == 6


def test_retrain_filters_short_series_and_backtests_held_out(monkeypatch, tmp_path):
    from taime_api.training import port_tsmixer

    # 30 series; entries 0 and 1 are shorter than input+output (48 + 48).
    series = [[0.0] * (50 if i < 2 else 200) + [i] for i in range(30)]
    covs = [f"cov-{i}" for i in range(30)]
    fitted: dict = {}
    backtested: dict = {}

    class FakePortModel:
        def fit(self, *, series, past_covariates):
            fitted["series"] = series
            fitted["covs"] = past_covariates

        def save(self, path, clean=False):
            Path(path).write_bytes(b"model")

    def fake_backtest(model, series, past_covariates, config):
        backtested["series"] = series
        backtested["covs"] = past_covariates
        backtested["last_points_only"] = config.last_points_only
        return {"accuracy": 1.0}

    monkeypatch.setattr(port_tsmixer, "load_port_timeseries", lambda d, m: (series, covs))
    monkeypatch.setattr(
        port_tsmixer,
        "_resolve_seed_recipe",
        lambda d, c, h: (
            dict(PORT_STRUCTURAL_DEFAULTS),
            {"architecture_source": "seed", "seed_pos_weight": 2.3},
        ),
    )
    monkeypatch.setattr(port_tsmixer, "build_tsmixer_model", lambda *a, **k: FakePortModel())
    monkeypatch.setattr(port_tsmixer, "run_backtest", fake_backtest)
    monkeypatch.setattr(port_tsmixer, "_make_port_progress_callback", lambda *_: None)
    monkeypatch.setattr(port_tsmixer, "_is_port_job_cancelled", lambda job_id: False)

    result = port_tsmixer.retrain_port_model(
        dataset_dir=tmp_path, config={"backtest_max_series": 5}, models_dir=tmp_path, job_id=1
    )

    usable = series[2:]
    held_out = backtested["series"]
    assert 1 <= len(held_out) <= 5
    assert all(any(s is u for u in usable) for s in held_out)
    assert not any(s is h for s in fitted["series"] for h in held_out)
    assert len(fitted["series"]) + len(held_out) == len(usable)
    # covariates stay paired with their series
    for s, c in zip(fitted["series"], fitted["covs"], strict=True):
        assert c == f"cov-{s[-1]}"
    assert result.metrics["series_skipped_too_short"] == 2
    assert result.metrics["series_used"] == len(fitted["series"])
    assert result.metrics["backtest_series"] == len(held_out)
    assert result.metrics["metrics_scope"] == "held_out_series"
    assert result.metrics["architecture_source"] == "seed"
    assert result.metrics["seed_pos_weight"] == 2.3
    assert result.metrics["hyperparameters"]["loss_fn"] == "BCEWithLogitsLoss"
    assert backtested["last_points_only"] is True
