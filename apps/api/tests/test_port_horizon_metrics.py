"""Full-horizon port metrics built on the partner's PrimerSecuenciaUnosDistance.

TAIME's last-point backtest only scores the 48th step ahead, where nearly every
window is 1, so an all-ones model and the partner's seed score the same. These
tests pin the full-window metrics (incl. the partner's first-ones distance and
its persistence / all-ones baselines) and the scoped seed-class alias.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchmetrics")


def test_partner_metric_hand_computed():
    from taime_api.training.legacy_metrics import PrimerSecuenciaUnosDistance

    metric = PrimerSecuenciaUnosDistance(horizon=4)
    target = torch.tensor([[0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]])
    ones_at = torch.tensor([[0, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]])
    logits = ones_at.float() * 10 - 5  # sigmoid >= 0.5 exactly where ones_at == 1
    metric.update(logits, target)
    # |2-1| + 0 (no ones anywhere) + |4-0| (gt has none -> 4) + |0-4| (pred has none -> 4)
    assert metric.compute().item() == pytest.approx((1 + 0 + 4 + 4) / 4)


def test_horizon_window_metrics_scores_model_and_baselines():
    from taime_api.training.port_tsmixer import horizon_window_metrics

    actual = np.array([[0, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=float)
    last_observed = np.array([0.0, 0.0, 1.0])
    predicted_ones = np.array([[0, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=float)
    logits = predicted_ones * 10 - 5  # a perfect model

    m = horizon_window_metrics(logits, actual, last_observed, threshold=0.5, horizon=4)

    assert m["horizon_windows"] == 3
    assert m["first_ones_distance"] == pytest.approx(0.0)
    assert m["horizon_accuracy"] == pytest.approx(1.0)
    assert m["horizon_f1"] == pytest.approx(1.0)
    assert m["horizon_f1_negative"] == pytest.approx(1.0)
    assert m["horizon_predicted_positive_rate"] == pytest.approx(6 / 12)
    assert m["horizon_actual_positive_rate"] == pytest.approx(6 / 12)
    # persistence repeats the last observed state: window 0 -> all 0 (gt starts at 2),
    # window 1 -> all 0 (both empty), window 2 -> all 1 (both start at 0)
    assert m["first_ones_distance_persistence"] == pytest.approx((2 + 0 + 0) / 3)
    # all-ones always starts at 0: |2-0| + |4-0| + 0
    assert m["first_ones_distance_all_ones"] == pytest.approx((2 + 4 + 0) / 3)
    assert m["first_ones_distance_all_ones"] > m["first_ones_distance_persistence"]


def test_horizon_window_metrics_flags_an_all_ones_model():
    from taime_api.training.port_tsmixer import horizon_window_metrics

    actual = np.array([[0, 0, 0, 0], [0, 0, 1, 1]], dtype=float)
    logits = np.full((2, 4), 0.01)  # the degenerate retrain: logits just above 0

    m = horizon_window_metrics(logits, actual, np.array([0.0, 0.0]), threshold=0.5, horizon=4)

    assert m["horizon_predicted_positive_rate"] == pytest.approx(1.0)
    assert m["horizon_f1_negative"] == pytest.approx(0.0)
    assert m["first_ones_distance"] == pytest.approx(m["first_ones_distance_all_ones"])


class FakeWindowModel:
    """historical_forecasts stand-in: each window's logits mirror the actual values."""

    def __init__(self, input_chunk_length=3, scale=10.0):
        self.input_chunk_length = input_chunk_length
        self.scale = scale

    def historical_forecasts(self, *, series, forecast_horizon, stride, start=None, **_):
        out = []
        first = max(self.input_chunk_length, start or 0)
        for ts in series:
            windows = []
            for start_pos in range(first, len(ts) - forecast_horizon + 1, stride):
                actual = ts[start_pos : start_pos + forecast_horizon]
                logits = (actual.all_values() - 0.5) * self.scale
                windows.append(actual.with_values(logits))
            out.append(windows)
        return out


def test_run_horizon_backtest_aligns_windows_and_persistence():
    pytest.importorskip("darts")
    import pandas as pd
    from darts import TimeSeries

    from taime_api.training.port_tsmixer import parse_port_config, run_horizon_backtest

    index = pd.date_range("2024-01-01", periods=10, freq="10min")
    values = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1], dtype=np.float32)
    series = [TimeSeries.from_times_and_values(index, values)]
    config = parse_port_config({"forecast_horizon": 4, "stride": 1})

    m = run_horizon_backtest(FakeWindowModel(), series, [None], config)

    # windows start at positions 3..6 -> 4 windows, and the fake model is perfect
    assert m["horizon_windows"] == 4
    assert m["first_ones_distance"] == pytest.approx(0.0)
    assert m["horizon_accuracy"] == pytest.approx(1.0)
    # persistence uses the value just before each window: positions 2,3,4,5 -> 0,1,1,0
    # windows: [1,1,0,0] [1,0,0,1] [0,0,1,1] [0,1,1,1]; gt starts 0,0,2,1
    # persistence all-0 starts at 4 (no ones): |0-4|, all-1 at 0: |0-0|, |2-0|, all-0: |1-4|
    assert m["first_ones_distance_persistence"] == pytest.approx((4 + 0 + 2 + 3) / 4)


def test_seed_first_ones_params_are_read_from_the_seed(tmp_path):
    from taime_api.training import legacy_metrics
    from taime_api.training.port_tsmixer import read_seed_model, seed_first_ones_params

    cls = legacy_metrics.PrimerSecuenciaUnosDistance
    main = sys.modules["__main__"]
    original_module = cls.__module__
    cls.__module__ = "__main__"  # pickled exactly as the partner's training script did
    main.PrimerSecuenciaUnosDistance = cls
    try:
        seed = {"torch_metrics": {"psud": cls(longitud_secuencia=3, horizon=48)}}
        torch.save(seed, tmp_path / "seed.pt")
    finally:
        cls.__module__ = original_module
        delattr(main, "PrimerSecuenciaUnosDistance")

    loaded, unresolved = read_seed_model(tmp_path / "seed.pt")

    assert unresolved == ["__main__.PrimerSecuenciaUnosDistance"]
    params = seed_first_ones_params(loaded["torch_metrics"])
    assert params == {"longitud_secuencia": 3, "threshold": 0.5}


def test_scoped_legacy_alias_loads_partner_pickles_and_restores_main(tmp_path):
    from taime_api.training import legacy_metrics
    from taime_api.training.port_tsmixer import legacy_seed_classes

    cls = legacy_metrics.PrimerSecuenciaUnosDistance
    main = sys.modules["__main__"]
    original_module = cls.__module__
    cls.__module__ = "__main__"
    main.PrimerSecuenciaUnosDistance = cls
    try:
        torch.save({"metric": cls(longitud_secuencia=2)}, tmp_path / "seed.pt")
    finally:
        cls.__module__ = original_module
        delattr(main, "PrimerSecuenciaUnosDistance")

    with pytest.raises(AttributeError, match="PrimerSecuenciaUnosDistance"):
        torch.load(tmp_path / "seed.pt", weights_only=False)

    with legacy_seed_classes():
        loaded = torch.load(tmp_path / "seed.pt", weights_only=False)

    assert isinstance(loaded["metric"], cls)
    assert loaded["metric"].longitud_secuencia == 2
    assert not hasattr(main, "PrimerSecuenciaUnosDistance")


def test_scoped_legacy_alias_never_overrides_an_existing_main_attribute():
    from taime_api.training.port_tsmixer import legacy_seed_classes

    main = sys.modules["__main__"]
    sentinel = object()
    main.PrimerSecuenciaUnosDistance = sentinel
    try:
        with legacy_seed_classes():
            assert main.PrimerSecuenciaUnosDistance is sentinel
        assert main.PrimerSecuenciaUnosDistance is sentinel
    finally:
        delattr(main, "PrimerSecuenciaUnosDistance")


def test_retrain_reports_horizon_metrics_on_held_out_series(monkeypatch, tmp_path):
    from taime_api.training import port_tsmixer
    from taime_api.training.port_common import PORT_STRUCTURAL_DEFAULTS

    series = [[0.0] * 200 + [i] for i in range(30)]
    covs = [f"cov-{i}" for i in range(30)]
    seen: dict = {}

    class FakePortModel:
        def fit(self, *, series, past_covariates):
            pass

        def save(self, path, clean=False):
            Path(path).write_bytes(b"model")

    def fake_horizon(model, series, past_covariates, config, first_ones_params=None, **kwargs):
        seen["series"] = series
        seen["params"] = first_ones_params
        return {"first_ones_distance": 3.5, "horizon_windows": 7}

    monkeypatch.setattr(port_tsmixer, "load_port_timeseries", lambda d, m: (series, covs))
    monkeypatch.setattr(
        port_tsmixer,
        "_resolve_seed_recipe",
        lambda d, c, h, apply_overrides=True: (
            dict(PORT_STRUCTURAL_DEFAULTS),
            {"architecture_source": "seed", "seed_first_ones_params": {"longitud_secuencia": 3}},
        ),
    )
    monkeypatch.setattr(port_tsmixer, "build_tsmixer_model", lambda *a, **k: FakePortModel())
    monkeypatch.setattr(port_tsmixer, "run_backtest", lambda *a, **k: {"accuracy": 1.0})
    monkeypatch.setattr(port_tsmixer, "run_horizon_backtest", fake_horizon)
    monkeypatch.setattr(port_tsmixer, "_make_port_progress_callback", lambda *_: None)
    monkeypatch.setattr(port_tsmixer, "_is_port_job_cancelled", lambda job_id: False)

    result = port_tsmixer.retrain_port_model(
        dataset_dir=tmp_path, config={"backtest_max_series": 5}, models_dir=tmp_path, job_id=1
    )

    assert result.metrics["first_ones_distance"] == 3.5
    assert result.metrics["accuracy"] == 1.0
    assert len(seen["series"]) == result.metrics["backtest_series"]
    assert seen["params"] == {"longitud_secuencia": 3}


def test_unloaded_network_keys_catches_keys_missing_from_the_checkpoint():
    from taime_api.training.port_tsmixer import _unloaded_network_keys

    loaded = {
        "fc.weight": torch.ones(2),
        "fc.bias": torch.zeros(1),
        "criterion.pos_weight": torch.ones(1),
    }
    saved = {"fc.weight": torch.ones(2)}  # fc.bias absent: it would stay randomly initialised

    assert _unloaded_network_keys(loaded, saved) == ["fc.bias"]
    assert _unloaded_network_keys(loaded, {**saved, "fc.bias": torch.zeros(1)}) == []


def test_partner_metric_uses_the_seed_threshold():
    from taime_api.training.port_tsmixer import horizon_window_metrics

    actual = np.array([[0, 0, 1, 1]], dtype=float)
    # sigmoid(1.0) = 0.73: a 1 at threshold 0.5, a 0 at the seed's 0.8
    logits = np.array([[1.0, 1.0, 5.0, 5.0]])

    default = horizon_window_metrics(logits, actual, np.array([0.0]), threshold=0.5, horizon=4)
    seeded = horizon_window_metrics(
        logits, actual, np.array([0.0]), threshold=0.5, horizon=4, first_ones_threshold=0.8
    )

    assert default["first_ones_distance"] == pytest.approx(2.0)  # predicted start 0 vs 2
    assert seeded["first_ones_distance"] == pytest.approx(0.0)  # predicted start 2 vs 2
    assert seeded["first_ones_threshold"] == pytest.approx(0.8)
    assert seeded["horizon_accuracy"] == default["horizon_accuracy"]  # classification keeps 0.5


def test_run_horizon_backtest_honours_a_common_start():
    pytest.importorskip("darts")
    import pandas as pd
    from darts import TimeSeries

    from taime_api.training.port_tsmixer import parse_port_config, run_horizon_backtest

    index = pd.date_range("2024-01-01", periods=12, freq="10min")
    series = [TimeSeries.from_times_and_values(index, np.zeros(12, dtype=np.float32))]
    config = parse_port_config({"forecast_horizon": 4, "stride": 1})

    short = run_horizon_backtest(FakeWindowModel(input_chunk_length=3), series, [None], config)
    common = run_horizon_backtest(
        FakeWindowModel(input_chunk_length=3), series, [None], config, start_position=6
    )

    assert short["horizon_windows"] == 6  # origins 3..8
    assert common["horizon_windows"] == 3  # origins 6..8, same as a model with input length 6


def test_retrain_scores_model_and_seed_from_the_same_forecast_origins(monkeypatch, tmp_path):
    from taime_api.training import port_tsmixer
    from taime_api.training.port_common import PORT_STRUCTURAL_DEFAULTS

    series = [[0.0] * 200 + [i] for i in range(30)]
    covs = [f"cov-{i}" for i in range(30)]
    calls: list = []

    class FakePortModel:
        input_chunk_length = 48

        def fit(self, *, series, past_covariates, **kwargs):
            pass

        def save(self, path, clean=False):
            Path(path).write_bytes(b"model")

    class FakeSeed:
        input_chunk_length = 96  # e.g. the seed differs from a scratch override

    def fake_horizon(model, series, past_covariates, config, first_ones_params=None, **kwargs):
        calls.append((type(model).__name__, kwargs.get("start_position")))
        return {key: 1.0 for key in port_tsmixer._SEED_SCORE_KEYS}

    monkeypatch.setattr(port_tsmixer, "load_port_timeseries", lambda d, m: (series, covs))
    monkeypatch.setattr(
        port_tsmixer,
        "_resolve_seed_recipe",
        lambda d, c, h, apply_overrides=True: (
            dict(PORT_STRUCTURAL_DEFAULTS),
            {"architecture_source": "seed"},
        ),
    )
    monkeypatch.setattr(
        port_tsmixer, "resolve_model_paths", lambda d, c: type("P", (), {"model_path": d})()
    )
    monkeypatch.setattr(
        port_tsmixer, "load_tsmixer_model", lambda path, accelerator="cpu": FakeSeed()
    )
    monkeypatch.setattr(port_tsmixer, "build_tsmixer_model", lambda *a, **k: FakePortModel())
    monkeypatch.setattr(port_tsmixer, "run_backtest", lambda *a, **k: {})
    monkeypatch.setattr(port_tsmixer, "run_horizon_backtest", fake_horizon)
    monkeypatch.setattr(port_tsmixer, "_make_port_progress_callback", lambda *_: None)
    monkeypatch.setattr(port_tsmixer, "_is_port_job_cancelled", lambda job_id: False)

    result = port_tsmixer.retrain_port_model(
        dataset_dir=tmp_path, config={}, models_dir=tmp_path, job_id=1
    )

    assert calls == [("FakePortModel", 96), ("FakeSeed", 96)]
    assert result.metrics["seed_first_ones_distance"] == 1.0


def test_evaluation_skips_short_series_and_keeps_horizon_metrics_if_last_point_fails(
    monkeypatch, tmp_path
):
    from taime_api.training import port_tsmixer

    short, long_ = [0.0] * 50, [0.0] * 200
    seen: dict = {}

    class Seed:
        input_chunk_length = 48
        model_params: dict = {}

    def failing_backtest(model, series, past_covariates, config):
        raise ValueError("series too short")

    def fake_horizon(model, series, past_covariates, config, **kwargs):
        seen["series"] = series
        return {"first_ones_distance": 2.0}

    monkeypatch.setattr(
        port_tsmixer, "load_port_timeseries", lambda d, m: ([short, long_], ["c-short", "c-long"])
    )
    monkeypatch.setattr(
        port_tsmixer,
        "resolve_model_paths",
        lambda d, c: port_tsmixer.PortModelPaths(model_path=tmp_path / "seed.pt", ckpt_path=None),
    )
    monkeypatch.setattr(port_tsmixer, "load_tsmixer_model", lambda path, accelerator="cpu": Seed())
    monkeypatch.setattr(port_tsmixer, "run_backtest", failing_backtest)
    monkeypatch.setattr(port_tsmixer, "run_horizon_backtest", fake_horizon)

    result = port_tsmixer.run_port_evaluation(tmp_path, {"forecast_horizon": 48})

    assert seen["series"] == [long_]  # 50 < 48 + 48: no full window, left out
    assert result.metrics["first_ones_distance"] == 2.0
    assert "too short" in result.metrics["backtest_error"]
    assert result.metrics["series_used"] == 1
