"""TSMixer utilities for the SINTEF port use case."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import pickle
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

from taime_api.db.engine import get_session
from taime_api.db.models import Job
from taime_api.training.cancel import TrainingCancelled
from taime_api.training.device import darts_accelerator, resolve_device
from taime_api.training.port_common import (
    PORT_STRUCTURAL_DEFAULTS,
    resolve_port_hparams,
    structural_overrides_from_config,
)
from taime_api.utils.port_dataset import PortDatasetError, load_meta, resolve_port_dataset_root

if TYPE_CHECKING:
    from darts import TimeSeries
    from darts.models import TSMixerModel

DATA_DIR = Path(os.getenv("DATA_DIR", "data")).resolve()
MODELS_DIR = DATA_DIR / "models"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PortTrainingConfig:
    """Configuration for port TSMixer evaluation."""

    forecast_horizon: int = 48
    stride: int = 1
    last_points_only: bool = False
    max_series: int | None = None
    backtest_max_series: int | None = 20
    run_backtest: bool = True
    accelerator: str = "cpu"
    metrics_mode: str = "classification"
    classification_threshold: float = 0.5


@dataclass(frozen=True)
class PortModelPaths:
    """Resolved model paths for TSMixer."""

    model_path: Path
    ckpt_path: Path | None


@dataclass(frozen=True)
class PortTrainingResult:
    """Training results for port evaluation."""

    model_path: Path
    ckpt_path: Path | None
    metrics: dict[str, Any]


def parse_port_config(config: dict[str, Any] | None) -> PortTrainingConfig:
    """Normalize job config into a PortTrainingConfig."""
    config = config or {}

    def _to_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _to_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).lower() in {"1", "true", "yes", "y"}

    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _to_cap(value: Any, default: int | None) -> int | None:
        # Series-count caps: None or <= 0 -> no cap (all series); invalid -> default.
        # services.jobs stores None for "all", so an explicit None must stay None.
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return None if parsed <= 0 else parsed

    metrics_mode = str(config.get("metrics_mode") or "classification").lower()
    threshold = _to_float(config.get("classification_threshold"))
    threshold = 0.5 if threshold is None else min(max(threshold, 0.0), 1.0)

    return PortTrainingConfig(
        forecast_horizon=max(1, _to_int(config.get("forecast_horizon"), 48)),
        stride=max(1, _to_int(config.get("stride"), 1)),
        last_points_only=_to_bool(config.get("last_points_only"), False),
        max_series=_to_cap(config.get("max_series"), None),
        backtest_max_series=_to_cap(config.get("backtest_max_series", 20), 20),
        run_backtest=_to_bool(config.get("backtest"), True),
        accelerator=str(config.get("accelerator") or "cpu"),
        metrics_mode=metrics_mode,
        classification_threshold=threshold,
    )


def resolve_model_paths(dataset_dir: Path, config: dict[str, Any] | None) -> PortModelPaths:
    """Resolve model paths from config or dataset directory."""
    dataset_dir = dataset_dir.resolve()
    config = config or {}

    def _resolve(path_value: str | None) -> Path | None:
        if not path_value:
            return None
        path = Path(path_value)
        if not path.is_absolute():
            path = dataset_dir / path
        return path.resolve()

    model_path = _resolve(config.get("model_path"))
    if model_path is None:
        candidates = [p for p in dataset_dir.rglob("*.pt") if not p.name.endswith(".ckpt")]
        if len(candidates) != 1:
            raise PortDatasetError(
                f"Expected a single .pt model file in the dataset package; found {len(candidates)}"
            )
        model_path = candidates[0].resolve()

    ckpt_path = _resolve(config.get("ckpt_path"))
    if ckpt_path is None:
        candidate = Path(str(model_path) + ".ckpt")
        if candidate.exists():
            ckpt_path = candidate.resolve()
        else:
            ckpt_candidates = list(dataset_dir.rglob("*.ckpt"))
            if len(ckpt_candidates) == 1:
                ckpt_path = ckpt_candidates[0].resolve()

    allowed_roots = {dataset_dir, MODELS_DIR.resolve()}
    for path in [model_path, ckpt_path]:
        if path is None:
            continue
        if not any(path.is_relative_to(root) for root in allowed_roots):
            raise PortDatasetError(
                "Model paths must be inside the dataset or data/models directory"
            )
        if not path.exists():
            raise PortDatasetError(f"Model file not found: {path}")

    return PortModelPaths(model_path=model_path, ckpt_path=ckpt_path)


def load_port_timeseries(
    dataset_dir: Path,
    max_series: int | None = None,
) -> tuple[list[TimeSeries], list[TimeSeries]]:
    """Load series and past covariates lists from port dataset."""
    root = resolve_port_dataset_root(dataset_dir)
    series = _load_series_list(root, "series", max_series=max_series)
    past_covariates = _load_series_list(root, "past_covs", max_series=max_series)
    if len(series) != len(past_covariates):
        raise PortDatasetError("Series and past covariates list lengths differ")
    series = [_to_float32_fast(ts) for ts in series]
    past_covariates = [_to_float32_fast(ts) for ts in past_covariates]
    return series, past_covariates


def load_tsmixer_model(model_path: Path, accelerator: str = "cpu") -> TSMixerModel:
    """Load the TSMixer model with safe defaults."""
    from darts.models import TSMixerModel

    accelerator = accelerator or "cpu"
    kwargs = {"pl_trainer_kwargs": {"accelerator": accelerator}}
    if accelerator == "cpu":
        kwargs["map_location"] = "cpu"
    return TSMixerModel.load(path=str(model_path), **kwargs)


def run_backtest(
    model: TSMixerModel,
    series: list[TimeSeries],
    past_covariates: list[TimeSeries],
    config: PortTrainingConfig,
) -> dict[str, Any]:
    """Run historical forecasts and backtest metrics."""
    forecasts = model.historical_forecasts(
        series=series,
        past_covariates=past_covariates,
        forecast_horizon=config.forecast_horizon,
        stride=config.stride,
        retrain=False,
        last_points_only=config.last_points_only,
        verbose=False,
    )

    metrics = _compute_classification_metrics(
        series,
        forecasts,
        threshold=config.classification_threshold,
    )
    metrics["classification_threshold"] = config.classification_threshold
    return metrics


def _compute_classification_metrics(
    series: list[TimeSeries] | TimeSeries,
    forecasts: list[TimeSeries] | TimeSeries,
    threshold: float,
) -> dict[str, float]:
    pairs = _pair_series(series, forecasts)
    tp = fp = tn = fn = 0
    total = 0
    reduced_forecasts = False

    for actual_ts, forecast_ts in pairs:
        forecast_ts, reduced = _normalize_forecast_series(forecast_ts)
        reduced_forecasts = reduced_forecasts or reduced
        actual_aligned, forecast_aligned = _align_series(actual_ts, forecast_ts)
        actual_vals = _flatten_values(actual_aligned)
        forecast_vals = _flatten_values(forecast_aligned)
        if actual_vals.size == 0 or forecast_vals.size == 0:
            continue
        size = min(actual_vals.size, forecast_vals.size)
        actual_vals = actual_vals[:size]
        forecast_vals = forecast_vals[:size]
        probs = _sigmoid(forecast_vals)
        preds = probs >= threshold
        actual = actual_vals >= 0.5

        tp += int(np.sum(preds & actual))
        tn += int(np.sum(~preds & ~actual))
        fp += int(np.sum(preds & ~actual))
        fn += int(np.sum(~preds & actual))
        total += size

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_score),
        "support": float(total),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }
    if reduced_forecasts:
        metrics["metrics_note"] = (
            "Historical forecasts returned multiple windows; evaluation used last-point "
            "summaries per window to avoid overlapping timestamps."
        )
    return metrics


def _pair_series(
    series: list[TimeSeries] | TimeSeries,
    forecasts: list[TimeSeries] | TimeSeries,
) -> list[tuple[TimeSeries, TimeSeries | list[TimeSeries]]]:
    if isinstance(series, list):
        if not isinstance(forecasts, list):
            raise PortDatasetError("Forecast output mismatch for port dataset")
        return list(zip(series, forecasts, strict=False))
    if isinstance(forecasts, list):
        raise PortDatasetError("Forecast output mismatch for port dataset")
    return [(series, forecasts)]


def _normalize_forecast_series(
    forecast_ts: TimeSeries | list[TimeSeries],
) -> tuple[TimeSeries, bool]:
    if isinstance(forecast_ts, list):
        return _reduce_forecast_list(forecast_ts), True
    return forecast_ts, False


def _reduce_forecast_list(forecast_list: list[TimeSeries]) -> TimeSeries:
    from darts import TimeSeries

    if not forecast_list:
        raise PortDatasetError("Empty forecast list returned for port dataset")
    times = []
    values = []
    for forecast in forecast_list:
        if forecast.n_timesteps == 0:
            continue
        times.append(forecast.end_time())
        values.append(forecast.values()[-1])
    if not times:
        raise PortDatasetError("Forecast windows contained no points to evaluate")
    time_index = pd.DatetimeIndex(times)
    values_arr = np.stack(values, axis=0)
    return TimeSeries.from_times_and_values(time_index, values_arr)


def _align_series(
    actual_ts: TimeSeries,
    forecast_ts: TimeSeries,
) -> tuple[TimeSeries, TimeSeries]:
    actual_slice = actual_ts.slice_intersect(forecast_ts)
    forecast_slice = forecast_ts.slice_intersect(actual_ts)
    return actual_slice, forecast_slice


def _flatten_values(ts: TimeSeries) -> np.ndarray:
    values = ts.all_values(copy=False)
    values = np.asarray(values)
    values = values.squeeze()
    return values.reshape(-1)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -20.0, 20.0)
    return 1 / (1 + np.exp(-clipped))


def _load_series_list(
    root: Path,
    prefix: str,
    max_series: int | None = None,
) -> list[TimeSeries]:
    meta_path = root / f"{prefix}_meta.json"
    meta = load_meta(meta_path)
    if max_series is not None:
        meta = meta[:max_series]

    series_list: list[TimeSeries] = []
    for entry in meta:
        file_name = entry.get("file")
        if not isinstance(file_name, str):
            raise PortDatasetError(f"Missing file entry in {meta_path.name}")
        df = pd.read_parquet(root / file_name)
        ts = _series_from_dataframe(df)
        sc_df = reconstruct_static_from_json(entry.get("static_covariates"))
        if sc_df is not None:
            ts = ts.with_static_covariates(sc_df)
        series_list.append(ts)

    return series_list


def _series_from_dataframe(df: pd.DataFrame) -> TimeSeries:
    from darts import TimeSeries

    if isinstance(df.index, pd.DatetimeIndex):
        return TimeSeries.from_dataframe(df)
    if "date" in df.columns:
        return TimeSeries.from_dataframe(df, time_col="date")
    return TimeSeries.from_dataframe(df)


def _to_float32_fast(ts: TimeSeries) -> TimeSeries:
    arr = ts.all_values(copy=False)
    arr = arr.astype(np.float32, copy=False)
    return ts.with_values(arr)


def reconstruct_static_from_json(js: Any) -> pd.DataFrame | None:
    """Reconstruct static covariates from a serialized JSON object."""
    if js is None:
        return None

    if isinstance(js, list):
        try:
            return pd.DataFrame(js)
        except Exception:
            return None

    if isinstance(js, dict):
        if all(not isinstance(v, (dict, list)) for v in js.values()):
            return pd.DataFrame([js])

        if all(isinstance(v, dict) for v in js.values()):
            try:
                return pd.DataFrame(js)
            except Exception:
                return None

        try:
            normalized = pd.json_normalize(js)
            if not normalized.empty:
                return normalized
        except Exception:
            return None

    try:
        df = pd.DataFrame([js])
    except Exception:
        return pd.DataFrame([{"static_covariates": str(js)}])

    if df.shape[1] == 1:
        first_val = df.iloc[0, 0]
        if isinstance(first_val, dict):
            try:
                return pd.DataFrame([first_val])
            except Exception:
                return df

    return df


def write_training_report(
    result_path: Path,
    metrics: dict[str, Any],
    config: PortTrainingConfig,
    summary: dict[str, Any],
) -> None:
    """Persist a JSON training summary for traceability."""
    payload = {
        "config": {
            "forecast_horizon": config.forecast_horizon,
            "stride": config.stride,
            "last_points_only": config.last_points_only,
            "max_series": config.max_series,
            "backtest_max_series": config.backtest_max_series,
            "run_backtest": config.run_backtest,
            "accelerator": config.accelerator,
            "metrics_mode": config.metrics_mode,
            "classification_threshold": config.classification_threshold,
        },
        "summary": summary,
        "metrics": metrics,
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_port_evaluation(
    dataset_dir: Path,
    config: dict[str, Any] | None,
    progress: Callable[[float, str], None] | None = None,
) -> PortTrainingResult:
    """Execute the port evaluation pipeline and return metrics."""
    port_config = parse_port_config(config)
    _validate_accelerator(port_config.accelerator)
    job_id = int((config or {}).get("job_id") or 0)
    if job_id:
        with get_session() as session:
            job = session.get(Job, job_id)
            if job and job.status == "cancelled":
                raise TrainingCancelled()
    if progress:
        progress(5.0, "Loading dataset")
    # Evaluation is a backtest only, so it is bounded like the retrain backtest.
    eval_limit = port_config.max_series or port_config.backtest_max_series
    series, past_covariates = load_port_timeseries(dataset_dir, eval_limit)

    if progress:
        progress(35.0, "Resolving model paths")
    model_paths = resolve_model_paths(dataset_dir, config)

    if progress:
        progress(45.0, "Loading model")
    model = load_tsmixer_model(model_paths.model_path, accelerator=port_config.accelerator)

    metrics: dict[str, Any] = {}
    if port_config.run_backtest and port_config.metrics_mode == "classification":
        if progress:
            progress(70.0, "Running backtest")
        metrics.update(run_backtest(model, series, past_covariates, port_config))
    elif port_config.metrics_mode == "pending":
        metrics.update(
            {
                "metrics_status": "pending_target_definition",
                "metrics_note": (
                    "Backtest skipped until target meaning and threshold are confirmed."
                ),
            }
        )
    else:
        metrics.update(
            {
                "metrics_status": "unsupported_metrics_mode",
                "metrics_note": f"Backtest skipped for metrics_mode='{port_config.metrics_mode}'.",
            }
        )

    metrics.update(
        {
            "series_used": len(series),
            "forecast_horizon": port_config.forecast_horizon,
            "stride": port_config.stride,
            "last_points_only": port_config.last_points_only,
        }
    )

    if progress:
        progress(90.0, "Backtest complete")

    return PortTrainingResult(
        model_path=model_paths.model_path,
        ckpt_path=model_paths.ckpt_path,
        metrics=metrics,
    )


def _validate_accelerator(accelerator: str) -> None:
    if accelerator in {"cuda", "gpu"} and not torch.cuda.is_available():
        raise PortDatasetError(
            "CUDA accelerator requested but no CUDA device is available. "
            "Set accelerator='cpu' or install a CUDA-enabled torch build."
        )


def _is_port_job_cancelled(job_id: int) -> bool:
    if not job_id:
        return False
    with get_session() as session:
        job = session.get(Job, job_id)
        return bool(job and job.status == "cancelled")


def extract_structural_params(model: Any) -> dict[str, Any]:
    """Best-effort recovery of TSMixer architecture params from a loaded model.

    ``input_chunk_length`` / ``output_chunk_length`` are public Darts attributes
    (reliable across versions); the TSMixer-specific sizes are read from the
    model's stored creation params when available. Returns only recovered keys.
    """
    recovered: dict[str, Any] = {}
    for attr in ("input_chunk_length", "output_chunk_length"):
        value = getattr(model, attr, None)
        if value:
            recovered[attr] = int(value)
    raw = getattr(model, "model_params", None) or getattr(model, "_model_params", None) or {}
    if isinstance(raw, dict):
        for key in ("hidden_size", "ff_size", "num_blocks", "activation"):
            if raw.get(key) is not None:
                recovered[key] = raw[key]
    return recovered


class _UnresolvedSeedClass:
    """Inert stand-in for a class pickled into a seed model but missing here."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        self.__dict__["_pickled_state"] = state


def read_seed_model(model_path: Path) -> tuple[Any, list[str]]:
    """Unpickle a seed ``TSMixerModel`` wrapper, tolerating classes this process lacks.

    Partner seeds can reference helpers defined in their training script's
    ``__main__`` (e.g. a custom torchmetrics metric), which makes
    ``TSMixerModel.load`` fail with "Can't get attribute ... on <module '__main__'>".
    Retraining only reuses the stored constructor params, so unresolvable classes
    become inert placeholders and are returned for reporting. Like
    ``TSMixerModel.load``, this unpickles the seed, i.e. trusts the uploaded package.
    """
    unresolved: list[str] = []
    placeholders: dict[str, type] = {}

    class _TolerantUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            try:
                return super().find_class(module, name)
            except (AttributeError, ImportError):
                qualified = f"{module}.{name}"
                if qualified not in placeholders:
                    unresolved.append(qualified)
                    placeholders[qualified] = type(name, (_UnresolvedSeedClass,), {})
                return placeholders[qualified]

    pickle_module = types.ModuleType("taime_tolerant_pickle")
    pickle_module.Unpickler = _TolerantUnpickler  # type: ignore[attr-defined]
    pickle_module.load = lambda f, **kw: _TolerantUnpickler(f, **kw).load()  # type: ignore[attr-defined]
    with open(model_path, "rb") as fh:
        model = torch.load(fh, map_location="cpu", weights_only=False, pickle_module=pickle_module)
    return model, unresolved


def seed_loss_info(seed_loss: Any) -> dict[str, Any]:
    """Describe the seed's loss for the job report; the retrain does not reuse it.

    The retrain always trains logits with a plain ``BCEWithLogitsLoss`` (see
    :func:`build_tsmixer_model`). A seed ``pos_weight`` is only reported: Darts
    0.39 cannot reload a ``clean=True`` artifact whose loss carries buffers (the
    rebuilt ``train_criterion``/``val_criterion`` lack them, so the strict
    state_dict load fails), and up-weighting the majority positive class pushes
    predictions further toward 1.
    """
    info: dict[str, Any] = {
        "seed_loss_fn": type(seed_loss).__name__ if seed_loss is not None else None
    }
    pos_weight = getattr(seed_loss, "pos_weight", None)
    if isinstance(pos_weight, torch.Tensor) and pos_weight.numel() == 1:
        info["seed_pos_weight"] = float(pos_weight.item())
    return info


def _resolve_seed_recipe(
    dataset_dir: Path,
    config: dict[str, Any],
    forecast_horizon: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve TSMixer architecture, preferring the seed model, then config, then defaults.

    Returns ``(structural, provenance)``. Which path was taken is logged loudly
    *and* recorded in ``provenance`` (merged into the job metrics), so a fallback
    that changes the architecture is visible in the report, not only in the
    container log.
    """
    structural = dict(PORT_STRUCTURAL_DEFAULTS)
    structural["output_chunk_length"] = forecast_horizon
    provenance: dict[str, Any] = {"architecture_source": "defaults"}

    try:
        paths = resolve_model_paths(dataset_dir, config)
        seed_model, unresolved = read_seed_model(paths.model_path)
        if unresolved:
            logger.warning(
                "Port retrain: seed model references classes unavailable here %s; "
                "ignored (only the seed's constructor params are reused)",
                unresolved,
            )
            provenance["seed_unresolved_classes"] = unresolved
        recovered = extract_structural_params(seed_model)
        if recovered:
            logger.info("Port retrain: recovered architecture from seed model: %s", recovered)
            structural.update(recovered)
            provenance["architecture_source"] = "seed"
        else:
            logger.warning(
                "Port retrain: could not read architecture from seed model; using defaults %s",
                structural,
            )
        seed_params = getattr(seed_model, "model_params", None)
        if isinstance(seed_params, dict):
            provenance.update(seed_loss_info(seed_params.get("loss_fn")))
    except Exception as exc:  # noqa: BLE001 - seed model is optional for from-scratch retrain
        logger.warning(
            "Port retrain: no usable seed model (%s); using default architecture %s",
            exc,
            structural,
        )

    overrides = structural_overrides_from_config(config)
    if overrides:
        logger.info("Port retrain: applying explicit structural overrides: %s", overrides)
        structural.update(overrides)
    return structural, provenance


def _split_backtest_holdout(
    pairs: list[tuple[TimeSeries, TimeSeries]],
    config: PortTrainingConfig,
    backtest: bool,
) -> tuple[list[tuple[TimeSeries, TimeSeries]], list[tuple[TimeSeries, TimeSeries]]]:
    """Hold out a few whole series (port calls) for the post-training backtest.

    Backtesting the training series only yields in-sample metrics, and a
    stride-1 backtest over every series of a full-size package is prohibitively
    slow. So a seeded sample of at most ``backtest_max_series`` series (and at
    most ~10%) is excluded from the fit and backtested instead. With fewer than
    two series nothing is held out and the backtest stays in-sample.
    """
    if not backtest or len(pairs) < 2:
        return pairs, []
    cap = config.backtest_max_series or len(pairs)
    n_holdout = max(1, min(cap, len(pairs) // 10))
    rng = np.random.default_rng(42)
    held = {int(i) for i in rng.choice(len(pairs), size=n_holdout, replace=False)}
    train = [pair for i, pair in enumerate(pairs) if i not in held]
    holdout = [pair for i, pair in enumerate(pairs) if i in held]
    return train, holdout


def _make_port_progress_callback(
    job_id: int,
    n_epochs: int,
    low: float = 55.0,
    high: float = 85.0,
) -> Any | None:
    """Build a Lightning callback that reports per-epoch progress and honors cancel.

    The TSMixer ``model.fit`` is the dominant cost of a port retrain, so without
    an in-loop hook the polled job sits at ``low`` % for the whole fit. This
    callback advances progress across ``[low, high]`` per epoch (and exposes the
    real epoch number) and stops training when the job is cancelled. It is fully
    defensive: it returns ``None`` if Lightning is unavailable, and every hook
    body swallows its own errors so a progress update can never break training.
    """
    try:
        from pytorch_lightning.callbacks import Callback
    except Exception:  # pragma: no cover - lightning is installed at runtime
        logger.debug("pytorch_lightning callback unavailable; skipping port progress hook")
        return None

    total = max(1, int(n_epochs))
    span = max(0.0, high - low)

    class _PortProgressCallback(Callback):  # type: ignore[misc, valid-type]
        def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
            try:
                epoch = int(getattr(trainer, "current_epoch", 0)) + 1
                value = low + min(epoch / total, 1.0) * span
                with get_session() as session:
                    job = session.get(Job, job_id)
                    if job is None:
                        return
                    job.progress = min(max(value, 0.0), 100.0)
                    job.current_epoch = min(epoch, total)
                    job.total_epochs = total
                    job.metrics = {"status": f"Training epoch {epoch}/{total}"}
                    session.add(job)
                    session.commit()
                    cancelled = job.status == "cancelled"
                if cancelled:
                    trainer.should_stop = True
            except Exception:  # never let a progress update break training
                logger.debug("Port progress callback failed", exc_info=True)

    return _PortProgressCallback()


def build_tsmixer_model(
    structural: dict[str, Any],
    hparams: dict[str, Any],
    accelerator: str,
    extra_callbacks: list[Any] | None = None,
) -> TSMixerModel:
    """Construct a fresh TSMixerModel from structural + tunable hyperparameters.

    The port target is a binary state per time step and the model output is read
    as a logit (TAIME's backtest, and the partner's assessment, threshold
    ``sigmoid(output) >= 0.5``), so the model trains with ``BCEWithLogitsLoss``
    like the partner's seed. Darts' default ``MSELoss`` trains a regressor whose
    outputs sit around [0, 1], where ``sigmoid >= 0.5`` everywhere: only 1s.
    """
    from darts.models import TSMixerModel

    pl_trainer_kwargs: dict[str, Any] = {
        "accelerator": accelerator,
        "devices": 1,
        "enable_progress_bar": False,
    }
    if extra_callbacks:
        pl_trainer_kwargs["callbacks"] = list(extra_callbacks)

    return TSMixerModel(
        input_chunk_length=int(structural["input_chunk_length"]),
        output_chunk_length=int(structural["output_chunk_length"]),
        hidden_size=int(structural["hidden_size"]),
        ff_size=int(structural["ff_size"]),
        num_blocks=int(structural["num_blocks"]),
        activation=str(structural["activation"]),
        dropout=float(hparams["dropout"]),
        norm_type=str(hparams["norm_type"]),
        normalize_before=bool(hparams["normalize_before"]),
        use_reversible_instance_norm=bool(hparams["use_reversible_instance_norm"]),
        # No pos_weight: loss buffers break Darts' reload of the clean artifact.
        loss_fn=torch.nn.BCEWithLogitsLoss(),
        batch_size=int(hparams["batch_size"]),
        n_epochs=int(hparams["n_epochs"]),
        optimizer_kwargs={"lr": float(hparams["learning_rate"])},
        lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        lr_scheduler_kwargs={
            # Darts pops "monitor" into the Lightning scheduler config; "train_loss"
            # is always logged (we fit without a separate validation series).
            "monitor": "train_loss",
            "factor": float(hparams["lr_scheduler_factor"]),
            "patience": int(hparams["lr_scheduler_patience"]),
        },
        pl_trainer_kwargs=pl_trainer_kwargs,
    )


def retrain_port_model(
    dataset_dir: Path,
    config: dict[str, Any] | None,
    models_dir: Path,
    job_id: int,
    progress: Callable[[float, str], None] | None = None,
) -> PortTrainingResult:
    """Retrain the port TSMixer model from scratch on the uploaded dataset.

    Honors the partner-approved tunable hyperparameters (batch size, dropout,
    learning rate, LR scheduler factor/patience, reversible-instance-norm,
    normalize_before, norm_type) while preserving the original architecture
    sizes when they can be recovered from the seed model. Trains on every series
    in the package unless ``max_series`` caps it.
    """
    config = config or {}
    port_config = parse_port_config(config)
    device = resolve_device(config)
    accelerator = darts_accelerator(device)
    hparams = resolve_port_hparams(config)
    backtest = port_config.run_backtest and port_config.metrics_mode == "classification"

    if _is_port_job_cancelled(job_id):
        raise TrainingCancelled()
    if progress:
        progress(5.0, "Loading dataset")
    series, past_covariates = load_port_timeseries(dataset_dir, port_config.max_series)

    if progress:
        progress(25.0, "Resolving architecture")
    structural, provenance = _resolve_seed_recipe(dataset_dir, config, port_config.forecast_horizon)

    # Darts aborts the whole fit on a series shorter than one training window, so
    # skip those port calls (and report how many) instead of failing the job.
    min_length = int(structural["input_chunk_length"]) + int(structural["output_chunk_length"])
    pairs = [(s, c) for s, c in zip(series, past_covariates, strict=True) if len(s) >= min_length]
    skipped = len(series) - len(pairs)
    if skipped:
        logger.warning(
            "Port retrain: skipped %d of %d series shorter than %d time steps",
            skipped,
            len(series),
            min_length,
        )
    if not pairs:
        raise PortDatasetError(
            f"No series is long enough to train on (need at least {min_length} time steps)"
        )
    train_pairs, holdout_pairs = _split_backtest_holdout(pairs, port_config, backtest)
    train_series = [s for s, _ in train_pairs]
    train_covariates = [c for _, c in train_pairs]
    eval_pairs = holdout_pairs or train_pairs[: port_config.backtest_max_series]

    if _is_port_job_cancelled(job_id):
        raise TrainingCancelled()
    if progress:
        progress(40.0, "Building model")
    # Per-epoch progress + cancellation during the long fit (defensive: None if
    # Lightning is unavailable, in which case fit proceeds exactly as before).
    progress_cb = _make_port_progress_callback(job_id, int(hparams["n_epochs"]))
    model = build_tsmixer_model(
        structural,
        hparams,
        accelerator,
        extra_callbacks=[progress_cb] if progress_cb is not None else None,
    )

    if progress:
        progress(55.0, "Training")
    model.fit(series=train_series, past_covariates=train_covariates)

    if _is_port_job_cancelled(job_id):
        raise TrainingCancelled()
    if progress:
        progress(85.0, "Saving model")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"port_model_{job_id}.pt"
    model.save(str(model_path), clean=True)

    metrics: dict[str, Any] = {}
    if backtest:
        if progress:
            progress(92.0, "Running backtest")
        try:
            # The metrics only score each window's last point, so ask Darts for just
            # those: same numbers, far less memory, and no frequency inference on
            # held-out series that yield only one or two windows.
            metrics.update(
                run_backtest(
                    model,
                    [s for s, _ in eval_pairs],
                    [c for _, c in eval_pairs],
                    dataclasses.replace(port_config, last_points_only=True),
                )
            )
        except Exception as exc:  # noqa: BLE001 - backtest is best-effort, never fail the retrain
            logger.warning("Port retrain backtest failed: %s", exc)
            metrics["backtest_error"] = str(exc)
        metrics["backtest_series"] = len(eval_pairs)
        metrics["metrics_scope"] = "held_out_series" if holdout_pairs else "in_sample"

    metrics.update(
        {
            "series_used": len(train_series),
            "series_skipped_too_short": skipped,
            "forecast_horizon": port_config.forecast_horizon,
            **provenance,
            "hyperparameters": {
                **hparams,
                "device": device,
                **structural,
                "loss_fn": "BCEWithLogitsLoss",
            },
        }
    )

    if progress:
        progress(100.0, "Complete")

    return PortTrainingResult(model_path=model_path, ckpt_path=None, metrics=metrics)
