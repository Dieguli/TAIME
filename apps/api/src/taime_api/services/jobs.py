"""Jobs service - business logic for training job management."""

from datetime import datetime
from typing import Any

from fastapi import HTTPException
from sqlmodel import select

from taime_api.api.v1.schemas import SUTType
from taime_api.db.engine import get_session
from taime_api.db.models import Dataset, Job
from taime_api.training.config_validation import ConfigValidationError, validate_job_config
from taime_api.training.port_common import resolve_port_hparams
from taime_api.workers.pool import submit_task
from taime_api.workers.runner import run_training_job


def _device_to_accelerator(device: Any) -> str:
    """Map a UI device choice to a PyTorch-Lightning accelerator label.

    Used only to record the port training report's ``accelerator`` field so it
    reflects the user's selection instead of being hard-coded to ``cpu``. The
    actual compute device is resolved separately by ``resolve_device`` at train
    time, and the truly-resolved device is recorded in metrics.hyperparameters.
    """
    value = str(device or "auto").strip().lower()
    if value in {"cuda", "gpu"}:
        return "gpu"
    if value == "cpu":
        return "cpu"
    return "auto"


def reconcile_interrupted_jobs() -> None:
    """Fail jobs left active by a previous process (crash, redeploy, restart).

    The worker pool and its in-memory results do not survive a process restart,
    so any job still marked ``pending``/``running`` at startup has no worker
    behind it. Mark those as ``failed`` so they do not linger as ghost rows that
    appear to be running forever.
    """
    with get_session() as session:
        stale = session.exec(select(Job).where(Job.status.in_(("pending", "running")))).all()
        for job in stale:
            job.status = "failed"
            job.error_message = "Interrupted by a server restart before completion."
            job.completed_at = datetime.utcnow()
            session.add(job)
        if stale:
            session.commit()


def _normalize_port_config(config: dict[str, Any]) -> dict[str, Any]:
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

    normalized = dict(config)
    normalized["forecast_horizon"] = max(1, _to_int(config.get("forecast_horizon"), 48))
    normalized["stride"] = max(1, _to_int(config.get("stride"), 1))
    normalized["last_points_only"] = _to_bool(config.get("last_points_only"), False)
    # Retrain on every series unless a positive cap is given (<= 0 also means "all").
    max_series_value = _to_int(config.get("max_series"), 0)
    normalized["max_series"] = None if max_series_value <= 0 else max_series_value
    # The post-training backtest runs on a few held-out series (<= 0 means "all").
    backtest_max_series = _to_int(config.get("backtest_max_series"), 20)
    normalized["backtest_max_series"] = None if backtest_max_series <= 0 else backtest_max_series
    normalized["backtest"] = _to_bool(config.get("backtest"), True)
    normalized["accelerator"] = config.get("accelerator") or _device_to_accelerator(
        config.get("device")
    )
    normalized["metrics_mode"] = str(config.get("metrics_mode") or "classification").lower()
    threshold = _to_float(config.get("classification_threshold"))
    normalized["classification_threshold"] = 0.5 if threshold is None else threshold
    return normalized


class JobService:
    """Service for training job operations."""

    async def create_job(
        self,
        dataset_id: int,
        sut_type: SUTType,
        config: dict[str, Any] | None = None,
    ) -> Job:
        """Create a new training job."""
        with get_session() as session:
            # Verify dataset exists
            dataset = session.get(Dataset, dataset_id)
            if not dataset:
                raise HTTPException(status_code=404, detail="Dataset not found")
            if dataset.sut_type != sut_type.value:
                raise HTTPException(status_code=400, detail="Dataset SUT type mismatch")

            # Create job record
            config = config or {}
            try:
                validate_job_config(sut_type.value, config)
            except ConfigValidationError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            if sut_type == SUTType.INFRA_PORT:
                config = _normalize_port_config(config)
                # TSMixer trains resolve_port_hparams()["n_epochs"] epochs (UI
                # default 10), so report the real count rather than 1 — the UI
                # shows current_epoch/total_epochs while polling.
                total_epochs = resolve_port_hparams(config)["n_epochs"]
            else:
                total_epochs = config.get("epochs", 10)

            job = Job(
                dataset_id=dataset_id,
                sut_type=sut_type.value,
                status="pending",
                config=config,
                total_epochs=total_epochs,
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            return job

    async def execute_job(self, job_id: int) -> None:
        """Execute a training job in the worker pool."""
        try:
            # Submit to worker pool
            submit_task(run_training_job, (job_id,))
        except RuntimeError:
            # Pool not available, run synchronously (for development)
            run_training_job(job_id)

    async def cancel_job(self, job_id: int) -> Job:
        """Cancel a running job."""
        with get_session() as session:
            job = session.get(Job, job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")

            if job.status not in ("pending", "running"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot cancel job with status: {job.status}",
                )

            job.status = "cancelled"
            job.completed_at = datetime.utcnow()
            session.add(job)
            session.commit()
            session.refresh(job)
            return job
