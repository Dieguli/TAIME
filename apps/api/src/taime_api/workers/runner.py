"""Training job runner - executes training jobs in worker processes."""

from __future__ import annotations

import logging
import os
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlmodel import func, select

from taime_api.api.v1.schemas import SUTType
from taime_api.db.engine import get_session
from taime_api.db.models import Dataset, Job, ModelVersion
from taime_api.training.cancel import TrainingCancelled
from taime_api.utils.port_dataset import PortDatasetError

if TYPE_CHECKING:
    from taime_api.training.port_tsmixer import PortTrainingResult

logger = logging.getLogger(__name__)

# Storage paths
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
MODELS_DIR = DATA_DIR / "models"


def run_training_job(job_id: int) -> dict[str, Any]:
    """
    Execute a training job.

    This function runs in a separate process and performs the actual
    model training based on the job configuration.

    Args:
        job_id: The job ID to execute

    Returns:
        dict with training results (metrics, model path, etc.)
    """
    try:
        with get_session() as session:
            job = session.get(Job, job_id)
            if not job:
                raise ValueError(f"Job {job_id} not found")
            if job.status == "cancelled":
                job.completed_at = datetime.utcnow()
                session.add(job)
                session.commit()
                return {"success": False, "cancelled": True}

            dataset = session.get(Dataset, job.dataset_id)
            if not dataset:
                raise ValueError(f"Dataset {job.dataset_id} not found")

            # Update status to running
            job.status = "running"
            job.started_at = datetime.utcnow()
            session.add(job)
            session.commit()

        config = job.config or {}

        if job.sut_type == SUTType.INFRA_PORT.value:
            logger.info("Starting port retraining job %s", job_id)
            result = _run_port_job(job_id, dataset, config)
            return _persist_model(job_id, result.model_path, result.metrics, config)

        if job.sut_type == SUTType.HEALTHCARE_PC.value:
            from taime_api.training.healthcare_mup import train_healthcare_model

            model_path, metrics = train_healthcare_model(
                Path(dataset.file_path),
                config,
                MODELS_DIR,
                job_id,
                progress_callback=lambda value: _update_progress(job_id, value),
            )
            return _persist_model(job_id, model_path, metrics, config)

        if job.sut_type == SUTType.DISINFO_FAKE.value:
            from taime_api.training.fake_news_transformer import train_fake_news_model

            model_path, metrics = train_fake_news_model(
                Path(dataset.file_path),
                config,
                MODELS_DIR,
                job_id,
                progress_callback=lambda value: _update_progress(job_id, value),
            )
            return _persist_model(job_id, model_path, metrics, config)

        if job.sut_type == SUTType.DISINFO_HATE.value:
            from taime_api.training.hate_speech_transformer import train_hate_speech_model

            model_path, metrics = train_hate_speech_model(
                Path(dataset.file_path),
                config,
                MODELS_DIR,
                job_id,
                progress_callback=lambda value: _update_progress(job_id, value),
            )
            return _persist_model(job_id, model_path, metrics, config)

        # Get training config
        epochs = config.get("epochs", 10)
        learning_rate = config.get("learning_rate", 0.001)
        batch_size = config.get("batch_size", 32)

        logger.info(
            f"Starting training job {job_id} - "
            f"epochs={epochs}, lr={learning_rate}, batch={batch_size}"
        )

        # Simulate training progress (to be replaced with actual training)
        metrics: dict[str, Any] = {}
        for epoch in range(1, epochs + 1):
            with get_session() as session:
                current = session.get(Job, job_id)
                if current and current.status == "cancelled":
                    raise TrainingCancelled()
            # Update progress
            with get_session() as session:
                job = session.get(Job, job_id)
                if job:
                    job.current_epoch = epoch
                    job.total_epochs = epochs
                    job.progress = (epoch / epochs) * 100
                    session.add(job)
                    session.commit()

            # Simulate training time
            time.sleep(0.5)

            # Simulate metrics
            metrics = {
                "epoch": epoch,
                "loss": 1.0 / epoch,
                "accuracy": min(0.5 + (epoch * 0.05), 0.99),
            }

            logger.info(f"Job {job_id} - Epoch {epoch}/{epochs}: {metrics}")

        # Create model artifact
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"model_{job_id}.pkl"
        model_path.write_text("# Placeholder model file")

        return _persist_model(job_id, model_path, metrics, config)

    except TrainingCancelled:
        with get_session() as session:
            job = session.get(Job, job_id)
            if job:
                job.status = "cancelled"
                job.completed_at = datetime.utcnow()
                session.add(job)
                session.commit()
        return {"success": False, "cancelled": True}
    except Exception as e:
        logger.exception(f"Training job {job_id} failed: {e}")

        # Update job as failed
        with get_session() as session:
            job = session.get(Job, job_id)
            if job:
                if job.status == "cancelled":
                    job.completed_at = datetime.utcnow()
                    session.add(job)
                    session.commit()
                    return {"success": False, "cancelled": True}
                job.status = "failed"
                job.error_message = str(e)
                job.completed_at = datetime.utcnow()
                session.add(job)
                session.commit()

        return {"success": False, "error": str(e)}


def _derive_epoch(progress: float, total_epochs: int) -> int:
    """Map a 0-100 progress value onto a 1..total_epochs current-epoch number.

    Used when the trainer reports overall progress but not a precise epoch, so
    the UI's epoch counter still advances with progress instead of sitting at a
    fixed value. Any non-zero progress yields at least epoch 1.
    """
    total = max(1, total_epochs)
    derived = int(round((progress / 100.0) * total))
    if progress > 0 and derived == 0:
        derived = 1
    return min(max(derived, 0), total)


def _update_progress(
    job_id: int,
    progress: float,
    epoch: int | None = None,
    total_epochs: int | None = None,
) -> None:
    with get_session() as session:
        job = session.get(Job, job_id)
        if job:
            safe_progress = min(max(progress, 0.0), 100.0)
            job.progress = safe_progress

            total_epochs_value = total_epochs or job.total_epochs or 1
            if epoch is None:
                epoch_value = _derive_epoch(safe_progress, total_epochs_value)
            else:
                epoch_value = epoch

            job.current_epoch = epoch_value
            if total_epochs is not None:
                job.total_epochs = total_epochs_value
            session.add(job)
            session.commit()


def _package_model_artifact(model_path: Path) -> tuple[Path, int]:
    """Package a trained model into a single downloadable ``.zip`` artifact.

    The four SUT trainers save their models in different shapes:

    * transformers (fake news, hate speech) save a *directory* via
      ``save_pretrained`` (config + weights + tokenizer),
    * Darts (port) saves a ``.pt`` file plus a sibling ``.pt.ckpt`` checkpoint,
    * XGBoost (healthcare) saves a single ``.pkl`` file.

    ``download_model`` serves a :class:`FileResponse`, which can only stream a
    regular file — a directory path raises ``RuntimeError`` at request time. So
    we always emit one ``.zip`` containing the complete artifact (including the
    Darts ``.ckpt`` sidecar) and return its path and size for the catalog entry.
    """
    model_path = Path(model_path)
    zip_path = model_path.parent / (model_path.name + ".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if model_path.is_dir():
            for item in sorted(model_path.rglob("*")):
                if item.is_file():
                    arcname = (Path(model_path.name) / item.relative_to(model_path)).as_posix()
                    archive.write(item, arcname=arcname)
        else:
            archive.write(model_path, arcname=model_path.name)
            # Include sibling sidecars sharing the model's name (e.g. Darts
            # writes "<name>.pt.ckpt" next to "<name>.pt"), but never the zip
            # itself or the model file we already added.
            for sibling in sorted(model_path.parent.glob(model_path.name + ".*")):
                if sibling.is_file() and sibling not in (zip_path, model_path):
                    archive.write(sibling, arcname=sibling.name)
    return zip_path, zip_path.stat().st_size


def _persist_model(
    job_id: int,
    model_path: Path,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    artifact_path, artifact_size = _package_model_artifact(model_path)
    with get_session() as session:
        job = session.get(Job, job_id)
        if not job:
            raise ValueError("Job not found")

        if job.status == "cancelled":
            # A cancel landed after the trainer's final cancellation check;
            # honor it instead of resurrecting the job as completed, and do not
            # register a model version for a job the user asked to stop.
            job.completed_at = job.completed_at or datetime.utcnow()
            session.add(job)
            session.commit()
            return {"success": False, "cancelled": True}

        max_version = session.exec(
            select(func.coalesce(func.max(ModelVersion.version), 0)).where(
                ModelVersion.sut_type == job.sut_type
            )
        ).one()
        version = int(max_version) + 1

        model = ModelVersion(
            name=f"{job.sut_type}_v{version}",
            job_id=job_id,
            sut_type=job.sut_type,
            version=version,
            file_path=str(artifact_path),
            file_size=artifact_size,
            metrics=metrics,
            config=config,
        )
        session.add(model)

        job.status = "completed"
        job.progress = 100.0
        job.metrics = metrics
        job.completed_at = datetime.utcnow()
        session.add(job)
        session.commit()

    return {"success": True, "model_path": str(model_path), "metrics": metrics}


def _run_port_job(job_id: int, dataset: Dataset, config: dict[str, Any]) -> PortTrainingResult:
    from taime_api.training.port_tsmixer import (
        PortTrainingResult,
        parse_port_config,
        retrain_port_model,
        run_port_evaluation,
        write_training_report,
    )

    dataset_path = Path(dataset.file_path)
    if not dataset_path.exists():
        raise PortDatasetError("Port dataset path does not exist")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = MODELS_DIR / f"infra_port_{job_id}"
    model_dir.mkdir(parents=True, exist_ok=True)

    def _progress(value: float, message: str) -> None:
        # Milestone progress for the phases around model.fit (load/build/save).
        # current_epoch is owned by the per-epoch Lightning callback during the
        # fit, so we deliberately leave it untouched here — otherwise a
        # progress-derived epoch would fight the callback's real epoch and flicker
        # non-monotonically at the fit boundaries.
        with get_session() as session:
            job = session.get(Job, job_id)
            if job:
                job.progress = min(max(value, 0.0), 100.0)
                job.metrics = {"status": message}
                session.add(job)
                session.commit()

    config_with_job = dict(config)
    config_with_job["job_id"] = job_id

    mode = str(config.get("mode") or "retrain").lower()
    if mode == "evaluate":
        # Backwards-compatible path: load the bundled model and backtest only.
        result = run_port_evaluation(dataset_path, config_with_job, progress=_progress)
        model_path = Path(result.model_path)
        target_model_path = model_dir / model_path.name
        target_model_path.write_bytes(model_path.read_bytes())
        if result.ckpt_path is not None:
            target_ckpt_path = model_dir / result.ckpt_path.name
            target_ckpt_path.write_bytes(result.ckpt_path.read_bytes())
        final_result = PortTrainingResult(
            model_path=target_model_path, ckpt_path=None, metrics=result.metrics
        )
    else:
        # Default: from-scratch retraining with the partner-approved hyperparameters.
        final_result = retrain_port_model(
            dataset_path, config_with_job, model_dir, job_id, progress=_progress
        )

    report_path = model_dir / "training_report.json"
    write_training_report(
        report_path,
        metrics=final_result.metrics,
        config=parse_port_config(config),
        summary={"dataset_id": dataset.id, "dataset_name": dataset.name},
    )

    return final_result
