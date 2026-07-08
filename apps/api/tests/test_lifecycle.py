"""Job-lifecycle hardening tests (no ML stack required).

Covers the restart reconciliation pass, the real port epoch count, the
device->accelerator report label, and the late-cancel (TOCTOU) guard in
``_persist_model``. None of these touch the heavy trainers.
"""

from pathlib import Path

import pytest
from sqlmodel import select

from taime_api.api.v1.schemas import SUTType
from taime_api.db.engine import get_session
from taime_api.db.models import Dataset, Job, ModelVersion
from taime_api.services.jobs import JobService, reconcile_interrupted_jobs
from taime_api.workers.runner import _persist_model


def _make_dataset(sut_type: str) -> int:
    with get_session() as session:
        ds = Dataset(
            name="ds", filename="d.zip", file_path="/tmp/d.zip", file_size=1, sut_type=sut_type
        )
        session.add(ds)
        session.commit()
        session.refresh(ds)
        return ds.id


def _make_job(status: str, sut_type: str = "healthcare_pc") -> int:
    with get_session() as session:
        job = Job(dataset_id=1, sut_type=sut_type, status=status)
        session.add(job)
        session.commit()
        session.refresh(job)
        return job.id


def _status(job_id: int) -> str:
    with get_session() as session:
        return session.get(Job, job_id).status


def test_reconcile_fails_only_active_jobs():
    ids = {s: _make_job(s) for s in ("pending", "running", "completed", "failed", "cancelled")}

    reconcile_interrupted_jobs()

    assert _status(ids["pending"]) == "failed"
    assert _status(ids["running"]) == "failed"
    assert _status(ids["completed"]) == "completed"
    assert _status(ids["failed"]) == "failed"
    assert _status(ids["cancelled"]) == "cancelled"
    with get_session() as session:
        job = session.get(Job, ids["running"])
        assert job.error_message and "restart" in job.error_message.lower()
        assert job.completed_at is not None


@pytest.mark.anyio
async def test_port_job_reports_real_total_epochs():
    ds = _make_dataset("infra_port")
    job = await JobService().create_job(
        ds, SUTType.INFRA_PORT, {"epochs": 7, "batch_size": 64, "device": "cpu"}
    )
    # Port used to hard-code total_epochs=1; it must reflect the resolved n_epochs.
    assert job.total_epochs == 7


@pytest.mark.anyio
@pytest.mark.parametrize(
    "device,expected",
    [("cuda", "gpu"), ("cpu", "cpu"), ("auto", "auto"), (None, "auto")],
)
async def test_port_config_accelerator_reflects_device(device, expected):
    ds = _make_dataset("infra_port")
    config = {"batch_size": 64}
    if device is not None:
        config["device"] = device
    job = await JobService().create_job(ds, SUTType.INFRA_PORT, config)
    assert job.config["accelerator"] == expected


def test_persist_model_skips_cancelled_job(tmp_path):
    job_id = _make_job("cancelled")
    artifact = tmp_path / "m.pkl"
    artifact.write_bytes(b"x")

    result = _persist_model(job_id, artifact, {"accuracy": 1.0}, {"epochs": 1})

    assert result == {"success": False, "cancelled": True}
    assert _status(job_id) == "cancelled"
    with get_session() as session:
        rows = session.exec(select(ModelVersion).where(ModelVersion.job_id == job_id)).all()
    assert rows == []


def test_persist_model_completes_active_job(tmp_path):
    job_id = _make_job("running")
    artifact = tmp_path / "m2.pkl"
    artifact.write_bytes(b"y")

    result = _persist_model(job_id, artifact, {"accuracy": 0.9}, {"epochs": 1})

    assert result["success"] is True
    assert _status(job_id) == "completed"
    with get_session() as session:
        rows = session.exec(select(ModelVersion).where(ModelVersion.job_id == job_id)).all()
    assert len(rows) == 1


def test_port_retrain_saves_clean_darts_artifact(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from taime_api.training.port_tsmixer import retrain_port_model

    class FakePortModel:
        def fit(self, *, series, past_covariates):
            assert series == ["series"]
            assert past_covariates == ["covariates"]

        def save(self, path, clean=False):
            assert clean is True
            Path(path).write_bytes(b"clean-darts-model")

    monkeypatch.setattr(
        "taime_api.training.port_tsmixer.load_port_timeseries",
        lambda dataset_dir, max_series: (["series"], ["covariates"]),
    )
    monkeypatch.setattr(
        "taime_api.training.port_tsmixer._resolve_structural_params",
        lambda dataset_dir, config, forecast_horizon: {
            "input_chunk_length": 24,
            "output_chunk_length": 12,
            "hidden_size": 16,
            "ff_size": 32,
            "num_blocks": 1,
            "activation": "ReLU",
        },
    )
    monkeypatch.setattr(
        "taime_api.training.port_tsmixer.build_tsmixer_model",
        lambda structural, hparams, accelerator, extra_callbacks=None: FakePortModel(),
    )
    monkeypatch.setattr("taime_api.training.port_tsmixer._make_port_progress_callback", lambda *_: None)
    monkeypatch.setattr("taime_api.training.port_tsmixer._is_port_job_cancelled", lambda job_id: False)

    result = retrain_port_model(
        dataset_dir=tmp_path / "dataset",
        config={"epochs": 1, "run_backtest": False},
        models_dir=tmp_path / "models",
        job_id=123,
    )

    assert result.model_path.name == "port_model_123.pt"
    assert result.model_path.read_bytes() == b"clean-darts-model"
