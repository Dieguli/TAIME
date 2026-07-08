"""API-level tests for job creation validation (training is mocked out)."""

import pytest

from taime_api.db.engine import get_session
from taime_api.db.models import Dataset


def _make_dataset(sut_type: str) -> int:
    with get_session() as session:
        dataset = Dataset(
            name="test-dataset",
            filename="test.zip",
            file_path="/tmp/test.zip",
            file_size=1,
            sut_type=sut_type,
        )
        session.add(dataset)
        session.commit()
        session.refresh(dataset)
        return dataset.id


@pytest.fixture(autouse=True)
def _no_real_training(monkeypatch):
    """Replace the worker entrypoint so a valid job never starts heavy training."""
    import taime_api.services.jobs as jobs_module

    monkeypatch.setattr(jobs_module, "run_training_job", lambda job_id: {"success": True})


@pytest.mark.anyio
async def test_invalid_port_batch_size_returns_422(client):
    dataset_id = _make_dataset("infra_port")
    resp = await client.post(
        "/api/v1/jobs",
        json={"dataset_id": dataset_id, "sut_type": "infra_port", "config": {"batch_size": 32}},
    )
    assert resp.status_code == 422
    assert "batch_size" in resp.text


@pytest.mark.anyio
async def test_invalid_transformer_lr_returns_422(client):
    dataset_id = _make_dataset("disinfo_fake")
    resp = await client.post(
        "/api/v1/jobs",
        json={
            "dataset_id": dataset_id,
            "sut_type": "disinfo_fake",
            "config": {"learning_rate": -1},
        },
    )
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_valid_port_job_created(client):
    dataset_id = _make_dataset("infra_port")
    resp = await client.post(
        "/api/v1/jobs",
        json={
            "dataset_id": dataset_id,
            "sut_type": "infra_port",
            "config": {
                "batch_size": 128,
                "dropout": 0.3,
                "norm_type": "TimeBatchNorm2d",
                "learning_rate": 1e-3,
                "device": "cpu",
            },
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["sut_type"] == "infra_port"
    assert body["config"]["batch_size"] == 128
    assert body["config"]["norm_type"] == "TimeBatchNorm2d"


@pytest.mark.anyio
async def test_valid_transformer_job_created(client):
    dataset_id = _make_dataset("disinfo_fake")
    resp = await client.post(
        "/api/v1/jobs",
        json={
            "dataset_id": dataset_id,
            "sut_type": "disinfo_fake",
            "config": {
                "epochs": 3,
                "batch_size": 16,
                "learning_rate": 5e-6,
                "weight_decay": 0.1,
                "seed": 42,
                "device": "auto",
            },
        },
    )
    assert resp.status_code == 201
    assert resp.json()["config"]["weight_decay"] == 0.1
