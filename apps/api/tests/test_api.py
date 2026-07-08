"""Tests for API endpoints."""

import pytest


@pytest.mark.anyio
async def test_health_check(client):
    """Test health check endpoint."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "2.0.0"


@pytest.mark.anyio
async def test_list_datasets_empty(client):
    """Test listing datasets when empty."""
    response = await client.get("/api/v1/datasets")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.anyio
async def test_list_jobs_empty(client):
    """Test listing jobs when empty."""
    response = await client.get("/api/v1/jobs")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.anyio
async def test_list_models_empty(client):
    """Test listing models when empty."""
    response = await client.get("/api/v1/models")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.anyio
async def test_runtime_capabilities(client):
    """Test runtime capability endpoint."""
    response = await client.get("/api/v1/runtime")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "2.0.0"
    assert isinstance(data["cuda_available"], bool)
    assert "image" in data
    assert "device_name" in data
    assert "torch_version" in data
