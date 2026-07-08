"""Test configuration and fixtures."""

import os
import tempfile
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport

TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="taime-tests-")).resolve()
os.environ.setdefault("DATA_DIR", str(TEST_DATA_DIR))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{TEST_DATA_DIR / 'taime.db'}")
os.environ.setdefault("TAIME_DISABLE_POOL", "1")
os.environ.setdefault("TAIME_DISABLE_STATIC", "1")

from taime_api.db.engine import init_db  # noqa: E402
from taime_api.main import app  # noqa: E402

init_db()


@pytest.fixture(name="client")
async def client_fixture():
    """Create an async test client."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
