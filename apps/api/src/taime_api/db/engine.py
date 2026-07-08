"""SQLite database engine configuration."""

import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy.pool import NullPool
from sqlmodel import Session, SQLModel, create_engine

# Get database path from environment or use default
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'taime.db'}")

# Create engine.
#
# SQLite hardening for the multiprocessing training path:
#   * ``timeout=30`` sets SQLite's busy timeout. The default is 0, which makes a
#     contended write raise "database is locked" immediately — and the training
#     worker writes progress while the API polls job status, so contention is
#     routine. 30s lets a caller wait for the lock instead of erroring out.
#   * ``poolclass=NullPool`` means each Session opens (and closes) its own
#     connection rather than reusing a pooled one. On Linux/Docker the worker is
#     forked from the API process after the engine has opened a connection;
#     NullPool guarantees no live connection/file-descriptor is shared across
#     that fork (which would otherwise risk "database is locked"/corruption).
engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False, "timeout": 30},
    poolclass=NullPool,
)


def init_db() -> None:
    """Initialize database - create all tables."""
    # Import models to ensure they're registered
    from taime_api.db import models  # noqa: F401

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Create all tables
    SQLModel.metadata.create_all(engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Get a database session context manager."""
    session = Session(engine, expire_on_commit=False)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
