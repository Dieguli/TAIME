"""SQLModel database models."""

from datetime import datetime
from typing import Any

from sqlmodel import JSON, Column, Field, SQLModel


class Dataset(SQLModel, table=True):
    """Dataset metadata table."""

    __tablename__ = "datasets"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(max_length=255, index=True)
    filename: str = Field(max_length=255)
    file_path: str = Field(max_length=1024)
    file_size: int = Field(default=0)
    sut_type: str = Field(max_length=50, index=True)
    description: str | None = Field(default=None, max_length=2048)
    row_count: int | None = Field(default=None)
    column_names: list[str] | None = Field(default=None, sa_column=Column(JSON))
    version: int = Field(default=1)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Job(SQLModel, table=True):
    """Training job table."""

    __tablename__ = "jobs"

    id: int | None = Field(default=None, primary_key=True)
    dataset_id: int = Field(foreign_key="datasets.id", index=True)
    sut_type: str = Field(max_length=50, index=True)
    status: str = Field(default="pending", max_length=20, index=True)
    progress: float = Field(default=0.0)
    current_epoch: int | None = Field(default=None)
    total_epochs: int | None = Field(default=None)
    config: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    metrics: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    error_message: str | None = Field(default=None, max_length=4096)
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ModelVersion(SQLModel, table=True):
    """Trained model version table."""

    __tablename__ = "model_versions"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(max_length=255, index=True)
    job_id: int = Field(foreign_key="jobs.id", index=True)
    sut_type: str = Field(max_length=50, index=True)
    version: int = Field(default=1)
    file_path: str = Field(max_length=1024)
    file_size: int = Field(default=0)
    metrics: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    config: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)
