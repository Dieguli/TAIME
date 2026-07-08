"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# Enums
class SUTType(str, Enum):
    """Supported SUT (System Under Test) types."""

    HEALTHCARE_PC = "healthcare_pc"  # Pancreatic Cancer Risk Predictor
    INFRA_PORT = "infra_port"  # Port operations (TSMixer)
    DISINFO_HATE = "disinfo_hate"  # Hate Speech Detector
    DISINFO_FAKE = "disinfo_fake"  # Fake News Detector


class JobStatus(str, Enum):
    """Training job status values."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Dataset schemas
class DatasetBase(BaseModel):
    """Base dataset schema."""

    name: str = Field(..., min_length=1, max_length=255)
    sut_type: SUTType
    description: str | None = None


class DatasetCreate(DatasetBase):
    """Schema for dataset creation (upload)."""

    pass


class DatasetResponse(DatasetBase):
    """Schema for dataset response."""

    id: int
    filename: str
    file_size: int
    row_count: int | None = None
    version: int = 1
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DatasetPreview(BaseModel):
    """Schema for dataset preview response."""

    id: int
    name: str
    columns: list[str]
    rows: list[dict[str, Any]]
    total_rows: int
    statistics: dict[str, Any] | None = None


# Job schemas
class JobBase(BaseModel):
    """Base job schema."""

    dataset_id: int
    sut_type: SUTType


class JobCreate(JobBase):
    """Schema for job creation."""

    config: dict[str, Any] = Field(
        default_factory=lambda: {
            "epochs": 10,
            "learning_rate": 0.001,
            "batch_size": 32,
        }
    )


class JobResponse(JobBase):
    """Schema for job response."""

    id: int
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    current_epoch: int | None = None
    total_epochs: int | None = None
    config: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Model schemas
class ModelBase(BaseModel):
    """Base model schema."""

    name: str
    sut_type: SUTType


class ModelResponse(ModelBase):
    """Schema for model response."""

    id: int
    job_id: int
    version: int
    file_path: str
    file_size: int
    metrics: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    created_at: datetime

    class Config:
        from_attributes = True


# Generic schemas
class PaginatedResponse(BaseModel):
    """Generic paginated response."""

    items: list[Any]
    total: int
    page: int = 1
    page_size: int = 50


class ErrorResponse(BaseModel):
    """Error response schema."""

    detail: str
    code: str | None = None


class RuntimeResponse(BaseModel):
    """Runtime capability response for the demo UI."""

    version: str
    image: str = "unknown"
    cuda_available: bool = False
    device_name: str | None = None
    torch_version: str | None = None
