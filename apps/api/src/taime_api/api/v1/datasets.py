"""Dataset API endpoints."""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from sqlmodel import select

from taime_api.api.v1.schemas import (
    DatasetPreview,
    DatasetResponse,
    SUTType,
)
from taime_api.db.engine import get_session
from taime_api.db.models import Dataset
from taime_api.services.datasets import DatasetService

router = APIRouter()

FILE_REQUIRED = File(...)
NAME_FORM = Form(...)
SUT_TYPE_FORM = Form(...)
DESCRIPTION_FORM = Form(None)


@router.get("", response_model=list[DatasetResponse])
async def list_datasets(
    sut_type: SUTType | None = None,
    skip: int = 0,
    limit: int = 50,
) -> list[Dataset]:
    """List all datasets, optionally filtered by SUT type."""
    with get_session() as session:
        query = select(Dataset)
        if sut_type:
            query = query.where(Dataset.sut_type == sut_type)
        query = query.offset(skip).limit(limit)
        datasets = session.exec(query).all()
        return list(datasets)


@router.post("", response_model=DatasetResponse, status_code=201)
async def upload_dataset(
    file: UploadFile = FILE_REQUIRED,
    name: str = NAME_FORM,
    sut_type: SUTType = SUT_TYPE_FORM,
    description: str | None = DESCRIPTION_FORM,
) -> Dataset:
    """Upload a new dataset."""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if file_ext != ".zip":
        raise HTTPException(status_code=400, detail="Only .zip uploads are supported")

    service = DatasetService()
    dataset = await service.create_dataset(
        file=file,
        name=name,
        sut_type=sut_type,
        description=description,
    )
    return dataset


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: int) -> Dataset:
    """Get a specific dataset by ID."""
    with get_session() as session:
        dataset = session.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return dataset


@router.get("/{dataset_id}/preview", response_model=DatasetPreview)
async def preview_dataset(dataset_id: int, rows: int = 100) -> DatasetPreview:
    """Preview the first N rows of a dataset."""
    with get_session() as session:
        dataset = session.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

    service = DatasetService()
    preview = await service.get_preview(dataset_id, max_rows=min(rows, 100))
    return preview


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: int) -> None:
    """Delete a dataset."""
    service = DatasetService()
    await service.delete_dataset(dataset_id)
