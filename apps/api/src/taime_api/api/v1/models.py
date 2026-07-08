"""Model management API endpoints."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from sqlmodel import select

from taime_api.api.v1.schemas import ModelResponse, SUTType
from taime_api.db.engine import get_session
from taime_api.db.models import ModelVersion

router = APIRouter()


@router.get("", response_model=list[ModelResponse])
async def list_models(
    sut_type: SUTType | None = None,
    skip: int = 0,
    limit: int = 50,
) -> list[ModelVersion]:
    """List all trained models, optionally filtered by SUT type."""
    with get_session() as session:
        query = select(ModelVersion)
        if sut_type:
            query = query.where(ModelVersion.sut_type == sut_type)
        query = query.order_by(ModelVersion.created_at.desc()).offset(skip).limit(limit)
        models = session.exec(query).all()
        return list(models)


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model_id: int) -> ModelVersion:
    """Get a specific model by ID."""
    with get_session() as session:
        model = session.get(ModelVersion, model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        return model


@router.get("/{model_id}/download")
async def download_model(model_id: int) -> FileResponse:
    """Download a trained model artifact."""
    with get_session() as session:
        model = session.get(ModelVersion, model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        file_path = Path(model.file_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Model file not found")

        return FileResponse(
            path=file_path,
            filename=file_path.name,
            media_type="application/octet-stream",
        )
