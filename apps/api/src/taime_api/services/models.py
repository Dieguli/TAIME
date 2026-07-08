"""Models service - business logic for model management."""

from sqlmodel import select

from taime_api.api.v1.schemas import SUTType
from taime_api.db.engine import get_session
from taime_api.db.models import ModelVersion


class ModelService:
    """Service for model operations."""

    async def list_models(
        self,
        sut_type: SUTType | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> list[ModelVersion]:
        """List all trained models."""
        with get_session() as session:
            query = select(ModelVersion)
            if sut_type:
                query = query.where(ModelVersion.sut_type == sut_type.value)
            query = query.order_by(ModelVersion.created_at.desc()).offset(skip).limit(limit)
            models = session.exec(query).all()
            return list(models)

    async def get_model(self, model_id: int) -> ModelVersion | None:
        """Get a specific model by ID."""
        with get_session() as session:
            return session.get(ModelVersion, model_id)
