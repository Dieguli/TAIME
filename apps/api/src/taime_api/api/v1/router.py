"""API v1 router - aggregates all endpoint routers."""

from fastapi import APIRouter

from taime_api.api.v1.datasets import router as datasets_router
from taime_api.api.v1.jobs import router as jobs_router
from taime_api.api.v1.models import router as models_router
from taime_api.api.v1.runtime import router as runtime_router

api_router = APIRouter()

api_router.include_router(datasets_router, prefix="/datasets", tags=["Datasets"])
api_router.include_router(jobs_router, prefix="/jobs", tags=["Jobs"])
api_router.include_router(models_router, prefix="/models", tags=["Models"])
api_router.include_router(runtime_router, prefix="/runtime", tags=["Runtime"])
