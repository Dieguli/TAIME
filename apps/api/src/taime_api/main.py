"""TAIME v2.0 - Trustworthy AI Models Engine API."""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request

from taime_api.api.v1.router import api_router
from taime_api.db.engine import init_db
from taime_api.services.jobs import reconcile_interrupted_jobs
from taime_api.workers.pool import shutdown_pool, start_pool

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = BASE_DIR / "static"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(1024 * 1024 * 1024)))


class LargeUploadRequest(Request):
    async def form(
        self,
        *,
        max_files: int | float = 1000,
        max_fields: int | float = 1000,
        max_part_size: int = MAX_UPLOAD_BYTES,
    ):
        return await super().form(
            max_files=max_files,
            max_fields=max_fields,
            max_part_size=max_part_size,
        )


class SPAStaticFiles(StaticFiles):
    """Serve the built SPA, falling back to ``index.html`` for history routes.

    The React app uses BrowserRouter, so a refresh, bookmark, or shared deep
    link to ``/jobs`` / ``/datasets`` / ``/models`` reaches the server as a real
    path. Plain :class:`StaticFiles` 404s those; we serve ``index.html`` instead
    so the client router can take over. API and docs routes are registered
    before this mount (so they never reach here), and we leave ``/api`` 404s as
    real 404s rather than masking them with the SPA shell.
    """

    async def get_response(self, path: str, scope):  # type: ignore[override]
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and not path.startswith("api"):
                return await super().get_response("index.html", scope)
            raise


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    # Startup
    init_db()
    # A previous process may have died (crash/redeploy/`docker compose restart`)
    # while jobs were pending/running; their worker is gone, so fail them now
    # instead of leaving ghost rows stuck "running" forever.
    reconcile_interrupted_jobs()
    start_pool()
    yield
    # Shutdown
    shutdown_pool()


app = FastAPI(
    title="TAIME v2.0",
    description="Trustworthy AI Models Engine - SUT Model Retraining Platform",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)
app.request_class = LargeUploadRequest

# Mount API router
app.include_router(api_router, prefix="/api/v1")


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "version": "2.0.0"}


# Mount static files (SPA) if directory exists
if STATIC_DIR.exists() and not os.getenv("TAIME_DISABLE_STATIC"):
    app.mount("/", SPAStaticFiles(directory=str(STATIC_DIR), html=True), name="static")
