# syntax=docker/dockerfile:1
# TAIME v2.0 — CPU image (primary demo deployment).
# Single container: FastAPI serves the API (/api/v1, /docs) and the built React SPA.

# --- Stage 1: build the React frontend -------------------------------------
FROM node:20-slim AS web-build
WORKDIR /app/apps/web
# Install deps from the committed lockfile (npm ci) for reproducible builds.
COPY apps/web/package.json apps/web/package-lock.json ./
RUN npm ci
COPY apps/web/ ./
# Vite outDir is ../api/static -> emits to /app/apps/api/static
RUN npm run build

# --- Stage 2: Python runtime (CPU) -----------------------------------------
FROM python:3.11-slim AS runtime

# CPU torch wheels by default; overridable for other CPU index mirrors.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# Bake the DistilBERT-cased fake-news base model at build time (needs network
# during build). This keeps runtime fully offline so the default `docker compose
# up` can retrain fake news with zero extra setup. Set to "false" to skip it.
ARG FETCH_MODELS=true

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_XET=1 \
    DATA_DIR=/data \
    HOST=0.0.0.0 \
    PORT=8000

# Fake-news base model lives outside the /data volume so the runtime bind mount
# cannot shadow the copy baked into the image at build time.
ENV FAKE_NEWS_MODEL_DIR=/opt/taime-models/hf/distilbert-base-cased

# libgomp1 is required at runtime by torch / xgboost.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU torch build first so the editable install reuses it
# (avoids pip pulling the much larger default CUDA wheel from PyPI).
RUN pip install --upgrade pip \
    && pip install "torch>=2.4,<2.8" --index-url ${TORCH_INDEX_URL}

# Backend source + editable install. Editable keeps the apps/api/src layout so
# main.py resolves STATIC_DIR to /app/apps/api/static.
COPY apps/api/ /app/apps/api/
COPY scripts/ /app/scripts/
# pyproject.toml references ../../README.md for its long description.
COPY README.md /app/README.md
RUN pip install -e /app/apps/api

# Built SPA from the web stage.
COPY --from=web-build /app/apps/api/static /app/apps/api/static

# Bake the fake-news base model into the image, outside the /data volume so the
# runtime bind mount does not hide it. DATA_DIR is overridden just for this step
# so fetch_models writes to <FAKE_NEWS_MODEL_DIR>'s parent (/opt/taime-models/hf).
# Remove hf_xet so huggingface_hub uses plain HTTP: darts forces
# huggingface_hub>=0.35.3 (Xet-capable), and the Xet read-token endpoint 404s
# during unauthenticated builds. HF_HUB_DISABLE_XET=1 is set as a belt-and-braces.
RUN if [ "$FETCH_MODELS" = "true" ]; then \
        pip uninstall -y hf_xet >/dev/null 2>&1 || true; \
        DATA_DIR=/opt/taime-models python /app/scripts/fetch_models.py; \
    fi

RUN mkdir -p /data/datasets /data/models /data/logs
VOLUME ["/data"]
EXPOSE 8000

# DB init + worker pool start are handled by the FastAPI lifespan in main.py.
CMD ["uvicorn", "taime_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
