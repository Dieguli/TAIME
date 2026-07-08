#!/usr/bin/env python3
"""Fetch base HuggingFace model assets for offline (local_files_only) training.

By default downloads **DistilBERT-cased** (the partner-approved fake-news base)
into ``$DATA_DIR/hf/<model-name>`` so the trainer can load it with
``local_files_only=True``. Safe to re-run; ``snapshot_download`` is incremental.

Requires network access. Used both by host setup and (optionally) at Docker build.

Usage:
    python scripts/fetch_models.py
    python scripts/fetch_models.py distilbert-base-cased distilbert-base-uncased
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Fall back to plain HTTP downloads instead of the Xet backend, whose read-token
# endpoint 404s for unauthenticated requests (e.g. during the Docker build).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

DEFAULT_MODELS = ["distilbert-base-cased"]
# Skip framework variants we never load (TensorFlow / Flax / ONNX / Rust).
IGNORE_PATTERNS = ["*.h5", "*.msgpack", "*.onnx", "*.ot", "rust_model.ot"]


def fetch(model_id: str, hf_root: Path) -> Path:
    from huggingface_hub import snapshot_download

    local_dir = hf_root / model_id.split("/")[-1]
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {model_id} -> {local_dir}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        ignore_patterns=IGNORE_PATTERNS,
    )
    return local_dir


def main(argv: list[str]) -> int:
    models = argv[1:] or DEFAULT_MODELS
    hf_root = Path(os.getenv("DATA_DIR", "data")) / "hf"
    for model_id in models:
        fetch(model_id, hf_root)
    print(f"Done. Assets in {hf_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
