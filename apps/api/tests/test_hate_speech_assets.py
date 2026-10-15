"""Hate-speech base-model hydration: each dataset trains on the bundle it was uploaded with.

The bundle files below are placeholders; every test stops before the tokenizer or
model is actually loaded, so no real RoBERTa weights are needed.
"""

import json
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("transformers")
from taime_api.training import hate_speech_transformer as hs  # noqa: E402

TOP = "hate_speech_dataset"


def _bundle(weights: bytes, weights_folder: str = "ROBERTA_hate_speech") -> dict[str, bytes]:
    models = f"{TOP}/models"
    return {
        f"{models}/tokenizer/vocab.json": b"{}",
        f"{models}/tokenizer/merges.txt": b"#version 0.2",
        f"{models}/{weights_folder}/config.json": json.dumps({"model_type": "roberta"}).encode(),
        f"{models}/{weights_folder}/model.safetensors": weights,
    }


def _csvs() -> dict[str, bytes]:
    mlma = "tweet,sentiment,target\n" + "".join(
        f"tweet {i},{'normal' if i % 2 else 'hateful'},origin\n" for i in range(10)
    )
    conan = "HATE_SPEECH,COUNTER_NARRATIVE,TARGET\n" + "".join(
        f"statement {i},reply {i},MIGRANTS\n" for i in range(10)
    )
    return {f"{TOP}/en_dataset.csv": mlma.encode(), f"{TOP}/Multitarget-CONAN.csv": conan.encode()}


def _upload(root: Path, name: str, members: dict[str, bytes]) -> Path:
    """Mirror an upload: the zip is stored in the dataset dir and extracted next to it."""
    dataset_dir = root / "datasets" / name
    dataset_dir.mkdir(parents=True)
    zip_path = dataset_dir / f"{name}.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for member, data in members.items():
            archive.writestr(member, data)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dataset_dir)
    return dataset_dir


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    # The trainer resolves its default paths relative to the working directory.
    monkeypatch.chdir(tmp_path)


def test_each_dataset_trains_on_its_own_model_bundle(tmp_path):
    first = _upload(tmp_path, "first", _bundle(b"weights-first"))
    second = _upload(tmp_path, "second", _bundle(b"weights-second"))

    hs._ensure_assets(first)
    assets = hs._ensure_assets(second)

    assert (assets["binary"] / "model.safetensors").read_bytes() == b"weights-second"


def test_leftover_shared_cache_does_not_override_the_uploaded_bundle(tmp_path):
    # State left behind by older TAIME versions after a failed first run:
    # tokenizer cached, weights folders empty.
    old_cache = tmp_path / "data" / "hf" / "hate_speech"
    (old_cache / "tokenizer").mkdir(parents=True)
    (old_cache / "tokenizer" / "vocab.json").write_text("{}")
    (old_cache / "tokenizer" / "merges.txt").write_text("#version 0.2")
    (old_cache / "binary").mkdir()
    (old_cache / "multilabel").mkdir()
    dataset = _upload(tmp_path, "fixed", _bundle(b"weights-fixed"))

    assets = hs._ensure_assets(dataset)

    assert (assets["binary"] / "model.safetensors").read_bytes() == b"weights-fixed"


def test_hydrated_bundle_is_reused_for_the_same_dataset(tmp_path):
    dataset = _upload(tmp_path, "reuse", _bundle(b"weights-reuse"))
    hs._ensure_assets(dataset)
    for zip_path in dataset.rglob("*.zip"):
        zip_path.unlink()  # a second run must not need to re-read the archive

    assets = hs._ensure_assets(dataset)

    assert (assets["binary"] / "model.safetensors").read_bytes() == b"weights-reuse"


def test_loose_model_files_fail_with_the_expected_layout(tmp_path):
    loose = {Path(name).name: data for name, data in _bundle(b"weights").items()}
    dataset = _upload(tmp_path, "loose", {**_csvs(), **loose})

    with pytest.raises(FileNotFoundError, match="models/ROBERTA_hate_speech/"):
        hs.train_hate_speech_model(
            dataset, {"classification_type": "binary"}, tmp_path / "models", job_id=1
        )


def test_multilabel_mode_needs_the_multilabel_weights_folder(tmp_path):
    dataset = _upload(tmp_path, "binary_only", {**_csvs(), **_bundle(b"weights")})

    with pytest.raises(FileNotFoundError, match="models/ROBERTA_hate_speech_multilabel/"):
        hs.train_hate_speech_model(
            dataset, {"classification_type": "multilabel"}, tmp_path / "models", job_id=1
        )
