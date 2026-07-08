"""Tests for model-artifact packaging and the download endpoint.

These cover the *export* step of the demo loop without the ML stack. The four
trainers save models in three on-disk shapes:

* transformers (fake news, hate speech) -> a ``save_pretrained`` **directory**,
* Darts (port) -> a ``.pt`` file plus a sibling ``.pt.ckpt`` checkpoint,
* XGBoost (healthcare) -> a single ``.pkl`` file.

``_package_model_artifact`` must turn each into one downloadable ``.zip`` (a
directory path previously made ``download_model`` 500), and
``GET /models/{id}/download`` must stream a valid archive.
"""

import io
import zipfile

import pytest

from taime_api.db.engine import get_session
from taime_api.db.models import ModelVersion
from taime_api.workers.runner import _package_model_artifact


def _insert_model(file_path: str, size: int, sut_type: str = "disinfo_fake") -> int:
    with get_session() as session:
        model = ModelVersion(
            name=f"{sut_type}_v1",
            job_id=1,
            sut_type=sut_type,
            version=1,
            file_path=file_path,
            file_size=size,
        )
        session.add(model)
        session.commit()
        session.refresh(model)
        return model.id


def test_package_directory_artifact(tmp_path):
    # Mimic transformers save_pretrained(): a directory of files.
    model_dir = tmp_path / "fake_news_model_7"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (model_dir / "model.safetensors").write_bytes(b"weights")

    zip_path, size = _package_model_artifact(model_dir)

    assert zip_path == tmp_path / "fake_news_model_7.zip"
    assert zip_path.is_file()
    assert size == zip_path.stat().st_size
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert names == {
        "fake_news_model_7/config.json",
        "fake_news_model_7/model.safetensors",
    }


def test_package_single_file_artifact(tmp_path):
    # Mimic XGBoost healthcare .pkl: a single file, no sidecars.
    model_file = tmp_path / "healthcare_pc_model_7.pkl"
    model_file.write_bytes(b"pickle-bytes")

    zip_path, size = _package_model_artifact(model_file)

    assert zip_path == tmp_path / "healthcare_pc_model_7.pkl.zip"
    assert size == zip_path.stat().st_size
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ["healthcare_pc_model_7.pkl"]
        assert archive.read("healthcare_pc_model_7.pkl") == b"pickle-bytes"


def test_package_file_with_sidecar(tmp_path):
    # Mimic Darts port save(): "<name>.pt" plus a sibling "<name>.pt.ckpt".
    model_file = tmp_path / "port_model_7.pt"
    model_file.write_bytes(b"darts-model")
    (tmp_path / "port_model_7.pt.ckpt").write_bytes(b"lightning-ckpt")

    zip_path, _ = _package_model_artifact(model_file)

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert names == {"port_model_7.pt", "port_model_7.pt.ckpt"}
    # The freshly-created zip must never package itself.
    assert "port_model_7.pt.zip" not in names


@pytest.mark.anyio
async def test_download_directory_model_returns_zip(client, tmp_path):
    # The case that used to 500: a directory-shaped model, packaged and served
    # by the download endpoint as a streamable, valid zip.
    model_dir = tmp_path / "hate_speech_model_binary_9"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (model_dir / "vocab.json").write_text("{}")
    zip_path, size = _package_model_artifact(model_dir)

    model_id = _insert_model(str(zip_path), size, sut_type="disinfo_hate")

    resp = await client.get(f"/api/v1/models/{model_id}/download")
    assert resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
    assert names == {
        "hate_speech_model_binary_9/config.json",
        "hate_speech_model_binary_9/vocab.json",
    }


@pytest.mark.anyio
async def test_download_missing_model_file_returns_404(client, tmp_path):
    missing = tmp_path / "gone.zip"
    model_id = _insert_model(str(missing), 0)
    resp = await client.get(f"/api/v1/models/{model_id}/download")
    assert resp.status_code == 404
