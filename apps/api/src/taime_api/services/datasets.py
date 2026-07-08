"""Dataset service - business logic for dataset management."""

import json
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException, UploadFile

from taime_api.api.v1.schemas import DatasetPreview, SUTType
from taime_api.db.engine import get_session
from taime_api.db.models import Dataset
from taime_api.utils.port_dataset import (
    PortDatasetError,
    load_meta,
    resolve_port_dataset_root,
    summarize_meta,
)
from taime_api.validators.port import validate_port_dataset

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATASETS_DIR = DATA_DIR / "datasets"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(1024 * 1024 * 1024)))
PORT_ZIP_EXT = ".zip"

HEALTHCARE_FEATURE_FILES = ("mup_risk_model_data.csv", "MUP_data_processed.csv")
HEALTHCARE_LABEL_FILES = ("y_train_model_data.csv",)
FAKE_NEWS_FILES = ("Fake.csv", "True.csv")
HATE_SPEECH_FILES = ("en_dataset.csv", "Multitarget-CONAN.csv")
HATE_SPEECH_MODEL_HINTS = (
    "atc-code-transformers-hate-speech-detection",
    "fakenews_hatespeech_model",
)


class DatasetService:
    """Service for dataset operations."""

    def __init__(self) -> None:
        """Initialize the dataset service."""
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    def _safe_extract_zip(self, zip_path: Path, extract_dir: Path) -> None:
        """Extract zip file while preventing path traversal."""
        extract_root = extract_dir.resolve()
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                member_path = (extract_dir / member.filename).resolve()
                if not member_path.is_relative_to(extract_root):
                    raise HTTPException(status_code=400, detail="Invalid zip contents")
            archive.extractall(extract_dir)

    def _find_first(self, dataset_dir: Path, filenames: tuple[str, ...]) -> Path | None:
        """Find the first matching file within a dataset directory."""
        for name in filenames:
            matches = list(dataset_dir.rglob(name))
            if matches:
                return matches[0]
        return None

    def _read_csv_flexible(self, file_path: Path) -> pd.DataFrame:
        """Read CSV data, falling back to semicolon delimiter if needed."""
        df = pd.read_csv(file_path)
        if len(df.columns) == 1 and ";" in df.columns[0]:
            df = pd.read_csv(file_path, sep=";")
        return df

    def _count_csv_rows(self, file_path: Path) -> tuple[int, list[str]]:
        """Count CSV rows without loading full dataset into memory."""
        row_count = 0
        columns: list[str] = []
        for chunk in pd.read_csv(file_path, sep=None, engine="python", chunksize=50000):
            if not columns:
                columns = list(chunk.columns)
            row_count += len(chunk)
        return row_count, columns

    def _inspect_healthcare_dataset(self, dataset_dir: Path) -> tuple[int | None, list[str] | None]:
        """Inspect healthcare dataset files for row count and feature columns."""
        feature_path = self._find_first(dataset_dir, HEALTHCARE_FEATURE_FILES)
        if not feature_path:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Healthcare dataset missing required file. Expected one of: "
                    f"{', '.join(HEALTHCARE_FEATURE_FILES)}"
                ),
            )
        df = self._read_csv_flexible(feature_path)
        row_count = len(df)
        columns = list(df.columns)
        return row_count, columns

    def _inspect_fake_news_dataset(self, dataset_dir: Path) -> tuple[int | None, list[str] | None]:
        """Inspect fake news dataset files for row count and columns."""
        fake_path = self._find_first(dataset_dir, ("Fake.csv",))
        true_path = self._find_first(dataset_dir, ("True.csv",))
        if not fake_path or not true_path:
            raise HTTPException(
                status_code=400,
                detail="Fake news dataset missing required files: Fake.csv and True.csv",
            )
        fake_rows = sum(
            len(chunk) for chunk in pd.read_csv(fake_path, usecols=[0], chunksize=50000)
        )
        true_rows = sum(
            len(chunk) for chunk in pd.read_csv(true_path, usecols=[0], chunksize=50000)
        )
        df = pd.read_csv(fake_path, nrows=0)
        return fake_rows + true_rows, list(df.columns)

    def _inspect_hate_speech_dataset(
        self, dataset_dir: Path
    ) -> tuple[int | None, list[str] | None]:
        """Inspect hate speech dataset files for row count and columns."""
        mlma_path = self._find_first(dataset_dir, ("en_dataset.csv",))
        conan_path = self._find_first(dataset_dir, ("Multitarget-CONAN.csv",))
        if not mlma_path or not conan_path:
            raise HTTPException(
                status_code=400,
                detail="Hate speech dataset missing required files: en_dataset.csv and Multitarget-CONAN.csv",
            )
        if not self._has_hate_speech_model_bundle(dataset_dir):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Hate speech dataset zip must include the model bundle "
                    "(e.g., atc-code-transformers-hate-speech-detection-*.zip "
                    "or FakeNews_HateSpeech_model.zip)."
                ),
            )
        mlma_rows = sum(
            len(chunk) for chunk in pd.read_csv(mlma_path, usecols=[0], chunksize=50000)
        )
        conan_rows = sum(
            len(chunk) for chunk in pd.read_csv(conan_path, usecols=[0], chunksize=50000)
        )
        df = pd.read_csv(mlma_path, nrows=0)
        return mlma_rows + conan_rows, list(df.columns)

    def _zip_contains_hate_assets(self, zip_path: Path) -> bool:
        """Check if a zip contains hate speech model assets or nested model zips."""
        try:
            with zipfile.ZipFile(zip_path) as archive:
                for member in archive.infolist():
                    name = member.filename
                    if "/models/ROBERTA_hate_speech/" in name:
                        return True
                    if "/models/ROBERTA_hate_speech_multilabel/" in name:
                        return True
                    if "/models/tokenizer/" in name:
                        return True
                    base = Path(name).name.lower()
                    if base in {"vocab.json", "merges.txt"}:
                        return True
                    if base.endswith(".zip") and any(
                        hint in base for hint in HATE_SPEECH_MODEL_HINTS
                    ):
                        return True
        except zipfile.BadZipFile:
            return False
        return False

    def _has_hate_speech_model_bundle(self, dataset_dir: Path) -> bool:
        """Ensure the dataset bundle includes hate speech model assets."""
        vocab = self._find_first(dataset_dir, ("vocab.json",))
        merges = self._find_first(dataset_dir, ("merges.txt",))
        if vocab and merges:
            return True
        for candidate in dataset_dir.rglob("*.zip"):
            if self._zip_contains_hate_assets(candidate):
                return True
        return False

    def _inspect_generic_bundle(self, dataset_dir: Path) -> tuple[int | None, list[str] | None]:
        """Inspect a generic bundle (first CSV/JSON encountered)."""
        csv_candidates = list(dataset_dir.rglob("*.csv"))
        if csv_candidates:
            return self._count_csv_rows(csv_candidates[0])

        json_candidates = list(dataset_dir.rglob("*.json"))
        if json_candidates:
            with open(json_candidates[0]) as f:
                data = json.load(f)
            if isinstance(data, list):
                if data and isinstance(data[0], dict):
                    return len(data), list(data[0].keys())
                return len(data), []
            if isinstance(data, dict):
                return 1, list(data.keys())

        return None, None

    def _inspect_port_dataset(self, dataset_dir: Path) -> tuple[int | None, list[str] | None]:
        """Inspect port dataset meta files to derive counts."""
        validation = validate_port_dataset(dataset_dir)
        if not validation.valid:
            raise PortDatasetError(
                "Port dataset invalid. Required files: series_meta.json, past_covs_meta.json, "
                "and referenced series_*.parquet / past_covs_*.parquet. "
                + "; ".join(validation.errors)
            )
        summary = validation.summary
        if summary is None:
            raise PortDatasetError("Missing port dataset summary")
        return summary.series_count, ["Estado"]

    def _port_preview_summary(self, dataset_dir: Path) -> DatasetPreview:
        """Return a safe preview summary for port datasets."""
        root = resolve_port_dataset_root(dataset_dir)
        series_meta = load_meta(root / "series_meta.json")
        past_meta = load_meta(root / "past_covs_meta.json")
        summary = summarize_meta(series_meta, past_meta)

        columns = [
            "series_count",
            "past_covariates_count",
            "series_columns",
            "past_covariates_columns",
            "index_names",
            "static_covariate_keys",
            "start_min",
            "end_max",
            "frequency",
        ]

        # Infer frequency from first series parquet if possible
        frequency = "unknown"
        series_files = sorted(root.glob("series_*.parquet"))
        if series_files:
            df = pd.read_parquet(series_files[0])
            if isinstance(df.index, pd.DatetimeIndex):
                idx = df.index
            elif "date" in df.columns:
                idx = pd.to_datetime(df["date"])
            else:
                idx = None
            if idx is not None and len(idx) > 2:
                idx = pd.DatetimeIndex(idx).sort_values()
                frequency = pd.infer_freq(idx) or "unknown"

        start_min = min(summary.start_values) if summary.start_values else None
        end_max = max(summary.end_values) if summary.end_values else None

        row = {
            "series_count": summary.series_count,
            "past_covariates_count": summary.past_covariates_count,
            "series_columns": sorted(summary.series_columns),
            "past_covariates_columns": sorted(summary.past_covariates_columns),
            "index_names": sorted(summary.index_names),
            "static_covariate_keys": sorted(summary.static_covariate_keys),
            "start_min": start_min,
            "end_max": end_max,
            "frequency": frequency,
        }

        return DatasetPreview(
            id=0,
            name="Port Dataset Summary",
            columns=columns,
            rows=[row],
            total_rows=summary.series_count,
            statistics=None,
        )

    def _summarize_text_lengths(
        self, file_path: Path, column: str, limit: int = 5000
    ) -> dict[str, float]:
        """Compute text length statistics from a limited sample."""
        lengths: list[int] = []
        for chunk in pd.read_csv(file_path, usecols=[column], chunksize=2000):
            series = chunk[column].fillna("").astype(str)
            lengths.extend(series.str.split().str.len().tolist())
            if len(lengths) >= limit:
                break
        if not lengths:
            return {"mean": 0.0, "p95": 0.0, "max": 0.0}
        lengths = lengths[:limit]
        lengths.sort()
        p95_index = int(0.95 * (len(lengths) - 1))
        return {
            "mean": float(sum(lengths) / len(lengths)),
            "p95": float(lengths[p95_index]),
            "max": float(lengths[-1]),
        }

    def _generic_dir_summary(self, dataset_dir: Path) -> DatasetPreview:
        """Return a safe summary for generic folder uploads."""
        files = [path for path in dataset_dir.rglob("*") if path.is_file()]
        total_size = sum(path.stat().st_size for path in files)
        extension_counts: dict[str, int] = {}
        sample_files: list[str] = []
        for path in files:
            ext = path.suffix.lower() or "no_ext"
            extension_counts[ext] = extension_counts.get(ext, 0) + 1
            if len(sample_files) < 10:
                sample_files.append(str(path.relative_to(dataset_dir)))

        stats = {
            "file_count": len(files),
            "total_size_bytes": total_size,
            "extensions": extension_counts,
            "sample_files": sample_files,
        }

        return DatasetPreview(
            id=0,
            name="Dataset Folder Summary",
            columns=[],
            rows=[],
            total_rows=len(files),
            statistics=stats,
        )

    def _healthcare_preview_summary(self, dataset_path: Path) -> DatasetPreview:
        """Return a safe preview summary for healthcare datasets."""
        if dataset_path.is_dir():
            feature_path = self._find_first(dataset_path, HEALTHCARE_FEATURE_FILES)
            label_path = self._find_first(dataset_path, HEALTHCARE_LABEL_FILES)
        else:
            feature_path = dataset_path
            label_path = self._find_first(dataset_path.parent, HEALTHCARE_LABEL_FILES)
        if not feature_path:
            raise HTTPException(status_code=400, detail="Healthcare feature file not found")

        df = self._read_csv_flexible(feature_path)
        labels: pd.Series | None = None
        label_column = None

        if label_path and label_path.exists():
            labels_df = pd.read_csv(label_path)
            label_column = "risk" if "risk" in labels_df.columns else labels_df.columns[0]
            labels = labels_df[label_column]
        else:
            for candidate in ("risk_ground_truth_decoded", "risk", "risk_prediction"):
                if candidate in df.columns:
                    label_column = candidate
                    labels = df[candidate]
                    df = df.drop(columns=[candidate])
                    break

        stats: dict[str, Any] = {
            "feature_count": len(df.columns),
            "row_count": len(df),
            "label_column": label_column,
        }
        if labels is not None:
            stats["label_counts"] = labels.value_counts(dropna=False).to_dict()

        return DatasetPreview(
            id=0,
            name="Healthcare Dataset Summary",
            columns=list(df.columns),
            rows=[],
            total_rows=len(df),
            statistics=stats,
        )

    def _fake_news_preview_summary(self, dataset_path: Path) -> DatasetPreview:
        """Return a safe preview summary for fake news datasets."""
        if dataset_path.is_dir():
            fake_path = self._find_first(dataset_path, ("Fake.csv",))
            true_path = self._find_first(dataset_path, ("True.csv",))
        else:
            fake_path = dataset_path if dataset_path.name == "Fake.csv" else None
            true_path = dataset_path if dataset_path.name == "True.csv" else None
            if dataset_path.parent:
                fake_path = fake_path or self._find_first(dataset_path.parent, ("Fake.csv",))
                true_path = true_path or self._find_first(dataset_path.parent, ("True.csv",))
        if not fake_path or not true_path:
            raise HTTPException(status_code=400, detail="Fake.csv/True.csv not found")

        fake_rows, columns = self._count_csv_rows(fake_path)
        true_rows, _ = self._count_csv_rows(true_path)
        text_stats = self._summarize_text_lengths(fake_path, "text")
        true_text_stats = self._summarize_text_lengths(true_path, "text")

        stats = {
            "row_count": fake_rows + true_rows,
            "label_counts": {"fake": fake_rows, "true": true_rows},
            "text_length_fake": text_stats,
            "text_length_true": true_text_stats,
        }

        return DatasetPreview(
            id=0,
            name="Fake News Dataset Summary",
            columns=columns,
            rows=[],
            total_rows=fake_rows + true_rows,
            statistics=stats,
        )

    def _map_hate_target(self, target: str) -> str:
        if target in {"MIGRANTS", "DISABLED", "disability", "POC", "origin"}:
            return "Racism"
        if target in {"WOMEN", "gender"}:
            return "Sexism"
        if target in {"sexual_orientation", "LGBT+"}:
            return "Sexual orientation"
        if target in {"MUSLIMS", "JEWS", "religion"}:
            return "Religious"
        return "Unknown"

    def _hate_speech_preview_summary(self, dataset_path: Path) -> DatasetPreview:
        """Return a safe preview summary for hate speech datasets."""
        if dataset_path.is_dir():
            mlma_path = self._find_first(dataset_path, ("en_dataset.csv",))
            conan_path = self._find_first(dataset_path, ("Multitarget-CONAN.csv",))
        else:
            mlma_path = dataset_path if dataset_path.name == "en_dataset.csv" else None
            conan_path = dataset_path if dataset_path.name == "Multitarget-CONAN.csv" else None
            if dataset_path.parent:
                mlma_path = mlma_path or self._find_first(dataset_path.parent, ("en_dataset.csv",))
                conan_path = conan_path or self._find_first(
                    dataset_path.parent, ("Multitarget-CONAN.csv",)
                )
        if not mlma_path or not conan_path:
            raise HTTPException(
                status_code=400,
                detail="en_dataset.csv and Multitarget-CONAN.csv not found",
            )

        mlma_rows, mlma_columns = self._count_csv_rows(mlma_path)
        conan_rows, _ = self._count_csv_rows(conan_path)

        mlma_text_stats = self._summarize_text_lengths(mlma_path, "tweet")
        conan_text_stats = self._summarize_text_lengths(conan_path, "HATE_SPEECH")

        label_counts = {"normal": 0, "hate": 0}
        df_mlma = pd.read_csv(mlma_path, usecols=["sentiment"])
        label_counts["normal"] = int((df_mlma["sentiment"] == "normal").sum())
        label_counts["hate"] = int((df_mlma["sentiment"] != "normal").sum())

        df_conan = pd.read_csv(conan_path, usecols=["TARGET"])
        target_counts: dict[str, int] = {}
        for target in df_conan["TARGET"].dropna().astype(str):
            mapped = self._map_hate_target(target)
            target_counts[mapped] = target_counts.get(mapped, 0) + 1

        stats = {
            "row_count": mlma_rows + conan_rows,
            "label_counts": label_counts,
            "target_counts": target_counts,
            "text_length_mlma": mlma_text_stats,
            "text_length_conan": conan_text_stats,
        }

        return DatasetPreview(
            id=0,
            name="Hate Speech Dataset Summary",
            columns=mlma_columns,
            rows=[],
            total_rows=mlma_rows + conan_rows,
            statistics=stats,
        )

    async def create_dataset(
        self,
        file: UploadFile,
        name: str,
        sut_type: SUTType,
        description: str | None = None,
    ) -> Dataset:
        """Create a new dataset from an uploaded zip file."""
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        if Path(file.filename).suffix.lower() != PORT_ZIP_EXT:
            raise HTTPException(status_code=400, detail="Only .zip uploads are supported")

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name).strip("._-")
        if not safe_name:
            safe_name = "dataset"

        row_count: int | None = None
        column_names: list[str] | None = None

        dataset_dir = DATASETS_DIR / f"{safe_name}_{timestamp}"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        base_filename = os.path.basename(file.filename)
        filename = f"{safe_name}_{timestamp}_{base_filename}"
        file_path = dataset_dir / filename
        file_size = 0
        with file_path.open("wb") as out_file:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Dataset zip is too large")
                out_file.write(chunk)

        try:
            self._safe_extract_zip(file_path, dataset_dir)
            if sut_type == SUTType.INFRA_PORT:
                row_count, column_names = self._inspect_port_dataset(dataset_dir)
            elif sut_type == SUTType.HEALTHCARE_PC:
                row_count, column_names = self._inspect_healthcare_dataset(dataset_dir)
            elif sut_type == SUTType.DISINFO_FAKE:
                row_count, column_names = self._inspect_fake_news_dataset(dataset_dir)
            elif sut_type == SUTType.DISINFO_HATE:
                row_count, column_names = self._inspect_hate_speech_dataset(dataset_dir)
            else:
                row_count, column_names = self._inspect_generic_bundle(dataset_dir)
                if row_count is None:
                    raise HTTPException(
                        status_code=400,
                        detail="Dataset zip must include at least one .csv or .json file",
                    )
        except PortDatasetError as exc:
            shutil.rmtree(dataset_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            shutil.rmtree(dataset_dir, ignore_errors=True)
            raise
        except Exception as exc:  # pragma: no cover - defensive cleanup
            shutil.rmtree(dataset_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        file_path = dataset_dir

        # Parse file to get row count and columns
        try:
            if row_count is None:
                row_count, column_names = self._inspect_generic_bundle(file_path)
        except Exception:
            pass  # Schema validation can be added later

        with get_session() as session:
            dataset = Dataset(
                name=name,
                filename=filename,
                file_path=str(dataset_dir),
                file_size=file_size,
                sut_type=sut_type.value,
                description=description,
                row_count=row_count,
                column_names=column_names,
            )
            session.add(dataset)
            session.commit()
            session.refresh(dataset)
            return dataset

    async def get_preview(self, dataset_id: int, max_rows: int = 100) -> DatasetPreview:
        """Get preview of dataset (first N rows)."""
        with get_session() as session:
            dataset = session.get(Dataset, dataset_id)
            if not dataset:
                raise HTTPException(status_code=404, detail="Dataset not found")

            file_path = Path(dataset.file_path)
            if not file_path.exists():
                raise HTTPException(status_code=404, detail="Dataset file not found")

            if dataset.sut_type == SUTType.INFRA_PORT.value:
                preview = self._port_preview_summary(file_path)
                preview.id = dataset.id or 0
                preview.name = dataset.name
                return preview
            if dataset.sut_type == SUTType.HEALTHCARE_PC.value:
                preview = self._healthcare_preview_summary(file_path)
                preview.id = dataset.id or 0
                preview.name = dataset.name
                return preview
            if dataset.sut_type == SUTType.DISINFO_FAKE.value:
                preview = self._fake_news_preview_summary(file_path)
                preview.id = dataset.id or 0
                preview.name = dataset.name
                return preview
            if dataset.sut_type == SUTType.DISINFO_HATE.value:
                preview = self._hate_speech_preview_summary(file_path)
                preview.id = dataset.id or 0
                preview.name = dataset.name
                return preview

            if file_path.is_dir():
                preview = self._generic_dir_summary(file_path)
                preview.id = dataset.id or 0
                preview.name = dataset.name
                return preview

            # Load data
            if dataset.filename.endswith(".csv"):
                df = pd.read_csv(file_path, nrows=max_rows)
            elif dataset.filename.endswith(".json"):
                with open(file_path) as f:
                    data = json.load(f)
                df = pd.DataFrame(data[:max_rows] if isinstance(data, list) else [data])
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format")

            # Calculate statistics
            statistics: dict[str, Any] = {}
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    statistics[col] = {
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "mean": float(df[col].mean()),
                        "std": float(df[col].std()),
                    }

            return DatasetPreview(
                id=dataset.id or 0,
                name=dataset.name,
                columns=list(df.columns),
                rows=df.to_dict(orient="records"),
                total_rows=dataset.row_count or len(df),
                statistics=statistics,
            )

    async def delete_dataset(self, dataset_id: int) -> None:
        """Delete a dataset and its file."""
        with get_session() as session:
            dataset = session.get(Dataset, dataset_id)
            if not dataset:
                raise HTTPException(status_code=404, detail="Dataset not found")

            # Delete file
            file_path = Path(dataset.file_path)
            if file_path.exists():
                if file_path.is_dir():
                    shutil.rmtree(file_path)
                else:
                    file_path.unlink()

            # Delete record
            session.delete(dataset)
            session.commit()
