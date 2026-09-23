# TAIME v2.0 — Trustworthy AI Models Engine

A demonstration-focused platform for **retraining System-Under-Test (SUT) models**,
built for the [THEMIS 5.0](https://www.themis-trust.eu/) project. Upload a dataset,
configure retraining hyperparameters, launch a background training job, watch its
progress, and download the resulting model artifact — all from one web app.

The frontend (React + Vite) and backend (FastAPI) run together in **a single
container**. State is stored in an embedded **SQLite** database and the local
filesystem. No external database, no managed services, no Kubernetes.

---

## ⚠️ Models and data are proprietary — bring your own

**This repository contains only the platform's source code.** The trained SUT
models and the datasets they are retrained on are **not distributed here**. To use
TAIME you supply your own:

- **Datasets** are uploaded through the web UI (drag & drop a `.zip`) and are
  persisted on the host via a bind mount (`./data`).
- **Base model bundles** that a SUT needs are placed in a host folder
  (`./models_datasets`) or shipped inside the uploaded dataset `.zip`.

See [Bring your own models & data](#bring-your-own-models--data) for the exact
folder locations and the per-SUT file contract.

---

## Supported SUTs

| SUT type        | Domain                        | Model                          | Framework      |
|-----------------|-------------------------------|--------------------------------|----------------|
| `healthcare_pc` | Healthcare                    | Pancreatic Cancer Risk Predictor | XGBoost      |
| `infra_port`    | Critical Infrastructure       | Port Operations forecaster     | Darts TSMixer  |
| `disinfo_fake`  | Disinformation                | Fake News Detector             | DistilBERT     |
| `disinfo_hate`  | Disinformation                | Hate Speech Detector           | RoBERTa        |

Every completed job produces a single **`.zip` artifact** (model weights +
config/tokenizer, or the Darts `.pt` + `.ckpt` pair) downloadable from the Models
page or the API.

---

## Requirements

- **Docker Engine + Docker Compose v2** (`docker compose …`) — the recommended path.
- **NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)** — only for the GPU image.
- **RAM**: 8 GB minimum (16 GB recommended). **Disk**: 20 GB minimum (50 GB recommended).
- Network access **during the image build** (to download Python/Node packages and,
  by default, the DistilBERT base model).
- For the host (non-Docker) path instead: **Python 3.10+** and **Node.js 18+**.

---

## Quick start (Docker — recommended)

```bash
# 1. Clone
git clone https://github.com/Dieguli/TAIME.git
cd TAIME

# 2. (Optional) tune configuration
cp .env.example .env        # edit if you need to

# 3a. CPU (default)
docker compose up --build
#  -> http://localhost:8000        the web app
#  -> http://localhost:8000/docs   Swagger UI  (API under /api/v1)

# 3b. GPU (requires NVIDIA driver + NVIDIA Container Toolkit on the host)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The build compiles the React frontend and bakes the **DistilBERT-cased** fake-news
base model into the image (so fake-news retraining works fully offline). To skip
baking the model, build with `--build-arg FETCH_MODELS=false`.

Stop with `Ctrl+C`; remove the container with `docker compose down`. Your data in
`./data` and `./models_datasets` persists across restarts.

---

## Bring your own models & data

### Host folders (bind-mounted into the container)

| Host path              | Container path            | Holds                                                                 |
|------------------------|---------------------------|-----------------------------------------------------------------------|
| `./data`               | `/data`                   | SQLite DB (`taime.db`), uploaded datasets, trained artifacts, HF assets, logs. Created/populated automatically. |
| `./data/datasets`      | `/data/datasets`          | Extracted uploaded datasets (written by the app when you upload).     |
| `./data/models`        | `/data/models`            | Trained model artifacts (written by the app when a job completes).    |
| `./data/hf`            | `/data/hf`                | Optional pre-fetched Hugging Face base models (advanced / offline).   |
| `./models_datasets`    | `/app/models_datasets`    | **You place** the hate-speech base model bundle here. See below.      |

You normally only ever put files into **`./models_datasets`** by hand — everything
under `./data` is managed by the application.

### Per-SUT requirements

| SUT             | Base model you must provide                                              | Where it goes                                                              | Dataset `.zip` must contain                                                                                     |
|-----------------|-------------------------------------------------------------------------|---------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| `healthcare_pc` | None — trains from scratch                                               | —                                                                         | `mup_risk_model_data.csv` (or `MUP_data_processed.csv`) **and** `y_train_model_data.csv` (fixed 79-column schema) |
| `infra_port`    | Seed TSMixer model `*.pt` (+ optional `*.ckpt`)                          | Inside the uploaded dataset `.zip` (or `data/models`)                      | the seed `*.pt`, `series_meta.json`, `past_covs_meta.json`, `series_*.parquet`, `past_covs_*.parquet`          |
| `disinfo_fake`  | DistilBERT-cased — **fetched automatically** (see below)                | `data/hf/distilbert-base-cased` (baked into the image by default)         | `Fake.csv`, `True.csv`                                                                                          |
| `disinfo_hate`  | RoBERTa bundle `FakeNews_HateSpeech_model.zip`                           | `./models_datasets/` (or shipped inside the dataset `.zip`)               | `en_dataset.csv`, `Multitarget-CONAN.csv` (+ the model bundle if not placed in `./models_datasets`)             |

> The exact filename validation for each SUT's upload is surfaced in the UI's
> **Datasets** page ("required files" per SUT lane).

### Placing the hate-speech (RoBERTa) bundle

```bash
mkdir -p models_datasets
cp /path/to/FakeNews_HateSpeech_model.zip models_datasets/
# Different name/location? point the env var at it:
#   HATE_SPEECH_MODEL_ZIP=models_datasets/<your-bundle>.zip
```

The bundle's expected internal layout is documented in
[`models_datasets/README.md`](models_datasets/README.md).

### The DistilBERT (fake-news) base model

By default it is **baked into the Docker image** at build time (`FETCH_MODELS=true`),
so fake-news retraining runs offline with no extra setup. If you build with
`--build-arg FETCH_MODELS=false`, provide it another way before running a fake-news
job:

- pre-fetch it on the host: `python scripts/fetch_models.py` (writes to `data/hf/`), or
- allow a runtime download: set `TAIME_ALLOW_HF_DOWNLOAD=1` in `.env`.

---

## Train a model — step by step

The flow is the same in the web UI or via the REST API:
**upload a dataset package → launch a retraining job → wait for it to finish →
download the artifact.** Datasets are always *uploaded* (never hand-placed under
`./data`); the only files you put on disk yourself are base-model bundles
(see [Bring your own models & data](#bring-your-own-models--data)).

> **First run:** `docker compose up --build` downloads the ML wheels and bakes the
> DistilBERT base — expect **~10–15 min** the first time (then it's cached). The app
> starts with an empty catalog; you populate it by uploading.

### A. In the web UI (`http://localhost:8000`)

1. **Control Center** — landing page; shows per-SUT readiness (dataset / job / model)
   and recent activity.
2. **Datasets** → drag & drop your SUT's `.zip`. It is validated against that SUT's
   required files, extracted, and previewed (columns, statistics, sample rows).
3. **Retraining** → select the dataset, pick a **fast-demo preset** or set
   hyperparameters (reference below), choose **CPU / GPU**, and **Launch**.
4. The job runs in the background and the page auto-refreshes status/progress; open a
   job to see its metrics and report.
5. **Models** → **download** the trained artifact `.zip` (traceable to its dataset + job).

### B. Via the REST API (same flow, copy-paste)

```bash
# 1) Upload a dataset package (multipart form).           -> {"id": 1, ...}
curl -s -F "file=@healthcare_pc.zip" -F "name=my-run" -F "sut_type=healthcare_pc" \
  http://localhost:8000/api/v1/datasets

# 2) Launch a retraining job (JSON body).                 -> {"id": 1, "status": "pending", ...}
curl -s -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": 1, "sut_type": "healthcare_pc", "config": {"n_estimators": 200, "device": "cpu"}}'

# 3) Poll until status == "completed" (progress 0 -> 100; "metrics" fills in).
curl -s http://localhost:8000/api/v1/jobs/1

# 4) Find the model for your job (its "job_id"), then download the artifact .zip.
curl -s http://localhost:8000/api/v1/models
curl -s http://localhost:8000/api/v1/models/1/download -o healthcare_model.zip
```

Send an empty `config` to accept the trainer defaults. Invalid values return
**HTTP 422** naming the offending field.

### Retraining parameters (the `config` object)

Only keys you send are applied; the rest fall back to the defaults shown.

| SUT | Framework | Key `config` fields *(default)* | Allowed / notes |
|-----|-----------|--------------------------------|-----------------|
| `healthcare_pc` | XGBoost | `n_estimators` *(120)*, `max_depth` *(6)*, `learning_rate` *(0.1)*, `test_split` *(0.2)*, `random_state` *(42)* | trains from scratch; runs in seconds |
| `disinfo_fake` | DistilBERT | `epochs` *(3)*, `batch_size` *(16)*, `learning_rate` *(5e-6)*, `weight_decay` *(0.1)*, `warmup_steps` *(0)*, `seed` *(42)*, `max_train_samples`, `max_length` *(128)* | `max_train_samples` caps rows for a fast demo |
| `disinfo_hate` | RoBERTa | `epochs` *(5)*, `batch_size` *(16)*, `learning_rate` *(5e-5)*, `weight_decay` *(0.01)*, `warmup_steps` *(500)*, `seed`, `max_train_samples`, `max_length`, `classification_type` *(binary)* | `classification_type` ∈ {`binary`, `multilabel`} |
| `infra_port` | Darts TSMixer | `epochs`/`n_epochs`, `batch_size` *(64)*, `dropout` *(0.2)*, `learning_rate` *(1e-3)*, `lr_scheduler_factor` *(0.5)*, `lr_scheduler_patience` *(5)*, `norm_type` *(LayerNorm)*, `use_reversible_instance_norm` *(false)*, `max_series` *(all)*, `backtest_max_series` *(20)* | `batch_size` ∈ {64,128,256}; `dropout` ∈ [0.2,0.5]; `learning_rate` ∈ [1e-5,1e-3]; `norm_type` ∈ {LayerNorm, LayerNormNoBias, TimeBatchNorm2d}; model architecture is recovered from the seed `.pt`, not tuned; trains logits with `BCEWithLogitsLoss`; trains on every series unless `max_series` caps it; metrics are a backtest on up to `backtest_max_series` held-out series |

Every SUT also accepts `device` (`auto` \| `cpu` \| `cuda`).

### What a completed job produces

The job's **`metrics`** (accuracy / precision / recall / F1, confusion matrix,
per-class report, sample counts) are shown in the UI and in `GET /api/v1/jobs/{id}`.
The downloadable artifact `.zip` contains:

| SUT | Artifact contents |
|-----|-------------------|
| `healthcare_pc` | `healthcare_pc_model_<job>.pkl` (XGBoost model) |
| `disinfo_fake` / `disinfo_hate` | model dir: `config.json`, `model.safetensors`, tokenizer files |
| `infra_port` | `port_model_<job>.pt` + `port_model_<job>.pt.ckpt` (Darts TSMixer) |

---

## Configuration

Copy `.env.example` to `.env` and adjust. The variables the application reads:

| Variable                 | Default                        | Purpose                                                        |
|--------------------------|--------------------------------|----------------------------------------------------------------|
| `DATA_DIR`               | `/data` (Docker) / `data`      | Root for DB, datasets, models, HF assets, logs.                |
| `DATABASE_URL`           | `sqlite:///data/taime.db`      | SQLite database URL.                                            |
| `MAX_WORKERS`            | `1`                            | Size of the background training pool.                          |
| `MAX_UPLOAD_BYTES`       | `1073741824` (1 GiB)           | Maximum dataset upload size.                                   |
| `HATE_SPEECH_MODEL_ZIP`  | `models_datasets/FakeNews_HateSpeech_model.zip` | Path to the hate-speech base bundle.          |
| `TAIME_ALLOW_HF_DOWNLOAD`| unset                          | Allow runtime download of base HF models when absent locally.  |
| `FAKE_NEWS_MODEL_DIR`    | `data/hf/distilbert-base-cased`| DistilBERT base directory (baked into the image by default).   |
| `CUDA_VISIBLE_DEVICES`   | unset                          | Restrict visible GPUs (GPU image).                             |
| `HOST` / `PORT`          | `0.0.0.0` / `8000`             | Bind address for the host (non-Docker) run.                    |

---

## Alternative: host install (no Docker)

Requires **Python 3.10+** and **Node.js 18+**.

```bash
./setup.sh     # create venv, install backend + frontend deps, scaffold ./data and .env
./start.sh     # build the frontend and run the server on :8000
# open http://localhost:8000
```

For the fake-news SUT on the host, fetch the base model once:
`python scripts/fetch_models.py`.

---

## API

Interactive docs once running:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

| Method | Endpoint                          | Description                          |
|--------|-----------------------------------|--------------------------------------|
| GET    | `/api/v1/datasets`                | List datasets                        |
| POST   | `/api/v1/datasets`                | Upload a dataset (`.zip`, multipart) |
| GET    | `/api/v1/datasets/{id}`           | Get a dataset                        |
| GET    | `/api/v1/datasets/{id}/preview`   | Preview (first rows + statistics)    |
| DELETE | `/api/v1/datasets/{id}`           | Delete a dataset                     |
| GET    | `/api/v1/jobs`                    | List training jobs                   |
| POST   | `/api/v1/jobs`                    | Create + schedule a training job     |
| GET    | `/api/v1/jobs/{id}`               | Get job status                       |
| POST   | `/api/v1/jobs/{id}/cancel`        | Cancel a running job                 |
| GET    | `/api/v1/models`                  | List trained models                  |
| GET    | `/api/v1/models/{id}`             | Get a model version                  |
| GET    | `/api/v1/models/{id}/download`    | Download a model artifact (`.zip`)   |
| GET    | `/api/v1/runtime`                 | Runtime capabilities (CUDA/torch)    |
| GET    | `/health`                         | Health check                         |

---

## Project structure

```
TAIME/
├── apps/
│   ├── api/                 # FastAPI backend (installable package `taime-api`)
│   │   ├── src/taime_api/   # api/, services/, workers/, training/, db/, ...
│   │   ├── tests/           # pytest suite
│   │   └── static/          # Vite build output (generated; gitignored)
│   └── web/                 # React frontend (Vite + TypeScript)
├── docker/
│   ├── Dockerfile           # CPU image
│   └── Dockerfile.gpu       # CUDA image
├── scripts/                 # ci.sh, lint.sh, fmt.sh, fetch_models.py
├── data/                    # runtime state (bind-mounted; gitignored)
├── models_datasets/         # your base model bundles (bind-mounted; gitignored)
├── docker-compose.yml       # CPU (primary)
├── docker-compose.gpu.yml   # GPU override
├── setup.sh / start.sh / dev.sh
└── .env.example
```

---

## Development

```bash
./dev.sh                     # backend (uvicorn --reload, :8000) + Vite dev (:5173)
bash scripts/ci.sh           # lint + backend tests + frontend typecheck
bash scripts/fmt.sh          # format (ruff + prettier)
```

Backend tests only: `cd apps/api && pytest tests/`.
Frontend checks: `cd apps/web && npm run lint && npm run typecheck`.

---

## Troubleshooting

- **GPU not detected** — confirm the NVIDIA driver and NVIDIA Container Toolkit are
  installed on the host and that `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
  works, then use the GPU compose command. Adjust `TORCH_INDEX_URL`
  (`--build-arg TORCH_INDEX_URL=…`) if your CUDA version differs from 12.4.
- **Fake-news job fails to find the base model** — you built with
  `FETCH_MODELS=false`. Set `TAIME_ALLOW_HF_DOWNLOAD=1` or run
  `python scripts/fetch_models.py`.
- **Hate-speech job errors on missing assets** — place
  `FakeNews_HateSpeech_model.zip` in `./models_datasets/` (or set
  `HATE_SPEECH_MODEL_ZIP`), or include the bundle in the dataset `.zip`.
- **Port 8000 already in use** — map another host port, e.g. edit the compose
  `ports:` to `"8080:8000"`.
- **Upload rejected** — datasets must be `.zip` and match the SUT's required files
  (shown in the Datasets page); large files may need a higher `MAX_UPLOAD_BYTES`.

---

## License & funding

Licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

THEMIS 5.0 has received funding from the European Union's Horizon Europe research
and innovation programme under grant agreement **No 101135049**. Views and opinions
expressed are those of the author(s) only and do not necessarily reflect those of
the European Union.
