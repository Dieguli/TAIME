# TAIME v2.0 User Guide

TAIME v2.0 is a demo-first retraining tool for moving through one workflow:

1. Load a curated dataset package.
2. Start a retraining job for the matching SUT lane.
3. Monitor progress and review metrics.
4. Download the exported retrained model artifact.

This guide focuses on using the web interface after the application is already open. It does not cover installation or server launch steps.

## Main Navigation

The left sidebar is the primary way to move through the tool.

| Page | What it is for |
| --- | --- |
| Control Center | High-level demo status across datasets, retraining jobs, and exported models. |
| Datasets | Upload curated ZIP packages, check readiness by SUT lane, preview dataset summaries, and delete uploaded packages. |
| Retraining | Create retraining runs, choose compute and hyperparameters, monitor job history, cancel active runs, and open job reports. |
| Models | Review exported artifacts, trace them back to their dataset and job, and download retrained model ZIPs. |
| API Docs | Opens the Swagger API documentation in a new browser tab. |

Use the Refresh button on each page when you want to force the UI to reload the latest server state. The Retraining page also polls job and model status automatically while it is open.

## Supported SUT Lanes

Each dataset and retraining job belongs to exactly one SUT lane. The selected dataset controls the model type used by a retraining run.

| SUT lane | UI label | Model | Required package contents |
| --- | --- | --- | --- |
| `healthcare_pc` | Healthcare - PC Risk Predictor | XGBoost | `mup_risk_model_data.csv` or `MUP_data_processed.csv`, plus `y_train_model_data.csv` |
| `infra_port` | Infrastructure - Port Operations | Darts TSMixer | `series_meta.json`, `past_covs_meta.json`, `series_*.parquet`, `past_covs_*.parquet`, a `.pt` model file, and a `.ckpt` checkpoint |
| `disinfo_fake` | Disinformation - Fake News | DistilBERT | `Fake.csv` and `True.csv` |
| `disinfo_hate` | Disinformation - Hate Speech | RoBERTa | `en_dataset.csv`, `Multitarget-CONAN.csv`, and a bundled hate-speech model ZIP or tokenizer/model assets |

All uploads must be `.zip` files. The server rejects non-ZIP uploads and ZIPs that do not match the selected SUT lane. The default maximum upload size is 1 GB unless the deployment has been configured differently.

## Control Center

The Control Center is the best first page for understanding the current demo state.

At the top, the workflow strip shows:

- Dataset packages: how many dataset packages are currently registered.
- Retraining jobs: how many runs have been created.
- Completed runs: how many runs finished successfully.
- Exported models: how many downloadable model artifacts exist.

The SUT Readiness cards show one card per SUT lane. Each card has three readiness lines:

- Dataset: whether at least one uploaded package exists for that lane.
- Latest job: the most recent retraining job and its status.
- Model: whether a downloadable artifact has been exported.

The main action button on each card changes based on state:

- Upload package: no dataset is available yet.
- Start retraining: a dataset exists but no active job or model is ready.
- Monitor job: a job is pending or running.
- Open model: an exported artifact exists.

If a model exists, the card also shows a Download button so you can retrieve the latest artifact without switching pages.

## Loading And Managing Datasets

Open Datasets from the sidebar to upload, inspect, and remove dataset packages.

### Check Readiness

The readiness cards at the top of the page show whether each SUT lane is ready for retraining.

- `ready` means at least one dataset has been uploaded for that lane.
- `needed` means the lane still needs a package before a retraining run can start.

The card text also shows the latest dataset name and row count when available.

### Upload A Dataset Package

Use the Upload Package panel.

1. Enter a Dataset name. Choose a name operators will recognize later in job and model trace views.
2. Select the SUT lane. This must match the contents of the ZIP package.
3. Add a short Description when helpful. This appears on dataset cards.
4. Choose a `.zip` package.
5. Select Upload dataset.

The Zip Contract panel on the right updates when you change the SUT lane. Use it as the checklist for the files the package must contain.

After a successful upload:

- The form clears.
- The Dataset Catalog refreshes.
- The matching readiness card changes to `ready`.
- The dataset becomes available in the Retraining page dataset selector.

If the upload fails, read the red error notice at the top of the page. Common causes are a non-ZIP file, a missing required file, a ZIP that belongs to a different SUT lane, or a package that exceeds the configured size limit.

### Preview A Dataset

In the Dataset Catalog, select Preview on a dataset card.

The Dataset Preview section shows a summary-first view:

- Total rows.
- Column names, shortened when there are many columns.
- Dataset statistics such as feature counts, label counts, text length summaries, time-series metadata, or file summary information depending on the lane.

For supported curated SUT datasets, TAIME avoids exposing raw records by showing operational summaries. If row samples are available for a generic preview, they appear inside an expandable Show sample rows section.

Select Close to hide the preview.

### Delete A Dataset

Select Delete on a dataset card to remove that dataset record and its stored package files.

Use deletion carefully:

- Deleting a dataset removes it from future retraining selectors.
- Existing jobs and models may still reference the old dataset name or ID in their trace information.
- If a lane still needs retraining, upload a replacement package before creating the next job.

## Retraining Models

Open Retraining from the sidebar to create and monitor jobs.

### Runtime Guardrail

The page header and Runtime Guardrail panel show the current runtime capabilities:

- Image: the runtime image or environment label.
- CUDA: whether CUDA is visible to the server.
- PyTorch: the detected PyTorch version.

Use Auto for normal demos. Select GPU (CUDA) only when CUDA is available and you need to demonstrate the GPU path. If CUDA is not visible and GPU is selected, TAIME blocks the job and shows a warning.

### Create A Retraining Run

Use the Create Retraining Run form.

Step 1: Dataset package

1. Open the Dataset dropdown.
2. Select the uploaded package you want to retrain on.
3. Confirm the Model type field. It is read-only and is set automatically from the dataset SUT lane.

When you select a dataset, TAIME fills the form with the fast demo preset for that lane.

Step 2: Demo preset

Use these controls to choose the run profile:

- Fast demo preset: restores the recommended demo settings for the selected SUT.
- Full dataset: disables sample limiting where applicable and trains on the full uploaded dataset.
- Compute device: Auto, GPU (CUDA), or CPU.
- Estimators/Epochs: named Estimators for Healthcare/XGBoost and Epochs for the other lanes.
- Learning rate: model optimizer learning rate.
- Batch size: training batch size for applicable model families.

Fast preset defaults:

| SUT lane | Fast demo preset |
| --- | --- |
| Healthcare | 120 estimators, full curated package |
| Port operations | 10 epochs, batch 64, scheduler enabled |
| Fake news | 3 epochs, 2,000-row demo sample |
| Hate speech | 5 epochs, binary mode, 2,000-row demo sample |

Select Start retraining when the form is ready. A new job is created with `pending` status, then moves through the worker lifecycle.

### Advanced Parameters

Open Advanced parameters when you need more control.

Port operations exposes:

- Dropout.
- Norm type: `LayerNorm`, `LayerNormNoBias`, or `TimeBatchNorm2d`.
- LR scheduler factor.
- LR scheduler patience.
- Reversible instance norm.
- Normalize before.

Fake news and hate speech expose:

- Weight decay.
- Seed.
- Sample mode: Limit samples or Full dataset.
- Max samples, enabled only when Sample mode is Limit samples.

Hate speech also exposes:

- Warmup steps.
- Classification mode: Binary or Multilabel.

For demos, start with the fast preset and adjust only the parameters you need to explain.

### Monitor Job History

The Job History table lists retraining runs. Use the segmented filter control to view:

- All.
- Active.
- Completed.
- Failed.
- Cancelled.

Each job row shows:

- Run ID.
- Dataset name and row count.
- Status badge.
- Progress bar and percent.
- Current epoch or estimator progress.
- Creation time.
- Available actions.

Statuses mean:

| Status | Meaning |
| --- | --- |
| `pending` | The job has been created and is waiting for the worker. |
| `running` | Training is in progress. |
| `completed` | Training finished and an artifact should be available. |
| `failed` | Training stopped with an error. Open the report to see the message. |
| `cancelled` | The job was cancelled before completion. |

Select Refresh to reload datasets, jobs, models, and runtime information. The page also refreshes jobs and models automatically every few seconds.

### Open A Job Report

Select Report on a job row.

The Job Report shows:

- Status and progress.
- Dataset name and row count.
- Start and completion timestamps.
- Artifact name and version when exported.
- Error message if the job failed.
- Headline metrics such as accuracy, F1, precision, recall, RMSE, MAE, MAPE, or validation loss depending on what the job reports.

Open Technical config and metrics to inspect the exact job configuration and metrics JSON. This is useful when comparing runs or documenting the settings used for a demo.

If a completed job has an exported artifact, the report includes Download model.

### Cancel A Run

Select Cancel on a job row when the status is `pending` or `running`.

Cancelled jobs do not export retrained model artifacts. If you need a model afterward, create a new retraining run.

## Downloading Retrained Models

TAIME exports a model artifact after a job completes successfully. Every exported artifact is downloaded as a ZIP or binary artifact returned by the model download endpoint.

You can download from three places:

- Control Center: use Download on a SUT card when a model exists.
- Retraining: use Download in the completed job row or Download model in the job report.
- Models: use Download on the latest model cards or in the Artifact Catalog.

### Use The Models Page

Open Models from the sidebar for the clearest artifact management view.

The top cards show the latest exported artifact for each SUT lane. A ready card includes:

- Artifact name.
- Job ID.
- Dataset name.
- Headline metrics.
- Job status.
- Download button.

The Artifact Catalog lists all exported model versions. Each row shows:

- Artifact name and version.
- SUT lane.
- Trace back to job and dataset.
- Metrics.
- File size.
- Creation time.
- Download action.

Select Download to save the model artifact through `/api/v1/models/{id}/download`.

### Understand The Artifact

The downloaded file is the retrained model package produced by the completed job.

Depending on the SUT lane, it may contain:

- Model weights and configuration.
- Tokenizer/config assets for transformer models.
- Darts `.pt` and `.ckpt` files for the Port operations lane.
- A training report or metadata files when produced by the training backend.

Use the artifact name, version, job ID, dataset name, and metrics in the Models page to confirm you downloaded the intended retrained model.

## Recommended End-To-End Smoke Test

Use this flow when validating the UI for a demo:

1. Open Control Center and note which SUT lanes are missing datasets or models.
2. Go to Datasets.
3. Upload a valid ZIP for one SUT lane.
4. Confirm the readiness card changes to `ready`.
5. Select Preview and review the summary.
6. Go to Retraining.
7. Select the uploaded dataset.
8. Keep the fast demo preset.
9. Leave Compute device as Auto unless testing GPU specifically.
10. Select Start retraining.
11. Watch the job move from `pending` to `running` to `completed`.
12. Open Report and review metrics.
13. Download the model from the report.
14. Go to Models.
15. Confirm the artifact appears in the latest card and Artifact Catalog.
16. Download it again from Models to verify the artifact route.

## Troubleshooting

Upload rejected as non-ZIP:

- Confirm the file extension is `.zip`.
- Repackage the dataset as a ZIP before uploading.

Upload rejected for missing files:

- Check that the selected SUT lane matches the package.
- Compare the ZIP contents with the Zip Contract panel.
- For nested folders, the required files can be inside the ZIP, but they must be present and readable.

Dataset appears but no job can start:

- Refresh the Retraining page.
- Confirm the dataset has a recognized SUT lane.
- Re-upload if the package was deleted.

GPU option is blocked:

- CUDA is not visible to the runtime.
- Switch Compute device to Auto or CPU, or use a runtime configured for CUDA.

Job failed:

- Open Report and read the error notice.
- Check that the dataset package belongs to the selected SUT lane.
- Use the fast demo preset for a known-good baseline.
- Reduce sample size for transformer lanes if the environment is memory constrained.

Job completed but no download appears:

- Select Refresh on Retraining or Models.
- Check the job report for an artifact name.
- If the job says completed but the artifact is missing, use Models to confirm whether the export record exists.

Downloaded the wrong model:

- Use the Models page rather than only the latest card.
- Match the artifact row by SUT lane, job ID, dataset name, creation time, and metrics before selecting Download.

## Operator Notes

- Do not upload raw or sensitive datasets unless they are approved for the demo environment.
- Do not rely on the browser preview for full data inspection. TAIME intentionally shows summaries for curated SUT packages.
- Use clear dataset names, because they become the easiest way to trace models back to their source package.
- Keep one known-good uploaded package per SUT lane before a live demo.
- Use fast presets for repeatable demonstrations and full-dataset runs for deeper validation.
