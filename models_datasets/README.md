# `models_datasets/` — bring-your-own model bundles

This folder is **bind-mounted into the container** at `/app/models_datasets`
(see `docker-compose.yml`). It is intentionally empty in the public repository —
the model bundles it holds are **proprietary and are not distributed here**.

## What goes here

The **hate-speech (RoBERTa) SUT** loads its base model from a bundle placed in
this folder. By default the trainer looks for:

```
models_datasets/FakeNews_HateSpeech_model.zip
```

Override the path with the `HATE_SPEECH_MODEL_ZIP` environment variable if your
bundle has a different name or location.

### Expected bundle layout

The `.zip` must contain the RoBERTa assets the trainer hydrates:

```
models/
├── ROBERTA_hate_speech/            # binary classifier weights + config
├── ROBERTA_hate_speech_multilabel/ # multilabel classifier weights + config
└── tokenizer/                      # tokenizer (incl. vocab.json, merges.txt)
```

Alternatively, you may ship this same bundle **inside the uploaded dataset
`.zip`** for the hate-speech SUT — see the main `README.md` for the per-SUT
dataset contract.

## The other SUTs

- **Fake news (DistilBERT)** — base model is fetched automatically (baked into
  the Docker image, or via `scripts/fetch_models.py` / `TAIME_ALLOW_HF_DOWNLOAD=1`).
  Nothing to place here.
- **Healthcare (XGBoost)** — trains from scratch; no base model needed.
- **Port operations (Darts TSMixer)** — the seed model travels inside the
  uploaded dataset `.zip`.

Files you add to this folder are git-ignored (except this README and `.gitkeep`),
so your proprietary bundles will never be accidentally committed.
