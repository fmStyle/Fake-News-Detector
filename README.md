# Fake News Detection — University Project

This is a small university project that explores fake-news detection using two approaches: a Convolutional Neural Network (CNN) on word embeddings and a Transformer-based classifier (Hugging Face). It was created as a learning exercise and portfolio piece — not intended for production use.

## Overview
- Goal: classify news/text as fake or real using language signals.
- Approaches: a custom CNN model and a fine-tuned Transformer model.
- Scope: research/educational — demonstrates data preprocessing, embedding construction, model training, evaluation, and visualization.

## Key Files
- `main.py` — high-level orchestration (preprocessing, embedding, training/eval with Transformers).
- `preprocesamiento.py` — text cleaning and tokenization utilities.
- `matriz_embedding.py` — builds embedding matrix (uses GloVe / tokenizers).
- `features.py` — dataset preparation, tokenization and train/test split helpers.
- `red.py` — CNN model definition.
- `red_con_gru.py` — alternative GRU-based model.
- `training.py` / `testing.py` — training and evaluation loops for custom models.
- `transformer/transformer.py` — Transformer fine-tuning script using Hugging Face.
- `transformer/dataset.hf/` — cached HF dataset files (train/test).
- `SherLockFakenewsNetOriginal.csv` — original dataset used in the project.
- `requirements.txt` — Python dependencies (use this to set up the environment).

## Pretrained embeddings
- Place `glove.twitter.27B.100d.txt` in the project root (same folder as `main.py`).
- Download it from: https://nlp.stanford.edu/projects/glove/
- `matriz_embedding.py` expects this file to build the embedding matrix used by the CNN experiments.

## Quickstart
1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Typical commands:
- Run preprocessing / full pipeline (may be wired in `main.py`):

```bash
python main.py
```

- Train or evaluate the Transformer model (see `transformer/transformer.py` for details):

```bash
python transformer/transformer.py
```

Adjust commands as needed — scripts have small entry points rather than a polished CLI.

<img width="963" height="786" alt="Captura de pantalla 2026-02-03 022336" src="https://github.com/user-attachments/assets/5a7a6cde-bcb4-46e3-b6ae-f0b5749f32f5" />



