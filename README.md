# CLIP Grounding

A hands-on exploration of CLIP (Contrastive Language-Image Pre-training) for image-text grounding tasks, including zero-shot classification and image-text retrieval. This project is part of a structured self-study curriculum on Vision-Language Models (VLMs).

## Overview

This repository documents practical experiments with OpenAI's CLIP model (`openai/clip-vit-base-patch32`), covering:

- Loading and running CLIP via Hugging Face `transformers`
- Encoding images and text into a shared embedding space
- Computing cosine similarity between image-text pairs
- Zero-shot image classification
- Image-text retrieval

## Model

- **Base model:** `openai/clip-vit-base-patch32`
- **Vision encoder:** ViT-B/32
- **Framework:** Hugging Face `transformers`
- **Hardware:** Apple Silicon (M2), using PyTorch's MPS backend for GPU acceleration

## Project Structure

```
clip-grounding/
├── notebooks/       # Exploratory notebooks
├── src/             # Scripts (smoke tests, experiments, utilities)
├── data/            # Local datasets (gitignored)
├── requirements.txt # Pinned dependencies
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd clip-grounding
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv clipgrounding
source clipgrounding/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Core dependencies include:

- `torch`, `torchvision`, `torchaudio` — PyTorch with MPS backend support for Apple Silicon
- `transformers` — Hugging Face model and processor loading
- `open_clip_torch` — alternative/reference CLIP implementation
- `wandb` — experiment tracking and logging
- `pillow`, `requests` — image loading and I/O

### 4. Verify MPS availability

```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
```

This should print `True` on Apple Silicon hardware with a recent PyTorch build.

## Hugging Face

The CLIP model and processor are loaded directly from the Hugging Face Hub:

```python
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
```

No Hugging Face account is strictly required to download this public checkpoint. However, setting an `HF_TOKEN` environment variable is recommended to avoid rate limits on unauthenticated requests:

```bash
huggingface-cli login
```

or

```bash
export HF_TOKEN=<your_token>
```

Generate a token at https://huggingface.co/settings/tokens.

## Weights & Biases

Experiment tracking is handled via Weights & Biases (W&B). Runs log metrics such as similarity scores, and (in later stages) training curves for fine-tuning experiments.

### Authentication

```bash
wandb login
```

This stores an API key locally in `~/.netrc`, so authentication is only required once per machine. Generate a key at https://wandb.ai/authorize.

### Usage in scripts

```python
import wandb

run = wandb.init(
    entity="<your-entity>",
    project="clip-grounding",
    config={
        "model": "openai/clip-vit-base-patch32",
        "device": "mps",
    },
)

run.log({"cosine_similarity": cosine_sim})
run.finish()
```

Runs and logged metrics are viewable on the W&B dashboard at `https://wandb.ai/<entity>/clip-grounding`.

## Running the Smoke Test

A minimal smoke test verifies that the model loads correctly and produces a sensible similarity score for a matching image-text pair:

```bash
python3 src/smoketest.py
```

Expected output:

```
Model loaded on mps
Cosine similarity: 0.28-0.35
```

Cosine similarity scores in the 0.25-0.35 range are typical for a genuinely matching image-text pair under CLIP; the model does not tend to produce very high similarity scores even for correct matches.

## Status

This repository currently covers the CLIP implementation phase of the VLM curriculum: environment setup, model loading, and basic image-text similarity scoring. Upcoming work includes zero-shot classification and image-text retrieval.

## References

- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP paper)
- Hugging Face CLIP documentation: https://huggingface.co/docs/transformers/model_doc/clip
- Weights & Biases documentation: https://docs.wandb.ai