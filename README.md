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

## Data

Experiments in `src/clip_similarity/` use a 300-image subset of the **COCO val2017** split, filtered to cluttered "tabletop" scenes (bottles, cups, books, laptops, etc.) with at least two matching objects per image.

### 1. Download COCO val2017

```bash
mkdir -p data/raw && cd data/raw
curl -fL -C - -O http://images.cocodataset.org/zips/val2017.zip
curl -fL -C - -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -o -q val2017.zip
unzip -o -q annotations_trainval2017.zip
cd ../..
```

Only `annotations/instances_val2017.json` and `annotations/captions_val2017.json` are needed; the other annotation files (train captions/instances, person keypoints) can be deleted to save space, along with both `.zip` files once extraction succeeds:

```bash
cd data/raw
rm -f annotations/captions_train2017.json annotations/instances_train2017.json \
      annotations/person_keypoints_train2017.json annotations/person_keypoints_val2017.json
rm -f val2017.zip annotations_trainval2017.zip
cd ../..
```

Final size: ~810 MB (`val2017/` images + the two kept annotation files). `data/` is gitignored, so this is a local step, not something committed.

### 2. Build the subset manifest

```bash
python3 src/clip_similarity/build_subset.py
```

This reads the two annotation files, keeps images with ≥2 tabletop-category objects, deterministically samples 300 of them (seed 42), and writes `data/subset_manifest.json` — a list of `{image_id, file_name, caption, objects}` entries used by the `TabletopDataset` loader in `dataset.py`.

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

## Similarity-Matrix Sanity Check

```bash
python3 src/clip_similarity/sanity_check.py
```

Encodes one batch (16 images, 16 captions) from the subset, computes the 16x16 cosine similarity matrix, prints diagonal vs. off-diagonal means and image-to-text / text-to-image top-1 accuracy, and saves a heatmap to `outputs/similarity_matrix.png`.

`src/clip_similarity/encode.py` runs the same encoder over the *entire* 300-image subset and caches the resulting image embeddings to `outputs/image_embeddings.npz` (~0.6 MB), so later text queries only need one text-encoder pass plus a matrix multiply against this cache.

## Zero-Shot Classification

```bash
python3 src/clip_similarity/zero_shot_class.py    # Oxford-IIIT Pet (3669 test images, 37 breeds)
python3 src/clip_similarity/zero_shot_objects.py  # cropped COCO objects, 2 backbones x 2 prompt templates
```

Both scripts compare a bare class name (e.g. `"beagle"`) against a captioning-style template (e.g. `"a photo of a beagle, a type of pet."`) as the zero-shot text prompt, holding the images and model fixed. The templated prompt reads closer to CLIP's web-caption training data and typically scores higher top-1 accuracy. `zero_shot_objects.py` extends this to a 2x2 grid across two vision backbones (ViT-B/32, ViT-B/16) to separate the effect of prompt phrasing from model capacity.

**Key takeaway:** CLIP's zero-shot "classifier" is just cosine similarity between image and text embeddings, so prompt wording is a real, measurable lever on accuracy, not just anecdotal advice — and it can be isolated from other levers (like backbone choice) via controlled A/B comparisons.

## Status

This repository currently covers the CLIP implementation phase of the VLM curriculum: environment setup, model loading, a COCO-derived tabletop subset, image-text similarity scoring, and a cached image-embedding pipeline. Upcoming work includes zero-shot classification and image-text retrieval.

## References

- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP paper)
- Hugging Face CLIP documentation: https://huggingface.co/docs/transformers/model_doc/clip
- Weights & Biases documentation: https://docs.wandb.ai