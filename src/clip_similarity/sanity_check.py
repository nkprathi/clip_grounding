"""
Similarity-matrix sanity check.

Encodes one batch of (image, caption) pairs, computes the cosine
similarity matrix, prints basic diagnostics, and saves a heatmap to
outputs/similarity_matrix.png.

Run from the project root:
    python3 src/clip_similarity/sanity_check.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import CLIPModel, CLIPProcessor

from dataset import make_loader
from similarity import similarity

MODEL_ID = "openai/clip-vit-base-patch32"
device = "mps" if torch.backends.mps.is_available() else "cpu"

torch.manual_seed(42)

processor = CLIPProcessor.from_pretrained(MODEL_ID)
model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()

loader = make_loader(processor, batch_size=16, seed=42)  # shuffle=True, drop_last=True
batch = next(iter(loader))
image_ids = batch.pop("image_ids")
batch = {k: v.to(device) for k, v in batch.items()}

with torch.no_grad():
    outputs = model(**batch)
    img_emb = outputs.image_embeds
    txt_emb = outputs.text_embeds

cos, logits = similarity(img_emb, txt_emb, model.logit_scale)

B = cos.shape[0]
diag = cos.diagonal()
off = cos[~torch.eye(B, dtype=bool, device=cos.device)]

print("matrix shape      :", cos.shape)            # (16, 16)
print("mean diagonal     :", diag.mean().item())   # ~0.28-0.33 typical
print("mean off-diagonal :", off.mean().item())    # ~0.10-0.18 typical
print("i->t top-1 acc     :", (cos.argmax(dim=1) == torch.arange(B, device=cos.device)).float().mean().item())
print("t->i top-1 acc     :", (cos.argmax(dim=0) == torch.arange(B, device=cos.device)).float().mean().item())

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cos.cpu().numpy(), cmap="viridis", vmin=-1, vmax=1)
ax.set_title("Image-text cosine similarity (one batch)")
ax.set_xlabel("caption index")
ax.set_ylabel("image index")
fig.colorbar(im, ax=ax, label="cosine similarity")
fig.tight_layout()
fig.savefig("outputs/similarity_matrix.png", dpi=150)
print("saved outputs/similarity_matrix.png")
