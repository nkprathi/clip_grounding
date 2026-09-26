import torch
from transformers import CLIPModel, CLIPProcessor
from dataset import make_loader

MODEL_ID = "openai/clip-vit-base-patch32"
device = "mps" if torch.backends.mps.is_available() else "cpu"

torch.manual_seed(42)

processor = CLIPProcessor.from_pretrained(MODEL_ID)
model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()

loader = make_loader(processor, batch_size=16, seed=42, shuffle=False, drop_last=False)

all_img_emb, all_ids = [], []

with torch.no_grad():
    for batch in loader:
        image_ids = batch.pop("image_ids")
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        img_emb = outputs.image_embeds
        all_img_emb.append(img_emb.cpu())
        all_ids.append(image_ids)

all_img_emb = torch.cat(all_img_emb, dim=0)
all_ids = torch.cat(all_ids, dim=0)

print("image embeddings:", all_img_emb.shape)   # expect torch.Size([300, 512])
print("image L2 norm     :", all_img_emb.norm(dim=-1)[:3])  # already ~1.0 (image_embeds are pre-normalized)

# Step 8 — cache the image embeddings (query-independent, computed once)
import numpy as np
np.savez("outputs/image_embeddings.npz",
          embeddings=all_img_emb.numpy().astype("float32"),
          image_ids=all_ids.numpy())
print("saved outputs/image_embeddings.npz")