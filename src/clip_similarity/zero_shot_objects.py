import json
from pathlib import Path

import torch
import wandb
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MANIFEST_PATH = "/Users/prathi/work/clip_grounding/data/subset_manifest.json"
IMAGES_DIR = "/Users/prathi/work/clip_grounding/data/raw/val2017"
BACKBONES = ["openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16"]
device = "mps" if torch.backends.mps.is_available() else "cpu"

run = wandb.init(
    entity="pratheeksha-naresh-thi",
    project="clip-grounding",
    job_type="zero_shot_objects",
    config={"backbones": BACKBONES, "device": device, "dataset": "COCO_subset_cropped_objects"},
)

# --- 1. Inspect manifest format ------------------------------------------------
manifest = json.loads(Path(MANIFEST_PATH).read_text())
print("1. num images:", len(manifest))
print("1. sample object:", manifest[0]["objects"][0])

# --- 2. Flatten to one record per object crop ----------------------------------
crop_records = []
for entry in manifest:
    file_name = entry["file_name"]
    for obj in entry["objects"]:
        x, y, w, h = obj["bbox"]
        crop_records.append({
            "image_path": str(Path(IMAGES_DIR) / file_name),
            "bbox": (x, y, x + w, y + h),  # PIL crop() wants (left, top, right, bottom)
            "label": obj["label"],
        })
print("2. num images:", len(manifest), "-> num object crops:", len(crop_records))
print("2. sample crop record:", crop_records[0])

# --- 3. Crop each object from its source image ---------------------------------
crops = []
kept_records = []
for rec in crop_records:
    img = Image.open(rec["image_path"]).convert("RGB")
    x1, y1, x2, y2 = rec["bbox"]
    if x2 - x1 < 1 or y2 - y1 < 1:
        continue  # skip degenerate boxes
    crop = img.crop((x1, y1, x2, y2))
    crops.append(crop)
    kept_records.append(rec)
print("3. num crops kept after filtering degenerate boxes:", len(crops))

# --- 4. Build class vocabulary (the "N object names") --------------------------
class_names = sorted({rec["label"] for rec in kept_records})
label_to_idx = {name: i for i, name in enumerate(class_names)}
labels_idx = torch.tensor([label_to_idx[rec["label"]] for rec in kept_records])
clean_classes = [c.replace("_", " ") for c in class_names]
print("4. N classes:", len(clean_classes), clean_classes)

# --- 5-8. Loop over backbones x prompt templates, encode, score ----------------
templates = {
    "A_bare": lambda c: c,
    "B_pet_style": lambda c: f"a photo of a {c}.",
}

results = {}
for backbone in BACKBONES:
    processor = CLIPProcessor.from_pretrained(backbone)
    model = CLIPModel.from_pretrained(backbone).to(device).eval()

    # --- 5. Encode all cropped images ---
    img_embs = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(crops), batch_size):
            batch_imgs = crops[i:i + batch_size]
            inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
            out = model.get_image_features(**inputs)
            if torch.is_tensor(out):
                feat = out
            elif hasattr(out, "image_embeds"):
                feat = out.image_embeds
            else:
                feat = out.pooler_output
            img_embs.append(feat)
    img_emb = torch.cat(img_embs, dim=0)
    img_emb_n = img_emb / img_emb.norm(dim=-1, keepdim=True)
    print(f"5. [{backbone}] img_emb.shape:", tuple(img_emb.shape))

    for template_name, template_fn in templates.items():
        # --- 6. Encode text for this condition ---
        prompts = [template_fn(c) for c in clean_classes]
        with torch.no_grad():
            text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
            out = model.get_text_features(**text_inputs)
            if torch.is_tensor(out):
                txt_emb = out
            elif hasattr(out, "text_embeds"):
                txt_emb = out.text_embeds
            else:
                txt_emb = out.pooler_output
        txt_emb_n = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

        # --- 7. Cosine similarity + argmax -> top-1 accuracy ---
        sims = img_emb_n @ txt_emb_n.T
        preds = sims.argmax(dim=1).cpu()
        acc = (preds == labels_idx).float().mean().item()

        # --- 8. Record result for this (backbone, template) combination ---
        results[(backbone, template_name)] = acc
        print(f"8. [{backbone} | {template_name}] top-1 acc: {acc:.4f}")
        run.log({f"{backbone}/{template_name}_top1_acc": acc})

print("\nFinal 2x2 results:")
for (backbone, template_name), acc in results.items():
    print(f"  {backbone:35s} {template_name:12s} {acc:.4f}")

# --- 9. Per-crop top-1 records: did the best-scoring class match ground truth? --
per_crop_records = []
idx_to_label = {i: name for name, i in label_to_idx.items()}
for i, rec in enumerate(kept_records):
    predicted_idx = preds[i].item()
    per_crop_records.append({
        "image_path": rec["image_path"],
        "true_label": rec["label"],
        "predicted_label": idx_to_label[predicted_idx],
        "correct": predicted_idx == labels_idx[i].item(),
    })

num_correct = sum(r["correct"] for r in per_crop_records)
top1_acc = num_correct / len(per_crop_records)
print(f"9. top-1 acc (last backbone/template run): {top1_acc:.4f} "
      f"({num_correct}/{len(per_crop_records)})")
print("9. sample per-crop record:", per_crop_records[0])

# --- 10. Render results as a 2x2 grid: backbone (rows) x template (cols) -------
template_names = list(templates.keys())
col_width = 14
header = " " * 26 + "".join(f"{t:>{col_width}}" for t in template_names)
print("\n10. 2x2 grid (backbone x template):")
print(header)
for backbone in BACKBONES:
    row = f"{backbone:26s}"
    for template_name in template_names:
        acc = results[(backbone, template_name)]
        row += f"{acc:>{col_width}.4f}"
    print(row)

run.log({"num_crops": len(crops), "num_classes": len(clean_classes)})
run.finish()
