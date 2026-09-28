import torch
import wandb
from torchvision.datasets import OxfordIIITPet
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"
device = "mps" if torch.backends.mps.is_available() else "cpu"

run = wandb.init(
    entity="pratheeksha-naresh-thi",
    project="clip-grounding",
    job_type="zero_shot_class",
    config={"model": MODEL_ID, "device": device, "dataset": "OxfordIIITPet"},
)

processor = CLIPProcessor.from_pretrained(MODEL_ID)
model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()

# --- 1. Load images + encode, confirm shape ---------------------------------
dataset = OxfordIIITPet(root="data", split="test", download=True)
images = [dataset[i][0].convert("RGB") for i in range(len(dataset))]
labels_idx = [dataset[i][1] for i in range(len(dataset))]
raw_classes = dataset.classes  # e.g. ['Abyssinian', 'american_bulldog', ...]

img_embs = []
batch_size = 64
with torch.no_grad():
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i + batch_size]
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

print("dataset.classes:", raw_classes)
print("1. img_emb.shape:", tuple(img_emb.shape))
assert img_emb.shape == (3669, 512), f"expected (3669, 512), got {tuple(img_emb.shape)}"

# --- 2. Clean labels (underscore -> space) -----------------------------------
clean_classes = [c.replace("_", " ") for c in raw_classes]
print("2. sample cleaned labels:", clean_classes[:5])

# --- 3. Condition A: bare class names -----------------------------------------
with torch.no_grad():
    text_inputs_a = processor(text=clean_classes, return_tensors="pt", padding=True).to(device)
    out_a = model.get_text_features(**text_inputs_a)
    if torch.is_tensor(out_a):
        txt_emb_a = out_a
    elif hasattr(out_a, "text_embeds"):
        txt_emb_a = out_a.text_embeds
    else:
        txt_emb_a = out_a.pooler_output

img_emb_n = img_emb / img_emb.norm(dim=-1, keepdim=True)
txt_emb_a_n = txt_emb_a / txt_emb_a.norm(dim=-1, keepdim=True)
sims_a = img_emb_n @ txt_emb_a_n.T
preds_a = sims_a.argmax(dim=1).cpu()
labels_idx_t = torch.tensor(labels_idx)
acc_a = (preds_a == labels_idx_t).float().mean().item()
print("3. Condition A top-1 acc:", acc_a)

# --- 4. Condition B: a photo of a {c}, pet-style prompt template --------------------------------
prompts_b = [f"a photo of a {c}, a type of pet." for c in clean_classes]
with torch.no_grad():
    text_inputs_b = processor(text=prompts_b, return_tensors="pt", padding=True).to(device)
    out_b = model.get_text_features(**text_inputs_b)
    if torch.is_tensor(out_b):
        txt_emb_b = out_b
    elif hasattr(out_b, "text_embeds"):
        txt_emb_b = out_b.text_embeds
    else:
        txt_emb_b = out_b.pooler_output

txt_emb_b_n = txt_emb_b / txt_emb_b.norm(dim=-1, keepdim=True)
sims_b = img_emb_n @ txt_emb_b_n.T
preds_b = sims_b.argmax(dim=1).cpu()
acc_b = (preds_b == labels_idx_t).float().mean().item()
print("4. Condition B top-1 acc:", acc_b)

run.log({
    "condition_A_bare_top1_acc": acc_a,
    "condition_B_template_top1_acc": acc_b,
    "num_images": img_emb.shape[0],
    "num_classes": len(clean_classes),
})
run.finish()
