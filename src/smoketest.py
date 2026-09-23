from transformers import CLIPModel, CLIPProcessor
import torch
from PIL import Image
import requests

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model.eval()  # inference mode
print(f"Model loaded on {device}")

# any test image — swap in a local path if you prefer
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

text = "a photo of two cats"

inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)

with torch.no_grad():
    outputs = model(**inputs)
    image_embeds = outputs.image_embeds   # shape: [1, 512]
    text_embeds = outputs.text_embeds     # shape: [1, 512]

# normalize before computing cosine similarity
image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

cosine_sim = (image_embeds @ text_embeds.T).item()
print(f"Cosine similarity: {cosine_sim:.4f}")