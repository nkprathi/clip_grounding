import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def ground_query(processor, model, device, image: Image.Image, query: str, threshold: float = 0.5):
    """Run CLIPSeg for a single (image, text query) pair.

    Returns:
        mask: HxW float array in [0, 1], resized to the original image size.
        bbox: (x1, y1, x2, y2) tight box around the thresholded mask, or None
              if nothing crosses the threshold.
        heatmap: HxW float array in [0, 1], the raw (unthresholded) response.
    """
    inputs = processor(text=[query], images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits  # shape (1, H', W') for a single query
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    probs = torch.sigmoid(logits)[0]  # H' x W'

    probs = F.interpolate(
        probs.unsqueeze(0).unsqueeze(0),
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    heatmap = probs.cpu().numpy()
    mask = (heatmap >= threshold).astype(np.uint8)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        bbox = None
    else:
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    return heatmap, bbox
