import torch
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

DEFAULT_CHECKPOINT = "CIDAS/clipseg-rd64-refined"


def get_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_clipseg(checkpoint=DEFAULT_CHECKPOINT, device=None):
    device = device or get_device()
    processor = CLIPSegProcessor.from_pretrained(checkpoint)
    model = CLIPSegForImageSegmentation.from_pretrained(checkpoint).to(device).eval()
    return processor, model, device
