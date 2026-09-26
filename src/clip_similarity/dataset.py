import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class TabletopDataset(Dataset):
    def __init__(self, manifest_path, images_dir):
        self.items = json.loads(Path(manifest_path).read_text())
        self.images_dir = Path(images_dir)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = Image.open(self.images_dir / it["file_name"]).convert("RGB")
        return {
            "image": img,
            "caption": it["caption"],
            "image_id": it["image_id"],
            "objects": it["objects"],
        }

def make_collate_fn(processor):
    def collate(batch):
        inputs = processor(
            text=[b["caption"] for b in batch],
            images=[b["image"] for b in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs["image_ids"] = torch.tensor([b["image_id"] for b in batch])
        return inputs
    return collate

def make_loader(processor, batch_size=16, seed=42, shuffle=True, drop_last=True):
    ds = TabletopDataset("data/subset_manifest.json", "data/raw/val2017")
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g,          # deterministic shuffle order
        num_workers=0,        # 0 on M2 avoids fork/PIL issues; raise later if needed
        collate_fn=make_collate_fn(processor),
        drop_last=drop_last,  # square batches for similarity demo; False when caching everything
    )