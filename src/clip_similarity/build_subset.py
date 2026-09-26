import json, random
from pathlib import Path

SEED = 42
N_IMAGES = 300
TABLETOP = {"bottle","wine glass","cup","fork","knife","spoon","bowl",
            "banana","apple","orange","book","scissors","cell phone",
            "remote","keyboard","mouse","laptop","vase","teddy bear","toothbrush"}

root = Path("data/raw/annotations")
inst = json.loads((root / "instances_val2017.json").read_text())
caps = json.loads((root / "captions_val2017.json").read_text())

# category id -> name, restricted to our set
cat_name = {c["id"]: c["name"] for c in inst["categories"] if c["name"] in TABLETOP}

# image_id -> list of (category_name, bbox)
boxes = {}
for a in inst["annotations"]:
    if a["category_id"] in cat_name:
        boxes.setdefault(a["image_id"], []).append(
            {"label": cat_name[a["category_id"]], "bbox": a["bbox"]}
        )

# keep images with >=2 tabletop objects -> genuinely cluttered scenes
candidates = sorted([i for i, b in boxes.items() if len(b) >= 2])   # sorted = deterministic

# image_id -> first caption, by ascending annotation id (deterministic tie-break)
cap_by_img = {}
for a in sorted(caps["annotations"], key=lambda x: x["id"]):
    cap_by_img.setdefault(a["image_id"], a["caption"].strip())

candidates = [i for i in candidates if i in cap_by_img]

rng = random.Random(SEED)
chosen = sorted(rng.sample(candidates, N_IMAGES))

file_name = {im["id"]: im["file_name"] for im in inst["images"]}
manifest = [{
    "image_id": i,
    "file_name": file_name[i],
    "caption": cap_by_img[i],
    "objects": boxes[i],
} for i in chosen]

Path("data/subset_manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"wrote {len(manifest)} entries")