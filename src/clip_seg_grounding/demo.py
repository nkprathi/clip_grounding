"""Zero-shot phrase grounding demo: CLIPSeg text query -> mask/box overlay.

For a handful of images, run several free-form text queries (e.g. "the red mug",
"the bowl on the left") through CLIPSeg, save a "query -> highlighted object"
image per query, and log everything to Weights & Biases so all queries for the
same source image can be compared side by side.
"""
import json
import re
from pathlib import Path

import wandb
from PIL import Image

from clip_seg_grounding.grounding import ground_query
from clip_seg_grounding.model import load_clipseg
from clip_seg_grounding.visualize import make_highlighted_image

ROOT = Path("/Users/prathi/work/clip_grounding")
MANIFEST_PATH = ROOT / "data/subset_manifest.json"
IMAGES_DIR = ROOT / "data/raw/val2017"
OUTPUT_DIR = ROOT / "outputs/grounding"
THRESHOLD = 0.5
NUM_IMAGES = 3


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def build_image_queries():
    """Pick a few images from the subset manifest and derive queries from their
    labeled objects, plus a couple of generic spatial phrases per image."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    image_queries = []
    for entry in manifest[:NUM_IMAGES]:
        file_name = entry["file_name"]
        labels = sorted({obj["label"].replace("_", " ") for obj in entry["objects"]})
        queries = [f"the {label}" for label in labels[:3]]
        queries += ["the object on the left", "the object in the center"]
        image_queries.append((file_name, queries))
    return image_queries


def main():
    device = None
    processor, model, device = load_clipseg()

    run = wandb.init(
        entity="pratheeksha-naresh-thi",
        project="clip-grounding",
        job_type="clipseg_phrase_grounding",
        config={"model": "CIDAS/clipseg-rd64-refined", "device": device, "threshold": THRESHOLD},
    )

    table = wandb.Table(columns=["image_id", "query", "input_image", "result_image"])
    image_queries = build_image_queries()

    for file_name, queries in image_queries:
        image_path = IMAGES_DIR / file_name
        image = Image.open(image_path).convert("RGB")
        stem = Path(file_name).stem
        out_dir = OUTPUT_DIR / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        per_image_results = []
        for query in queries:
            heatmap, bbox = ground_query(processor, model, device, image, query, threshold=THRESHOLD)
            highlighted = make_highlighted_image(image, heatmap, bbox, query)

            out_path = out_dir / f"{slugify(query)}.png"
            highlighted.save(out_path)
            print(f"[{file_name}] '{query}' -> {out_path} (bbox={bbox})")

            table.add_data(stem, query, wandb.Image(str(image_path)), wandb.Image(str(out_path)))
            per_image_results.append(wandb.Image(str(out_path), caption=query))

        # group all query results for this image together in one media panel
        run.log({f"grounding/{stem}": per_image_results})

    run.log({"query_grounding_results": table})
    run.finish()


if __name__ == "__main__":
    main()
