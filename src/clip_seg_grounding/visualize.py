import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm


def make_highlighted_image(image: Image.Image, heatmap: np.ndarray, bbox, query: str, alpha: float = 0.45):
    """Overlay a heatmap + bbox on the image and caption it with the query."""
    base = image.convert("RGB")

    colored = cm.get_cmap("jet")(heatmap)[:, :, :3]  # HxWx3 in [0,1]
    colored = (colored * 255).astype(np.uint8)
    heat_img = Image.fromarray(colored).resize(base.size)

    overlay = Image.blend(base, heat_img, alpha=alpha)

    draw = ImageDraw.Draw(overlay)
    if bbox is not None:
        draw.rectangle(bbox, outline=(255, 255, 255), width=3)

    caption_h = 30
    canvas = Image.new("RGB", (overlay.width, overlay.height + caption_h), (0, 0, 0))
    canvas.paste(overlay, (0, 0))
    cdraw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    cdraw.text((6, overlay.height + 6), f'query: "{query}"', fill=(255, 255, 255), font=font)

    return canvas
