"""Image encoding with an explicit, logged resolution policy.

Resolution is a confounder (O3-Bench measured about 10 points from the image
cap alone), so every image is resized deterministically to fit `max_side`
before sending, and the sent size is recorded in the trajectory.
"""

from __future__ import annotations

import base64
import io

from PIL import Image


def fit(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return img


def encode_image(img: Image.Image, max_side: int) -> tuple[dict, tuple[int, int]]:
    sent = fit(img.convert("RGB"), max_side)
    buf = io.BytesIO()
    sent.save(buf, format="PNG", optimize=True)
    data = buf.getvalue()
    if len(data) > 4_500_000:  # stay well under the per-image request limit
        buf = io.BytesIO()
        sent.save(buf, format="JPEG", quality=92)
        data, media = buf.getvalue(), "image/jpeg"
    else:
        media = "image/png"
    block = {"type": "image", "source": {"type": "base64", "media_type": media,
                                         "data": base64.standard_b64encode(data).decode()}}
    return block, sent.size


def crop_zoom(img: Image.Image, box: tuple[float, float, float, float], max_side: int) -> Image.Image:
    """Crop a region given in *original-image* pixel coordinates and upscale it to use the full budget."""
    W, H = img.size
    x0, y0, x1, y1 = box
    x0, x1 = sorted((max(0, min(W, x0)), max(0, min(W, x1))))
    y0, y1 = sorted((max(0, min(H, y0)), max(0, min(H, y1))))
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise ValueError("crop region too small (min 8x8 px)")
    region = img.crop((round(x0), round(y0), round(x1), round(y1)))
    w, h = region.size
    s = min(max_side / max(w, h), 4.0)  # at most 4x upsampling
    return region.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
