"""High-resolution visual search: find tiny, confusable targets in a very large image.

The image is large (up to 3600 px wide) and the tags are small, so after the
harness downsamples to the model's input limit the codes can't be read. That
rewards (and measures) active perception: zooming or cropping in the agentic
harness. The distractor codes are visually confusable with the target
(7Q vs 7O vs 1Q vs TQ).
"""

from __future__ import annotations

import math

import numpy as np

from visionbench.core.types import AnswerSpec
from visionbench.tasks.base import TaskFamily, register
from visionbench.tasks.render import NAMED_COLORS, canvas, finish, font

CONFUSABLE = {
    "7Q": ["7O", "1Q", "TQ", "7G", "70"],
    "B8": ["8B", "BB", "88", "E8", "B3"],
    "5S": ["S5", "55", "SS", "56", "6S"],
    "MN": ["NM", "MM", "NN", "HN", "MW"],
    "2Z": ["Z2", "22", "ZZ", "27", "2Y"],
    "QO": ["OQ", "OO", "QQ", "CO", "Q0"],
}
#              d1    d2    d3    d4    d5
WIDTH = [0, 1600, 2200, 2800, 3200, 3600]
N_TAGS = [0, 60, 120, 200, 280, 360]
TAG_FONT = [0, 22, 18, 15, 13, 12]
N_TARGETS = [0, (2, 4), (2, 5), (3, 6), (3, 7), (4, 8)]
COLORED = [0, 0, 0, 1, 1, 1]   # target count restricted to one tag colour


@register
class VisualSearch(TaskFamily):
    name = "visual_search"
    description = "Count tiny confusable codes in a very high-resolution image (rewards zoom/crop)."
    skills = ["fine_grained_perception", "visual_search", "counting"]

    def build(self, rng, style, difficulty, noise_rng):
        W = WIDTH[difficulty]
        H = int(W * 0.62)
        target = str(rng.choice(list(CONFUSABLE)))
        lo, hi = N_TARGETS[difficulty]
        n_t = int(rng.integers(lo, hi + 1))
        colors = ["red", "blue", "green"]
        tcol = str(rng.choice(colors)) if COLORED[difficulty] else None
        fsize = TAG_FONT[difficulty]
        f = font(fsize, "DejaVu Sans Mono", True)
        tw, th = fsize * 1.6, fsize * 1.25

        img, draw = canvas(W, H, style)
        # background clutter: faint lines and blobs so the tags don't pop out
        for _ in range(int(W / 8)):
            x, y = rng.uniform(0, W), rng.uniform(0, H)
            a = rng.uniform(0, math.pi)
            L = rng.uniform(20, 160)
            g = int(rng.integers(200, 235))
            draw.line([(x, y), (x + L * math.cos(a), y + L * math.sin(a))], fill=(g, g, g), width=2)

        placed: list[tuple[float, float]] = []
        tags = []
        n_total = N_TAGS[difficulty]
        # targets first: those matching both code and (if relevant) colour
        specs = []
        for _ in range(n_t):
            specs.append((target, tcol or str(rng.choice(colors)), True))
        # same code, wrong colour (only when colour matters)
        if tcol:
            for _ in range(int(rng.integers(2, 5))):
                specs.append((target, str(rng.choice([c for c in colors if c != tcol])), False))
        while len(specs) < n_total:
            code = str(rng.choice(CONFUSABLE[target])) if rng.random() < 0.6 else \
                str(rng.choice([k for k in CONFUSABLE if k != target]))
            specs.append((code, str(rng.choice(colors)), False))
        order = rng.permutation(len(specs))
        for i in order:
            code, col, is_t = specs[i]
            for _ in range(200):
                x, y = rng.uniform(40, W - 40 - tw), rng.uniform(40, H - 40 - th)
                if all(abs(x - px) > tw + 8 or abs(y - py) > th + 8 for px, py in placed):
                    break
            else:
                continue
            placed.append((x, y))
            c = NAMED_COLORS[col]
            draw.rounded_rectangle([x, y, x + tw, y + th], radius=3, outline=c, width=2, fill=(255, 255, 255))
            draw.text((x + tw / 2, y + th / 2), code, font=f, fill=c, anchor="mm")
            tags.append({"code": code, "color": col, "x": round(x), "y": round(y), "target": is_t})
        count = sum(t["target"] for t in tags)
        img = finish(img, style, noise_rng)
        what = f"the code {target}" if not tcol else f"the code {target} printed in {tcol}"
        prompt = (f"This is a large, high-resolution image ({W}x{H} pixels) containing many small tags, each with a "
                  f"two-character code. Several codes look alike, so read carefully. How many tags show exactly "
                  f"{what}?")
        boxes = [[t["x"], t["y"], t["x"] + tw, t["y"] + th] for t in tags if t["target"]]
        return {
            "prompt": prompt,
            "answer": AnswerSpec(kind="integer", value=int(count), format_hint="a single integer"),
            "images": [img],
            "facts": {"target_code": target, "target_color": tcol, "n_tags": len(tags),
                      "evidence_boxes": boxes, "image_size": [W, H], "tag_font_px": fsize},
        }
