"""Deterministic rendering helpers shared by procedural task families.

Problem content comes from the problem RNG; everything here that varies
(palette, fonts, stroke widths, paper tone, sensor noise) comes from the
*style* RNG, so a re-render keeps the answer and changes only the look.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_FAMILIES = ["DejaVu Sans", "DejaVu Serif", "DejaVu Sans Mono"]

# Colour names a model can be asked about, with RGB values far enough apart
# that naming them is unambiguous.
NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 40, 40),
    "blue": (40, 90, 220),
    "green": (30, 150, 60),
    "orange": (245, 140, 20),
    "purple": (130, 60, 180),
    "black": (25, 25, 25),
    "yellow": (235, 200, 30),
    "gray": (140, 140, 140),
    "brown": (130, 80, 40),
    "pink": (235, 120, 180),
    "cyan": (30, 180, 200),
}


@lru_cache(maxsize=64)
def font(size: int, family: str = "DejaVu Sans", bold: bool = False) -> ImageFont.FreeTypeFont:
    prop = font_manager.FontProperties(family=family, weight="bold" if bold else "normal")
    path = font_manager.findfont(prop, fallback_to_default=True)
    return ImageFont.truetype(path, size)


@dataclass
class Style:
    background: tuple[int, int, int]
    ink: tuple[int, int, int]
    font_family: str
    line_scale: float
    noise: float
    blur: float
    hue_shift: int

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "Style":
        tone = int(rng.integers(238, 256))
        warm = int(rng.integers(0, 8))
        return cls(
            background=(tone, tone - warm // 2, tone - warm),
            ink=(int(rng.integers(10, 40)),) * 3,
            font_family=str(rng.choice(FONT_FAMILIES)),
            line_scale=float(rng.uniform(0.85, 1.25)),
            noise=float(rng.choice([0.0, 0.0, 3.0, 6.0])),
            blur=float(rng.choice([0.0, 0.0, 0.0, 0.4])),
            hue_shift=int(rng.integers(0, 3)),
        )

    @classmethod
    def plain(cls) -> "Style":
        return cls((255, 255, 255), (20, 20, 20), "DejaVu Sans", 1.0, 0.0, 0.0, 0)

    def lw(self, base: float) -> int:
        return max(1, int(round(base * self.line_scale)))

    def font(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        return font(size, self.font_family, bold)


def canvas(w: int, h: int, style: Style) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (w, h), style.background)
    return img, ImageDraw.Draw(img)


def finish(img: Image.Image, style: Style, rng: np.random.Generator | None = None) -> Image.Image:
    """Apply sensor-style degradations that never change the answer."""
    if style.blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(style.blur))
    if style.noise > 0:
        rng = rng or np.random.default_rng(0)
        arr = np.asarray(img).astype(np.int16)
        arr = arr + rng.normal(0, style.noise, arr.shape).astype(np.int16)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str,
                fnt: ImageFont.FreeTypeFont, fill, stroke: int = 0, stroke_fill=None) -> None:
    x, y = xy
    draw.text((x, y), text, font=fnt, fill=fill, anchor="mm",
              stroke_width=stroke, stroke_fill=stroke_fill)


def bezier(points: list[tuple[float, float]], n: int = 200) -> list[tuple[float, float]]:
    """Evaluate a Bezier curve of arbitrary degree (de Casteljau)."""
    pts = np.array(points, dtype=float)
    out = []
    for t in np.linspace(0, 1, n):
        p = pts.copy()
        while len(p) > 1:
            p = (1 - t) * p[:-1] + t * p[1:]
        out.append((float(p[0, 0]), float(p[0, 1])))
    return out


def polyline_intersections(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> int:
    """Count proper crossings between two polylines."""
    def seg_cross(p1, p2, p3, p4) -> bool:
        def orient(p, q, r):
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
        d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)
        return (d1 * d2 < 0) and (d3 * d4 < 0)

    A = np.array(a)
    B = np.array(b)
    count = 0
    for i in range(len(A) - 1):
        ax0, ax1 = sorted((A[i, 0], A[i + 1, 0]))
        ay0, ay1 = sorted((A[i, 1], A[i + 1, 1]))
        # coarse bbox filter
        mask = ~((np.maximum(B[:-1, 0], B[1:, 0]) < ax0) | (np.minimum(B[:-1, 0], B[1:, 0]) > ax1) |
                 (np.maximum(B[:-1, 1], B[1:, 1]) < ay0) | (np.minimum(B[:-1, 1], B[1:, 1]) > ay1))
        for j in np.nonzero(mask)[0]:
            if seg_cross(A[i], A[i + 1], B[j], B[j + 1]):
                count += 1
    return count


def regular_polygon(cx: float, cy: float, r: float, n: int, rot: float = 0.0) -> list[tuple[float, float]]:
    return [(cx + r * math.cos(rot + 2 * math.pi * k / n), cy + r * math.sin(rot + 2 * math.pi * k / n))
            for k in range(n)]


def arrow(draw: ImageDraw.ImageDraw, p0, p1, fill, width: int = 3, head: float = 12) -> None:
    draw.line([p0, p1], fill=fill, width=width)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    left = (p1[0] - head * math.cos(ang - 0.4), p1[1] - head * math.sin(ang - 0.4))
    right = (p1[0] - head * math.cos(ang + 0.4), p1[1] - head * math.sin(ang + 0.4))
    draw.polygon([p1, left, right], fill=fill)
