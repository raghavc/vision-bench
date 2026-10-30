"""Compositional counting under clutter and partial occlusion.

Counting items that satisfy a conjunction of attributes (shape, colour, size,
fill, region membership, negation) is a known frontier weakness. Occlusion is
capped at 25% of any shape's area, so every shape stays identifiable; shapes
never straddle the region boundary.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from visionbench.core.types import AnswerSpec
from visionbench.tasks.base import TaskFamily, register
from visionbench.tasks.render import NAMED_COLORS, canvas, finish, regular_polygon

SHAPES = ["circle", "square", "triangle", "star", "hexagon"]
COLORS = ["red", "blue", "green", "orange", "purple"]
#             d1  d2  d3  d4  d5
N_SHAPES = [0, 12, 18, 26, 34, 44]
MAX_OCCL = [0, 0.0, 0.0, 0.15, 0.25, 0.25]
REGION = [0, 0, 0, 0, 1, 1]
N_PRED = [0, 1, 2, 2, 3, 3]


def shape_points(kind, cx, cy, r, rot):
    if kind == "square":
        return regular_polygon(cx, cy, r * 1.15, 4, rot + math.pi / 4)
    if kind == "triangle":
        return regular_polygon(cx, cy, r * 1.2, 3, rot - math.pi / 2)
    if kind == "hexagon":
        return regular_polygon(cx, cy, r * 1.05, 6, rot)
    if kind == "star":
        pts = []
        for k in range(10):
            rr = r * 1.25 if k % 2 == 0 else r * 0.55
            a = rot - math.pi / 2 + k * math.pi / 5
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        return pts
    raise ValueError(kind)


def draw_shape(draw, s, ink_w, color=None):
    color = NAMED_COLORS[s["color"]] if color is None else color
    if s["fill"] == "filled":
        fill, outline = color, color
    else:
        fill, outline = None, color
    if s["shape"] == "circle":
        box = [s["x"] - s["r"], s["y"] - s["r"], s["x"] + s["r"], s["y"] + s["r"]]
        draw.ellipse(box, fill=fill, outline=outline, width=ink_w)
    else:
        draw.polygon(shape_points(s["shape"], s["x"], s["y"], s["r"], s["rot"]), fill=fill, outline=outline,
                     width=ink_w)


@register
class CompositionalCounting(TaskFamily):
    name = "counting"
    description = "Count objects matching compositional attribute queries under clutter/occlusion."
    skills = ["counting", "attribute_binding", "perception"]

    def build(self, rng, style, difficulty, noise_rng):
        W, H = 1100, 800
        region = None
        if REGION[difficulty]:
            rw, rh = rng.uniform(380, 520), rng.uniform(280, 380)
            rx, ry = rng.uniform(60, W - rw - 60), rng.uniform(60, H - rh - 60)
            region = (rx, ry, rx + rw, ry + rh)
        for _ in range(300):
            shapes = self._place(rng, difficulty, W, H, region)
            if shapes is None:
                continue
            q = self._query(rng, difficulty, shapes, region)
            if q is not None:
                break
        else:  # pragma: no cover
            raise RuntimeError("counting layout failed")
        text, count, preds = q
        img, draw = canvas(W, H, style)
        if region:
            x0, y0, x1, y1 = region
            for t in np.arange(x0, x1, 18):
                draw.line([(t, y0), (min(t + 10, x1), y0)], fill=style.ink, width=3)
                draw.line([(t, y1), (min(t + 10, x1), y1)], fill=style.ink, width=3)
            for t in np.arange(y0, y1, 18):
                draw.line([(x0, t), (x0, min(t + 10, y1))], fill=style.ink, width=3)
                draw.line([(x1, t), (x1, min(t + 10, y1))], fill=style.ink, width=3)
        for s in shapes:
            draw_shape(draw, s, style.lw(4))
        img = finish(img, style, noise_rng)
        legend = ("Shapes are circles, squares, triangles, stars or hexagons; colours are red, blue, green, orange "
                  "or purple; each shape is either small or large, and either filled (solid) or outlined (hollow).")
        return {
            "prompt": f"{legend} {text}",
            "answer": AnswerSpec(kind="integer", value=count, format_hint="a single integer"),
            "images": [img],
            "facts": {"shapes": [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in s.items()}
                                 for s in shapes],
                      "predicates": preds, "total_shapes": len(shapes)},
        }

    def _place(self, rng, d, W, H, region):
        shapes = []
        masks = []
        n = N_SHAPES[d]
        tries = 0
        while len(shapes) < n and tries < n * 60:
            tries += 1
            size = str(rng.choice(["small", "large"]))
            r = rng.uniform(15, 19) if size == "small" else rng.uniform(32, 38)
            x, y = rng.uniform(r + 10, W - r - 10), rng.uniform(r + 10, H - r - 10)
            if region is not None:
                x0, y0, x1, y1 = region
                inside_x = x0 + r * 1.4 < x < x1 - r * 1.4 and y0 + r * 1.4 < y < y1 - r * 1.4
                outside = x + r * 1.4 < x0 or x - r * 1.4 > x1 or y + r * 1.4 < y0 or y - r * 1.4 > y1
                if not (inside_x or outside):
                    continue
            s = {"shape": str(rng.choice(SHAPES)), "color": str(rng.choice(COLORS)), "size": size,
                 "fill": str(rng.choice(["filled", "outlined"])), "x": float(x), "y": float(y), "r": float(r),
                 "rot": float(rng.uniform(0, 2 * math.pi)) if d >= 3 else 0.0}
            if region is not None:
                s["in_region"] = bool(inside_x)
            m = Image.new("1", (W, H), 0)
            sd = ImageDraw.Draw(m)
            draw_shape(sd, {**s, "fill": "filled"}, 1, color=1)
            mask = np.asarray(m)
            # the new shape (drawn on top) may cover at most MAX_OCCL of each earlier shape;
            # with MAX_OCCL == 0 we also keep a small clearance
            ok = True
            for s2, m2 in zip(shapes, masks):
                if math.hypot(s2["x"] - x, s2["y"] - y) > (s2["r"] + r) * 1.5:
                    continue
                if MAX_OCCL[d] == 0:
                    ok = False
                    break
                covered = (m2 & mask).sum() / max(1, m2.sum())
                s2_cover = s2.get("_covered", 0.0) + covered
                if s2_cover > MAX_OCCL[d]:
                    ok = False
                    break
            if not ok:
                continue
            for s2, m2 in zip(shapes, masks):
                if math.hypot(s2["x"] - x, s2["y"] - y) <= (s2["r"] + r) * 1.5:
                    s2["_covered"] = s2.get("_covered", 0.0) + (m2 & mask).sum() / max(1, m2.sum())
            shapes.append(s)
            masks.append(mask)
        if len(shapes) < n:
            return None
        for s in shapes:
            s.pop("_covered", None)
        return shapes

    def _query(self, rng, d, shapes, region):
        attrs = ["shape", "color", "size", "fill"]
        k = N_PRED[d]
        preds = []
        chosen = list(rng.choice(attrs, size=k, replace=False))
        for a in chosen:
            if a == "shape":
                v = str(rng.choice(SHAPES))
            elif a == "color":
                v = str(rng.choice(COLORS))
            elif a == "size":
                v = str(rng.choice(["small", "large"]))
            else:
                v = str(rng.choice(["filled", "outlined"]))
            neg = bool(d >= 3 and rng.random() < 0.3)
            preds.append({"attr": a, "value": v, "negated": neg})
        if region is not None:
            preds.append({"attr": "in_region", "value": bool(rng.random() < 0.6), "negated": False})

        def match(s):
            for p in preds:
                hit = s[p["attr"]] == p["value"]
                if p["negated"]:
                    hit = not hit
                if not hit:
                    return False
            return True
        count = sum(match(s) for s in shapes)
        if count < 2 or count == len(shapes):
            return None
        phrase = self._phrase(preds)
        return f"Count the {phrase}. How many are there?", count, preds

    @staticmethod
    def _phrase(preds):
        """Natural noun phrase, e.g. 'small outlined shapes that are not red, inside the dashed rectangle'."""
        adj, noun, clauses, region = [], "shapes", [], ""
        for a in ("size", "fill", "color", "shape"):
            for p in preds:
                if p["attr"] != a:
                    continue
                if p["negated"]:
                    clauses.append(f"not {p['value']}s" if a == "shape" else f"not {p['value']}")
                elif a == "shape":
                    noun = p["value"] + "es" if p["value"].endswith("x") else p["value"] + "s"
                else:
                    adj.append(p["value"])
        for p in preds:
            if p["attr"] == "in_region":
                region = "inside the dashed rectangle" if p["value"] else "outside the dashed rectangle"
        phrase = " ".join(adj + [noun])
        if clauses:
            phrase += " that are " + " and ".join(clauses)
        if region:
            phrase += (", " if clauses else " ") + region
        return phrase
