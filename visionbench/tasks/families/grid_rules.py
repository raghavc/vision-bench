"""Abstract rule induction on coloured grids (ARC-style), with a uniqueness guarantee.

A hidden program (1-3 primitives composed) maps input grids to output grids.
The model sees example pairs as an *image only* and must write the test output
as digits. Before an instance is accepted, every program of the same depth in
the DSL is enumerated (with state de-duplication). If any program that fits all
examples predicts a different test output, the instance is ambiguous and is
resampled. Each puzzle therefore has exactly one answer under the DSL.
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np
from scipy import ndimage

from visionbench.core.types import AnswerSpec
from visionbench.tasks.base import TaskFamily, register
from visionbench.tasks.render import canvas, finish

PALETTE = {
    0: ((255, 255, 255), "white (empty)"),
    1: ((30, 100, 220), "blue"),
    2: ((220, 40, 40), "red"),
    3: ((40, 170, 70), "green"),
    4: ((245, 205, 30), "yellow"),
    5: ((128, 128, 128), "gray"),
    6: ((230, 80, 190), "magenta"),
    7: ((250, 135, 25), "orange"),
    8: ((80, 205, 230), "cyan"),
    9: ((135, 75, 35), "brown"),
}

Grid = np.ndarray


# ------------------------------------------------------------------ primitives
def _components(g: Grid):
    lab, n = ndimage.label(g > 0)
    return lab, n


def flip_h(g): return g[:, ::-1]
def flip_v(g): return g[::-1, :]
def rot90(g): return np.rot90(g, -1)
def rot180(g): return np.rot90(g, 2)
def transpose(g): return g.T


def gravity_down(g):
    out = np.zeros_like(g)
    for c in range(g.shape[1]):
        col = g[:, c][g[:, c] > 0]
        if len(col):
            out[-len(col):, c] = col
    return out



def gravity_left(g):
    out = np.zeros_like(g)
    for r in range(g.shape[0]):
        row = g[r][g[r] > 0]
        out[r, : len(row)] = row
    return out


def outline(g):
    """Keep only object cells that touch the background (or the border) 4-adjacently."""
    pad = np.pad(g, 1)
    interior = (pad[1:-1, 1:-1] > 0) & (pad[:-2, 1:-1] > 0) & (pad[2:, 1:-1] > 0) & \
               (pad[1:-1, :-2] > 0) & (pad[1:-1, 2:] > 0)
    out = g.copy()
    out[interior] = 0
    return out


def largest_only(g):
    lab, n = _components(g)
    if n == 0:
        return g.copy()
    sizes = ndimage.sum(np.ones_like(g), lab, range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    return np.where(lab == keep, g, 0)


def recolor_by_size(g):
    lab, n = _components(g)
    out = g.copy()
    for i in range(1, n + 1):
        out[lab == i] = min(9, int((lab == i).sum()))
    return out


def swap_top2(g):
    vals, counts = np.unique(g[g > 0], return_counts=True)
    if len(vals) < 2:
        return g.copy()
    order = np.argsort(-counts, kind="stable")
    a, b = vals[order[0]], vals[order[1]]
    out = g.copy()
    out[g == a], out[g == b] = b, a
    return out


def mirror_lr(g):
    out = g.copy()
    w = g.shape[1]
    out[:, w - w // 2:] = g[:, : w // 2][:, ::-1]
    return out


def crop_bbox(g):
    ys, xs = np.nonzero(g)
    if len(ys) == 0:
        return g.copy()
    return g[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def scale2(g):
    return np.kron(g, np.ones((2, 2), dtype=g.dtype)) if g.size <= 64 else g.copy()


def make_fill_enclosed(color: int) -> Callable[[Grid], Grid]:
    def f(g):
        empty = g == 0
        lab, n = ndimage.label(empty)
        border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
        out = g.copy()
        for i in range(1, n + 1):
            if i not in border:
                out[lab == i] = color
        return out
    f.__name__ = f"fill_enclosed_{color}"
    return f


def make_rays(direction: str) -> Callable[[Grid], Grid]:
    def f(g):
        lab, n = _components(g)
        out = g.copy()
        for i in range(1, n + 1):
            cells = np.argwhere(lab == i)
            if len(cells) != 1:
                continue
            r, c = cells[0]
            col = g[r, c]
            if direction == "right":
                seg = out[r, c + 1:]
            elif direction == "left":
                seg = out[r, :c][::-1]
            elif direction == "down":
                seg = out[r + 1:, c]
            else:
                seg = out[:r, c][::-1]
            for k in range(len(seg)):
                if seg[k] != 0:
                    break
                seg[k] = col
        return out
    f.__name__ = f"rays_{direction}"
    return f


PRIMITIVES: dict[str, Callable[[Grid], Grid]] = {
    f.__name__: f for f in [flip_h, flip_v, rot90, rot180, transpose, gravity_down, gravity_left, outline,
                            largest_only, recolor_by_size, swap_top2, mirror_lr, crop_bbox, scale2]
}
PRIMITIVES.update({f.__name__: f for f in [make_fill_enclosed(c) for c in (2, 3, 4, 6, 8)]})
PRIMITIVES.update({f.__name__: f for f in [make_rays(d) for d in ("right", "left", "down", "up")]})

DESCRIPTIONS = {
    "flip_h": "mirror left-right", "flip_v": "mirror top-bottom", "rot90": "rotate 90° clockwise",
    "rot180": "rotate 180°", "transpose": "transpose", "gravity_down": "cells fall to the bottom",
    "gravity_left": "cells slide to the left", "outline": "hollow out objects", "largest_only": "keep largest object",
    "recolor_by_size": "recolour each object by its size", "swap_top2": "swap the two most common colours",
    "mirror_lr": "complete left-right symmetry from the left half", "crop_bbox": "crop to the objects",
    "scale2": "scale up 2x",
}

#              d1 d2 d3 d4 d5
DEPTH =     [0, 1, 1, 2, 2, 3]
N_EXAMPLES = [0, 3, 3, 3, 4, 4]
SIZE_RANGE = [(0, 0), (5, 7), (6, 9), (6, 9), (7, 10), (7, 11)]


@register
class GridRules(TaskFamily):
    name = "grid_rules"
    description = "Induce a hidden grid transformation from image-only examples (uniqueness-checked)."
    skills = ["abstract_reasoning", "rule_induction", "perception"]

    def build(self, rng, style, difficulty, noise_rng):
        depth = DEPTH[difficulty]
        names = list(PRIMITIVES)
        for _ in range(400):
            prog = [str(rng.choice(names)) for _ in range(depth)]
            if len(set(prog)) < len(prog):
                continue
            inputs = [self._random_grid(rng, difficulty) for _ in range(N_EXAMPLES[difficulty] + 1)]
            try:
                outputs = [run(prog, x) for x in inputs]
            except Exception:
                continue
            # every example must actually change, and grids must stay a sane size
            if any(np.array_equal(x, y) for x, y in zip(inputs, outputs)):
                continue
            if any(y.size == 0 or max(y.shape) > 16 for y in outputs):
                continue
            # no step of the program may be a no-op on all inputs (then it isn't identifiable)
            if any(all(np.array_equal(run(prog[:k], x), run(prog[:k + 1], x)) for x in inputs)
                   for k in range(depth)):
                continue
            ok, n_consistent = unique_prediction(inputs[:-1], outputs[:-1], inputs[-1], depth)
            if ok:
                break
        else:  # pragma: no cover
            raise RuntimeError("no unambiguous grid puzzle found")

        img = self._render(inputs, outputs, style)
        img = finish(img, style, noise_rng)
        legend = ", ".join(f"{k} = {v[1]}" for k, v in PALETTE.items())
        test_out = outputs[-1]
        prompt = (
            "The image shows example pairs of coloured grids: each input grid on the left is turned into the "
            "output grid on the right by the same hidden rule. Work out the rule and apply it to the final test "
            f"input. Write the test output grid using digits for colours ({legend}). "
            f"Note the output may have a different size than the input."
        )
        return {
            "prompt": prompt,
            "answer": AnswerSpec(kind="grid", value=test_out.tolist(), partial_credit=True,
                                 format_hint="one row per line, digits separated by spaces, e.g.\n0 1 1\n2 0 0"),
            "images": [img],
            "facts": {"program": prog, "program_text": " then ".join(DESCRIPTIONS.get(p, p) for p in prog),
                      "test_input": inputs[-1].tolist(), "consistent_programs": n_consistent,
                      "example_inputs": [x.tolist() for x in inputs[:-1]],
                      "example_outputs": [y.tolist() for y in outputs[:-1]]},
        }

    def _random_grid(self, rng, difficulty) -> Grid:
        lo, hi = SIZE_RANGE[difficulty]
        h, w = int(rng.integers(lo, hi + 1)), int(rng.integers(lo, hi + 1))
        g = np.zeros((h, w), dtype=int)
        colors = rng.choice(np.arange(1, 10), size=int(rng.integers(2, 4)), replace=False)
        n_obj = int(rng.integers(2, 4 + difficulty // 2))
        for _ in range(n_obj):
            kind = rng.choice(["hollow", "solid", "pixel", "blob"], p=[0.3, 0.25, 0.25, 0.2])
            col = int(rng.choice(colors))
            if kind == "pixel":
                g[rng.integers(h), rng.integers(w)] = col
                continue
            bh, bw = int(rng.integers(2, max(3, h // 2) + 1)), int(rng.integers(2, max(3, w // 2) + 1))
            if kind == "hollow":
                bh, bw = max(bh, 3), max(bw, 3)
            if bh >= h or bw >= w:
                continue
            r, c = int(rng.integers(0, h - bh + 1)), int(rng.integers(0, w - bw + 1))
            if kind == "solid":
                g[r:r + bh, c:c + bw] = col
            elif kind == "hollow":
                g[r:r + bh, c:c + bw] = col
                g[r + 1:r + bh - 1, c + 1:c + bw - 1] = 0
            else:
                for _ in range(bh * bw // 2 + 1):
                    g[r + rng.integers(bh), c + rng.integers(bw)] = col
        return g

    def _render(self, inputs, outputs, style):
        cell = 24
        gap = 70
        rows = []
        for x, y in zip(inputs, outputs):
            rows.append((x, y))
        widths = [x.shape[1] * cell + gap + y.shape[1] * cell for x, y in rows]
        heights = [max(x.shape[0], y.shape[0]) * cell for x, y in rows]
        W = max(widths) + 220
        H = sum(heights) + 50 * len(rows) + 40
        img, draw = canvas(W, H, style)
        f = style.font(20, bold=True)
        y0 = 30
        for i, (x, y) in enumerate(rows):
            last = i == len(rows) - 1
            label = "Test" if last else f"Example {i + 1}"
            draw.text((20, y0 + heights[i] / 2), label, font=f, fill=style.ink, anchor="lm")
            x0 = 150
            self._grid(draw, x, x0, y0, cell)
            ax = x0 + x.shape[1] * cell + 12
            draw.line([(ax, y0 + heights[i] / 2), (ax + gap - 24, y0 + heights[i] / 2)], fill=style.ink, width=3)
            draw.polygon([(ax + gap - 24, y0 + heights[i] / 2 - 8), (ax + gap - 24, y0 + heights[i] / 2 + 8),
                          (ax + gap - 12, y0 + heights[i] / 2)], fill=style.ink)
            ox = x0 + x.shape[1] * cell + gap
            if last:
                draw.text((ox + 30, y0 + heights[i] / 2), "?", font=style.font(48, bold=True), fill=style.ink, anchor="mm")
            else:
                self._grid(draw, y, ox, y0, cell)
            y0 += heights[i] + 50
        return img

    @staticmethod
    def _grid(draw, g, x0, y0, cell):
        h, w = g.shape
        for r in range(h):
            for c in range(w):
                draw.rectangle([x0 + c * cell, y0 + r * cell, x0 + (c + 1) * cell, y0 + (r + 1) * cell],
                               fill=PALETTE[int(g[r, c])][0], outline=(170, 170, 170), width=1)
        draw.rectangle([x0, y0, x0 + w * cell, y0 + h * cell], outline=(60, 60, 60), width=2)


def run(prog: list[str], g: Grid) -> Grid:
    for p in prog:
        g = PRIMITIVES[p](g)
    return g


def unique_prediction(xs, ys, test_x, depth) -> tuple[bool, int]:
    """Enumerate all programs up to `depth` (deduplicating intermediate states).

    Returns (all consistent programs agree on the test output, #distinct consistent states).
    """
    def key(grids):
        return tuple((g.shape, g.tobytes()) for g in grids)

    start = list(xs) + [test_x]
    frontier = {key(start): start}
    seen = set(frontier)
    predictions = set()
    n_consistent = 0
    for _ in range(depth):
        nxt = {}
        for state in frontier.values():
            for f in PRIMITIVES.values():
                try:
                    new = [f(g) for g in state]
                except Exception:
                    continue
                if any(g.size == 0 or max(g.shape) > 24 for g in new):
                    continue
                k = key(new)
                if k in seen:
                    continue
                seen.add(k)
                nxt[k] = new
                if all(np.array_equal(a, b) for a, b in zip(new[:-1], ys)):
                    n_consistent += 1
                    predictions.add((new[-1].shape, new[-1].tobytes()))
        frontier = nxt
    return len(predictions) == 1, n_consistent


def parse_grid(text: str) -> list[list[int]] | None:
    rows = []
    for line in str(text).strip().splitlines():
        digits = re.findall(r"\d", line)
        if digits:
            rows.append([int(d) for d in digits])
    return rows or None
