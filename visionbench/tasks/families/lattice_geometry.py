"""Lattice-polygon geometry read off a labeled coordinate grid.

The vertices are lattice points but are not labeled: the model has to read
coordinates from the axes (which don't start at 0 and are labeled only every
other unit at higher levels) and then compute an exact area, an interior
lattice point count (Pick's theorem) or the area of a polygon with a
rectangular hole.
"""

from __future__ import annotations

import math

import numpy as np

from visionbench.core.types import AnswerSpec
from visionbench.tasks.base import TaskFamily, register
from visionbench.tasks.render import canvas, finish

#           d1 d2 d3 d4 d5
N_VERT = [0, 4, 5, 7, 8, 8]
SPAN = [0, 10, 12, 14, 16, 16]
LABEL_EVERY = [0, 1, 1, 2, 2, 2]
QUESTION = [None, "area", "area", "area", "interior_points", "area_with_hole"]


def shoelace2(pts) -> int:
    """Twice the signed area (exact integer)."""
    s = 0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        s += x1 * y2 - x2 * y1
    return s


def boundary_points(pts) -> int:
    return sum(math.gcd(abs(x2 - x1), abs(y2 - y1)) for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]))


def point_in_poly(px, py, pts) -> bool:
    inside = False
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        if (y1 > py) != (y2 > py):
            xi = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < xi:
                inside = not inside
    return inside


def simple_polygon(pts) -> bool:
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def inter(p1, p2, p3, p4):
        d1, d2, d3, d4 = cross(p3, p4, p1), cross(p3, p4, p2), cross(p1, p2, p3), cross(p1, p2, p4)
        return d1 * d2 < 0 and d3 * d4 < 0

    n = len(pts)
    edges = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        if cross(pts[i - 1], pts[i], pts[(i + 1) % n]) == 0:
            return False  # collinear consecutive vertices: an invisible "vertex"
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if inter(*edges[i], *edges[j]):
                return False
    return True


@register
class LatticeGeometry(TaskFamily):
    name = "lattice_geometry"
    description = "Read polygon vertices off a coordinate grid and compute exact area / lattice counts."
    skills = ["measurement", "geometry", "arithmetic"]

    def build(self, rng, style, difficulty, noise_rng):
        span = SPAN[difficulty]
        ox, oy = int(rng.integers(-6, 4)), int(rng.integers(-6, 4))
        for _ in range(2000):
            pts = self._polygon(rng, N_VERT[difficulty], span, ox, oy, convex=difficulty <= 2)
            if pts is None:
                continue
            twice = abs(shoelace2(pts))
            if twice < span * 6:
                continue
            hole = None
            if QUESTION[difficulty] == "area_with_hole":
                hole = self._hole(rng, pts, ox, oy, span)
                if hole is None:
                    continue
            break
        else:  # pragma: no cover
            raise RuntimeError("no polygon")
        q = QUESTION[difficulty]
        B = boundary_points(pts)
        A2 = abs(shoelace2(pts))
        interior = (A2 - B + 2) // 2  # Pick: A = I + B/2 - 1
        if q == "area":
            value, text = A2 / 2, "What is the exact area of the shaded polygon, in square units?"
            kind = "number"
        elif q == "interior_points":
            value = int(interior)
            text = ("How many lattice points (points with integer coordinates) lie strictly inside the shaded "
                    "polygon (not on its boundary)?")
            kind = "integer"
        else:
            hx0, hy0, hx1, hy1 = hole
            value = A2 / 2 - (hx1 - hx0) * (hy1 - hy0)
            text = ("The shaded region is the polygon with the white rectangular hole removed. What is the exact "
                    "area of the shaded region, in square units?")
            kind = "number"
        img = self._render(pts, hole, ox, oy, span, style, LABEL_EVERY[difficulty])
        img = finish(img, style, noise_rng)
        prompt = ("All vertices of the polygon lie on grid points of the labeled coordinate grid. " + text)
        return {
            "prompt": prompt,
            "answer": AnswerSpec(kind=kind, value=value, tolerance=1e-6,
                                 format_hint="a single number (decimals allowed, e.g. 37.5)"),
            "images": [img],
            "facts": {"vertices": pts, "hole": hole, "twice_area": A2, "boundary_points": B,
                      "interior_points": int(interior)},
        }

    def _polygon(self, rng, n, span, ox, oy, convex):
        cx, cy = ox + span / 2, oy + span / 2
        angles = np.sort(rng.uniform(0, 2 * np.pi, n))
        if np.min(np.diff(np.concatenate([angles, [angles[0] + 2 * np.pi]]))) < 0.35:
            return None
        pts = []
        for a in angles:
            r = span * (0.42 if convex else rng.uniform(0.18, 0.48))
            x = int(round(cx + r * math.cos(a)))
            y = int(round(cy + r * math.sin(a)))
            x = min(max(x, ox + 1), ox + span - 1)
            y = min(max(y, oy + 1), oy + span - 1)
            pts.append((x, y))
        if len(set(pts)) < n or not simple_polygon(pts):
            return None
        if convex:
            # require strict convexity
            signs = set()
            for i in range(n):
                (x0, y0), (x1, y1), (x2, y2) = pts[i - 1], pts[i], pts[(i + 1) % n]
                signs.add(np.sign((x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1)))
            if len(signs) != 1:
                return None
        return pts

    def _hole(self, rng, pts, ox, oy, span):
        for _ in range(200):
            w, h = int(rng.integers(2, 5)), int(rng.integers(2, 4))
            x0, y0 = int(rng.integers(ox + 1, ox + span - w)), int(rng.integers(oy + 1, oy + span - h))
            samples = [(x0 + t * w, y0 + u * h) for t in np.linspace(0, 1, 9) for u in np.linspace(0, 1, 9)]
            # strictly inside with a margin: nudge samples outward by 0.3 to keep clear of the boundary
            ring = [(x0 - 0.3 + t * (w + 0.6), y0 - 0.3 + u * (h + 0.6)) for t in np.linspace(0, 1, 13)
                    for u in (0, 1)] + [(x0 - 0.3 + t * (w + 0.6), y0 - 0.3 + u * (h + 0.6))
                                         for u in np.linspace(0, 1, 13) for t in (0, 1)]
            if all(point_in_poly(px + 1e-3, py + 1e-3, pts) for px, py in samples + ring):
                return (x0, y0, x0 + w, y0 + h)
        return None

    def _render(self, pts, hole, ox, oy, span, style, label_every):
        unit = 44
        pad = 70
        W = H = pad * 2 + unit * span
        img, draw = canvas(W, H, style)

        def P(x, y):
            return pad + (x - ox) * unit, pad + (oy + span - y) * unit

        poly = [P(*p) for p in pts]
        # fill first, then grid lines over it so vertices stay readable inside the shading
        draw.polygon(poly, fill=(176, 210, 244))
        if hole:
            x0, y0, x1, y1 = hole
            draw.rectangle([P(x0, y1), P(x1, y0)], fill=style.background)
        for k in range(span + 1):
            col = (190, 190, 190)
            draw.line([P(ox + k, oy), P(ox + k, oy + span)], fill=col, width=1)
            draw.line([P(ox, oy + k), P(ox + span, oy + k)], fill=col, width=1)
        f = style.font(16)
        for k in range(span + 1):
            if (ox + k) % label_every == 0:
                x, y = P(ox + k, oy)
                draw.text((x, y + 16), str(ox + k), font=f, fill=style.ink, anchor="mm")
            if (oy + k) % label_every == 0:
                x, y = P(ox, oy + k)
                draw.text((x - 20, y), str(oy + k), font=f, fill=style.ink, anchor="mm")
        if ox <= 0 <= ox + span:
            draw.line([P(0, oy), P(0, oy + span)], fill=style.ink, width=2)
        if oy <= 0 <= oy + span:
            draw.line([P(ox, 0), P(ox + span, 0)], fill=style.ink, width=2)
        draw.polygon(poly, outline=(30, 80, 160), width=style.lw(3))
        if hole:
            draw.rectangle([P(x0, y1), P(x1, y0)], outline=(30, 80, 160), width=style.lw(2))
        for p in poly:
            draw.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=(30, 80, 160))
        return img
