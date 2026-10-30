"""Gear trains: propagate rotation direction and speed through meshes and belts.

Ground truth comes from kinematics on the generated topology: a mesh reverses
direction and scales speed by T_in/T_out, an open belt keeps direction, a
crossed belt reverses it. At higher difficulty there are idler gears, branches,
crossed belts, and "near miss" gears separated by a clearly visible gap (they
never turn). The model has to see the topology; teeth counts alone don't give
the answer.
"""

from __future__ import annotations

import math
import re
import string
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from visionbench.core.types import AnswerSpec
from visionbench.tasks.base import TaskFamily, register
from visionbench.tasks.render import arrow, canvas, finish

MODULE = 5.0          # px of pitch radius per tooth/2
ADDENDUM = 5.0
#            d1 d2 d3 d4 d5
N_GEARS = [0, 3, 5, 6, 7, 9]
N_BELTS = [0, 0, 0, 1, 1, 2]
N_DECOYS = [0, 0, 0, 0, 1, 2]
P_CROSSED = [0, 0, 0, 0, 0.5, 0.6]


@dataclass
class Gear:
    label: str
    x: float
    y: float
    teeth: int
    parent: int | None
    link: str            # "drive" | "mesh" | "open_belt" | "crossed_belt" | "gap"
    phase: float = 0.0

    @property
    def r(self) -> float:
        return self.teeth * MODULE / 2

    @property
    def r_out(self) -> float:
        return self.r + ADDENDUM


@register
class GearTrain(TaskFamily):
    name = "gear_train"
    version = "2"  # v2: at most one belt per gear (collinear belt segments looked like one belt; found live)
    description = "Propagate rotation direction and speed through gear meshes and belts."
    skills = ["physical_reasoning", "spatial_topology", "arithmetic"]

    def build(self, rng, style, difficulty, noise_rng):
        W, H = 1200, 900
        for _ in range(500):
            gears, belts = self._layout(rng, difficulty, W, H)
            if gears is not None:
                break
        else:  # pragma: no cover
            raise RuntimeError("gear layout failed")

        rpm_in = int(rng.choice([12, 18, 20, 24, 30, 36, 40, 45, 48, 60]))
        cw_in = bool(rng.integers(0, 2))
        motion = self._solve(gears, rpm_in, cw_in)

        candidates = [i for i in range(1, len(gears))]
        driven = [i for i in candidates if motion[i] is not None]
        stationary = [i for i in candidates if motion[i] is None]
        if stationary and rng.random() < 0.35:
            target = int(rng.choice(stationary))
        else:
            # prefer the deepest driven gear: longest chain of reasoning
            depth = {i: self._depth(gears, i) for i in driven}
            maxd = max(depth.values())
            target = int(rng.choice([i for i in driven if depth[i] >= maxd - 1]))

        img = self._render(gears, belts, style, rpm_in, cw_in, W, H)
        img = finish(img, style, noise_rng)

        if motion[target] is None:
            truth = {"direction": "stationary", "rpm": Fraction(0)}
        else:
            cw, rpm = motion[target]
            truth = {"direction": "clockwise" if cw else "counterclockwise", "rpm": rpm}

        prompt = (
            f"Gear {gears[0].label} is the driver: it turns {'clockwise' if cw_in else 'counterclockwise'} "
            f"at {rpm_in} rpm, as marked. Each gear is labeled with a letter and its number of teeth (T). "
            "Touching gears mesh. A belt or chain connects two gears; a belt drawn as an X (crossed) "
            "reverses direction, and speeds through belts scale by teeth counts just like meshes. "
            "Gears that don't touch anything are not driven. "
            f"In which direction does gear {gears[target].label} turn, and at what speed in rpm?"
        )
        return {
            "prompt": prompt,
            "answer": AnswerSpec(
                kind="custom",
                value={"direction": truth["direction"], "rpm": float(truth["rpm"])},
                format_hint=("direction then rpm (exact fraction or at least 2 decimals), e.g. 'clockwise 22.5' "
                             "or 'counterclockwise 128/3', or 'stationary' if it does not turn"),
            ),
            "images": [img],
            "facts": {
                "gears": {g.label: {"teeth": g.teeth, "parent": gears[g.parent].label if g.parent is not None else None,
                                    "link": g.link} for g in gears},
                "motion": {gears[i].label: (None if m is None else
                                            {"cw": m[0], "rpm": float(m[1])}) for i, m in enumerate(motion)},
                "target": gears[target].label,
                "chain_length": self._depth(gears, target),
            },
        }

    # ------------------------------------------------------------------ layout
    def _layout(self, rng, d, W, H):
        labels = list(string.ascii_uppercase)
        rng_labels = rng.permutation(len(labels[:N_GEARS[d] + N_DECOYS[d]]))
        lab = [labels[i] for i in rng_labels]
        gears: list[Gear] = []
        teeth0 = int(rng.integers(14, 19)) * 2
        r0 = teeth0 * MODULE / 2
        gears.append(Gear(lab[0], rng.uniform(r0 + 60, W * 0.4), rng.uniform(r0 + 60, H - r0 - 60), teeth0, None, "drive"))
        belts: list[tuple[int, int, bool]] = []
        n_belts = N_BELTS[d]
        n_total = N_GEARS[d]
        while len(gears) < n_total:
            # prefer extending a chain from recent gears
            parent = int(rng.choice(range(len(gears)), p=self._recent_weights(len(gears))))
            teeth = int(rng.integers(8, 21)) * 2
            remaining = n_total - len(gears)
            belted = {a for a, _, _ in belts} | {b for _, b, _ in belts}
            use_belt = n_belts > 0 and parent not in belted and (rng.random() < 0.4 or remaining <= n_belts)
            link = "mesh"
            if use_belt:
                link = "crossed_belt" if rng.random() < P_CROSSED[d] else "open_belt"
            g = self._place(rng, gears, parent, teeth, link, W, H, belts)
            if g is None:
                return None, None
            gears.append(g)
            if use_belt:
                belts.append((parent, len(gears) - 1, link == "crossed_belt"))
                n_belts -= 1
        for _ in range(N_DECOYS[d]):
            parent = int(rng.integers(0, len(gears)))
            teeth = int(rng.integers(8, 16)) * 2
            g = self._place(rng, gears, parent, teeth, "gap", W, H, belts)
            if g is None:
                return None, None
            gears.append(g)
        for i, g in enumerate(gears):
            g.label = lab[i]
        return gears, belts

    @staticmethod
    def _recent_weights(n: int) -> np.ndarray:
        w = np.array([1.5 ** i for i in range(n)])
        return w / w.sum()

    def _place(self, rng, gears, parent, teeth, link, W, H, belts) -> Gear | None:
        p = gears[parent]
        r = teeth * MODULE / 2
        for _ in range(60):
            ang = rng.uniform(0, 2 * math.pi)
            if link == "mesh":
                dist = p.r + r
            elif link == "gap":
                dist = p.r_out + r + ADDENDUM + rng.uniform(16, 26)
            else:
                dist = p.r_out + r + rng.uniform(110, 260)
            x, y = p.x + dist * math.cos(ang), p.y + dist * math.sin(ang)
            if not (r + 50 < x < W - r - 50 and r + 50 < y < H - r - 50):
                continue
            ok = True
            for j, q in enumerate(gears):
                dd = math.hypot(x - q.x, y - q.y)
                if j == parent:
                    continue
                if dd < q.r_out + r + ADDENDUM + 28:  # never touch or almost-touch a non-parent
                    ok = False
                    break
            if not ok:
                continue
            g = Gear("?", x, y, teeth, parent, link)
            if link in ("open_belt", "crossed_belt"):
                segs = belt_segments(p, g, link == "crossed_belt")
                others = [q for j, q in enumerate(gears) if j != parent]
                if any(seg_circle_dist(s, (q.x, q.y)) < q.r_out + 14 for s in segs for q in others):
                    continue
            # existing belts must not pass through the new gear
            if any(seg_circle_dist(s, (x, y)) < r + ADDENDUM + 14
                   for a, b, crossed in belts for s in belt_segments(gears[a], gears[b], crossed)):
                continue
            if link == "mesh":
                # align teeth: a gap of the child faces a tooth of the parent
                pitch_p = 2 * math.pi / p.teeth
                k = round((ang - p.phase) / pitch_p)
                contact = p.phase + k * pitch_p
                g.phase = contact + math.pi + math.pi / teeth
            else:
                g.phase = rng.uniform(0, 2 * math.pi)
            return g
        return None

    # ------------------------------------------------------------- kinematics
    @staticmethod
    def _solve(gears: list[Gear], rpm_in: int, cw_in: bool):
        motion: list[tuple[bool, Fraction] | None] = [None] * len(gears)
        motion[0] = (cw_in, Fraction(rpm_in))
        for i in range(1, len(gears)):  # parents always precede children
            g = gears[i]
            pm = motion[g.parent]
            if pm is None or g.link == "gap":
                continue
            cw, rpm = pm
            new_rpm = rpm * Fraction(gears[g.parent].teeth, g.teeth)
            if g.link in ("mesh", "crossed_belt"):
                cw = not cw
            motion[i] = (cw, new_rpm)
        return motion

    @staticmethod
    def _depth(gears, i) -> int:
        d = 0
        while gears[i].parent is not None:
            i = gears[i].parent
            d += 1
        return d

    # --------------------------------------------------------------- render
    def _render(self, gears, belts, style, rpm_in, cw_in, W, H):
        img, draw = canvas(W, H, style)
        ink = style.ink
        belt_color = (90, 90, 90)
        for a, b, crossed in belts:
            for s in belt_segments(gears[a], gears[b], crossed):
                draw.line(s, fill=belt_color, width=style.lw(5))
        fills = [(214, 222, 235), (232, 220, 200), (215, 232, 214), (235, 214, 214), (226, 216, 236)]
        for i, g in enumerate(gears):
            pts = gear_polygon(g)
            draw.polygon(pts, fill=fills[i % len(fills)], outline=ink, width=style.lw(2))
            draw.ellipse([g.x - 6, g.y - 6, g.x + 6, g.y + 6], fill=ink)
            big = 22 if g.r >= 50 else 18
            draw.text((g.x, g.y - 17), g.label, font=style.font(big, bold=True), fill=ink, anchor="mm")
            draw.text((g.x, g.y + 18), f"{g.teeth}T", font=style.font(big - 6), fill=ink, anchor="mm")
        # driver: rotation arrow drawn inside the gear body so it can't collide with neighbours
        d = gears[0]
        rr = d.r - ADDENDUM - 12
        a0, a1 = (200, 340) if cw_in else (340, 200)
        n = 24
        arc = [(d.x + rr * math.cos(math.radians(a0 + (a1 - a0) * t / n)),
                d.y + rr * math.sin(math.radians(a0 + (a1 - a0) * t / n))) for t in range(n + 1)]
        draw.line(arc, fill=(200, 30, 30), width=style.lw(4))
        arrow(draw, arc[-3], arc[-1], (200, 30, 30), width=style.lw(4), head=14)
        draw.text((d.x, d.y + 44), "DRIVER", font=style.font(15, bold=True), fill=(200, 30, 30), anchor="mm")
        return img

    # ---------------------------------------------------------------- check
    def check(self, instance, parsed):
        truth = instance.answer.value
        text = str(parsed).lower().strip()
        if truth["direction"] == "stationary":
            ok = bool(re.search(r"\b(stationary|does not (turn|rotate)|doesn't (turn|rotate)|not driven|0 rpm)\b", text)) \
                 and not re.search(r"\b(counter)?clockwise\b", text)
            return float(ok), {"direction_ok": ok, "rpm_ok": ok}
        m_dir = re.search(r"\b(counter-?clockwise|anti-?clockwise|ccw|clockwise|cw)\b", text)
        direction = None
        if m_dir:
            tok = m_dir.group(1).replace("-", "")
            direction = "clockwise" if tok in ("clockwise", "cw") else "counterclockwise"
        rpm = parse_number(text)
        dir_ok = direction == truth["direction"]
        # exact (fractions allowed) or correctly rounded to the precision the answer states (>= 1 decimal)
        rpm_ok = rpm is not None and (math.isclose(rpm, truth["rpm"], rel_tol=1e-3, abs_tol=0.01)
                                      or _rounds_to(truth["rpm"], text))
        return float(dir_ok and rpm_ok), {"direction": direction, "rpm": rpm,
                                          "direction_ok": dir_ok, "rpm_ok": rpm_ok}


def _rounds_to(truth: float, text: str) -> bool:
    m = re.search(r"(-?\d+)\.(\d+)", text.replace(",", ""))
    if not m:
        return False
    decimals = len(m.group(2))
    return abs(float(m.group(0)) - truth) <= 0.5 * 10 ** -decimals + 1e-9


def parse_number(text: str) -> float | None:
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", text)
    if m and float(m.group(2)) != 0:
        return float(m.group(1)) / float(m.group(2))
    m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(m.group(0)) if m else None


def gear_polygon(g: Gear, steps_per_tooth: int = 8) -> list[tuple[float, float]]:
    pts = []
    root, tip = g.r - ADDENDUM, g.r_out
    for k in range(g.teeth):
        base = g.phase + 2 * math.pi * k / g.teeth
        half = math.pi / g.teeth
        # trapezoid tooth profile centred on `base`
        for frac, rad in ((-1.0, root), (-0.45, tip), (0.45, tip), (1.0, root)):
            a = base + frac * half * 0.9
            pts.append((g.x + rad * math.cos(a), g.y + rad * math.sin(a)))
    return pts


def belt_segments(a: Gear, b: Gear, crossed: bool):
    """Tangent segments of an open (external tangents) or crossed (internal tangents) belt."""
    r1, r2 = a.r - 2, b.r - 2
    dx, dy = b.x - a.x, b.y - a.y
    d = math.hypot(dx, dy)
    base = math.atan2(dy, dx)
    segs = []
    for sign in (1, -1):
        if crossed:
            ang = math.acos(min(1.0, (r1 + r2) / d))
            p1 = (a.x + r1 * math.cos(base + sign * ang), a.y + r1 * math.sin(base + sign * ang))
            p2 = (b.x - r2 * math.cos(base + sign * ang), b.y - r2 * math.sin(base + sign * ang))
        else:
            ang = math.acos(max(-1.0, min(1.0, (r1 - r2) / d)))
            p1 = (a.x + r1 * math.cos(base + sign * ang), a.y + r1 * math.sin(base + sign * ang))
            p2 = (b.x + r2 * math.cos(base + sign * ang), b.y + r2 * math.sin(base + sign * ang))
        segs.append([p1, p2])
    return segs


def seg_circle_dist(seg, c) -> float:
    (x1, y1), (x2, y2) = seg
    px, py = c
    vx, vy = x2 - x1, y2 - y1
    L = vx * vx + vy * vy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((px - x1) * vx + (py - y1) * vy) / L))
    return math.hypot(x1 + t * vx - px, y1 + t * vy - py)
