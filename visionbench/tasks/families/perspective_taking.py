"""Egocentric perspective taking on a top-down map.

A robot on a floor plan faces some direction. The model reports which objects
are on the robot's left or right, or in front of or behind it, after optional
turns and moves, or in the intersection of two robots' frames. Allocentric to
egocentric transforms are a well-known spatial-reasoning gap. Objects within
15° of any decision boundary are excluded, so every answer has a clear margin.
"""

from __future__ import annotations

import math
import string

import numpy as np

from visionbench.core.types import AnswerSpec
from visionbench.tasks.base import TaskFamily, register
from visionbench.tasks.render import arrow, canvas, finish

#            d1 d2 d3 d4 d5
N_OBJ = [0, 5, 7, 8, 9, 10]
DIRS8 = [0, 0, 0, 1, 1, 1]       # heading may be diagonal
QUESTION = [None, "left_set", "front_left_set", "quadrant_map", "after_moves", "two_robots"]
MARGIN_DEG = 15
HEADINGS = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0),
            "north-east": (1, -1), "south-east": (1, 1), "south-west": (-1, 1), "north-west": (-1, -1)}


def ego(robot_xy, heading, obj_xy):
    """Return (forward, right) coordinates of obj in the robot's frame (image y grows downward)."""
    hx, hy = heading
    n = math.hypot(hx, hy)
    hx, hy = hx / n, hy / n
    dx, dy = obj_xy[0] - robot_xy[0], obj_xy[1] - robot_xy[1]
    fwd = dx * hx + dy * hy
    right = dx * (-hy) + dy * hx
    return fwd, right


LR = (0, 180)          # boundaries that matter for left/right judgments (the robot's forward axis)
FB = (90, -90)         # boundaries for in-front/behind
ALL = LR + FB


def margin_ok(f, r, bounds=ALL) -> bool:
    """True if the object is at least MARGIN_DEG away from every decision boundary that matters."""
    ang = math.degrees(math.atan2(r, f))
    return all(abs(((ang - b) + 180) % 360 - 180) > MARGIN_DEG for b in bounds) and math.hypot(f, r) > 0.9


@register
class PerspectiveTaking(TaskFamily):
    name = "perspective_taking"
    description = "Egocentric left/right/front/behind judgments from a top-down map, incl. after moves."
    skills = ["spatial_reasoning", "perspective_taking"]

    def build(self, rng, style, difficulty, noise_rng):
        G = 10
        q = QUESTION[difficulty]
        heads = list(HEADINGS) if DIRS8[difficulty] else ["north", "east", "south", "west"]
        for _ in range(2000):
            cells = rng.choice(G * G, size=N_OBJ[difficulty] + 2, replace=False)
            pos = [(int(c % G), int(c // G)) for c in cells]
            robot, robot2, objs = pos[0], pos[1], pos[2:]
            h1 = str(rng.choice(heads))
            h2 = str(rng.choice(heads))
            names = list(string.ascii_uppercase[:len(objs)])
            res = self._answer(q, rng, robot, h1, robot2, h2, objs, names, G)
            if res is not None:
                break
        else:  # pragma: no cover
            raise RuntimeError("no unambiguous layout")
        text, spec, facts = res
        img = self._render(G, robot, h1, robot2 if q == "two_robots" else None, h2, objs, names, style)
        img = finish(img, style, noise_rng)
        prompt = ("Top-down view of a room on a square grid; north is up. Objects are labeled with letters. "
                  "The robot (blue, labeled R1) faces the direction of its arrow. "
                  + ("A second robot (orange, R2) also faces along its arrow. " if q == "two_robots" else "")
                  + "'Left'/'right' and 'in front'/'behind' are always from the named robot's own point of view. "
                  + text)
        return {"prompt": prompt, "answer": spec, "images": [img], "facts": facts}

    def _answer(self, q, rng, robot, h1, robot2, h2, objs, names, G):
        hv = HEADINGS[h1]
        rel = {n: ego(robot, hv, o) for n, o in zip(names, objs)}
        bounds = {"left_set": LR, "front_left_set": ALL, "quadrant_map": ALL, "two_robots": FB}.get(q)
        if bounds and not all(margin_ok(*v, bounds) for v in rel.values()):
            return None
        if q == "left_set":
            val = sorted(n for n, (f, r) in rel.items() if r < 0)
            if not (1 <= len(val) < len(names)):
                return None
            return ("Which objects are on R1's left? List all of them.",
                    AnswerSpec(kind="set", value=val, partial_credit=True, format_hint="comma-separated letters"),
                    {"robot": robot, "heading": h1, "ego": rel})
        if q == "front_left_set":
            val = sorted(n for n, (f, r) in rel.items() if r < 0 and f > 0)
            if not (1 <= len(val) <= len(names) - 2):
                return None
            return ("Which objects are both in front of R1 and to its left? List all of them.",
                    AnswerSpec(kind="set", value=val, partial_credit=True, format_hint="comma-separated letters"),
                    {"robot": robot, "heading": h1, "ego": rel})
        if q == "quadrant_map":
            pick = sorted(rng.choice(names, size=4, replace=False).tolist())
            quad = {n: ("front" if rel[n][0] > 0 else "back") + "-" + ("right" if rel[n][1] > 0 else "left")
                    for n in pick}
            return (f"For each of the objects {', '.join(pick)}, say whether it is front-left, front-right, "
                    "back-left or back-right of R1.",
                    AnswerSpec(kind="mapping", value=quad, partial_credit=True,
                               format_hint="A=front-left, B=back-right, ..."),
                    {"robot": robot, "heading": h1, "ego": rel})
        if q == "after_moves":
            turn = str(rng.choice(["left", "right", "around"]))
            steps = int(rng.integers(2, 5))
            ang = {"left": -90, "right": 90, "around": 180}[turn]
            a = math.atan2(hv[1], hv[0]) + math.radians(ang)
            nh = (round(math.cos(a), 6), round(math.sin(a), 6))
            unit = (nh[0] / math.hypot(*nh), nh[1] / math.hypot(*nh))
            diag = abs(unit[0]) > 0.1 and abs(unit[1]) > 0.1
            step_vec = (round(unit[0] * (math.sqrt(2) if diag else 1)), round(unit[1] * (math.sqrt(2) if diag else 1)))
            new = (robot[0] + step_vec[0] * steps, robot[1] + step_vec[1] * steps)
            if not (0 <= new[0] < G and 0 <= new[1] < G) or new in objs:
                return None
            # path must not pass through objects
            path = [(robot[0] + step_vec[0] * k, robot[1] + step_vec[1] * k) for k in range(1, steps + 1)]
            if any(p in objs for p in path):
                return None
            rel2 = {n: ego(new, step_vec, o) for n, o in zip(names, objs)}
            if not all(margin_ok(*v, LR) for v in rel2.values()):
                return None
            val = sorted(n for n, (f, r) in rel2.items() if r > 0)
            if not (1 <= len(val) < len(names)):
                return None
            turn_txt = {"left": "turns 90° to its left", "right": "turns 90° to its right",
                        "around": "turns around (180°)"}[turn]
            sq = "diagonal squares" if diag else "squares"
            return (f"R1 {turn_txt}, then moves forward {steps} {sq}. After that, which objects are on R1's right? "
                    "List all of them.",
                    AnswerSpec(kind="set", value=val, partial_credit=True, format_hint="comma-separated letters"),
                    {"robot": robot, "heading": h1, "turn": turn, "steps": steps, "new_pos": new, "ego_after": rel2})
        # two robots
        rel2 = {n: ego(robot2, HEADINGS[h2], o) for n, o in zip(names, objs)}
        if not all(margin_ok(*v, LR) for v in rel2.values()):
            return None
        val = sorted(n for n in names if rel[n][0] > 0 and rel2[n][1] < 0)
        if not (1 <= len(val) <= len(names) - 2):
            return None
        return ("Which objects are in front of R1 and, at the same time, on R2's left? List all of them.",
                AnswerSpec(kind="set", value=val, partial_credit=True, format_hint="comma-separated letters"),
                {"robot1": robot, "heading1": h1, "robot2": robot2, "heading2": h2, "ego1": rel, "ego2": rel2})

    def _render(self, G, robot, h1, robot2, h2, objs, names, style):
        cell = 70
        pad = 50
        W = H = pad * 2 + G * cell
        img, draw = canvas(W, H, style)
        for k in range(G + 1):
            draw.line([(pad + k * cell, pad), (pad + k * cell, pad + G * cell)], fill=(210, 210, 210), width=1)
            draw.line([(pad, pad + k * cell), (pad + G * cell, pad + k * cell)], fill=(210, 210, 210), width=1)
        draw.rectangle([pad, pad, pad + G * cell, pad + G * cell], outline=style.ink, width=3)
        draw.text((W / 2, 22), "N ↑", font=style.font(22, bold=True), fill=style.ink, anchor="mm")

        def center(p):
            return pad + p[0] * cell + cell / 2, pad + p[1] * cell + cell / 2

        f = style.font(28, bold=True)
        for n, o in zip(names, objs):
            cx, cy = center(o)
            draw.rounded_rectangle([cx - 24, cy - 24, cx + 24, cy + 24], radius=8, fill=(236, 232, 220),
                                   outline=style.ink, width=2)
            draw.text((cx, cy), n, font=f, fill=style.ink, anchor="mm")
        for rpos, h, color, lab in ((robot, h1, (40, 100, 220), "R1"), (robot2, h2, (240, 140, 20), "R2")):
            if rpos is None:
                continue
            cx, cy = center(rpos)
            hx, hy = HEADINGS[h]
            n = math.hypot(hx, hy)
            draw.ellipse([cx - 22, cy - 22, cx + 22, cy + 22], fill=color)
            arrow(draw, (cx + 18 * hx / n, cy + 18 * hy / n), (cx + 62 * hx / n, cy + 62 * hy / n), color, width=7, head=20)
            draw.text((cx, cy), lab, font=style.font(18, bold=True), fill=(255, 255, 255), anchor="mm")
        return img
