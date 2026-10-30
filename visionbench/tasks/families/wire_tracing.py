"""Wire tracing: follow tangled curves from labeled left terminals to numbered right terminals.

Tests continuous visual tracking through crossings, which chain-of-thought over
a caption can't replace. At higher difficulty every wire has the same colour,
so colour can't be used as a shortcut. Layouts where two wires run parallel and
close (where even a human couldn't tell them apart) are rejected.
"""

from __future__ import annotations

import string

import numpy as np

from visionbench.core.types import AnswerSpec
from visionbench.tasks.base import TaskFamily, register
from visionbench.tasks.render import NAMED_COLORS, bezier, canvas, finish, polyline_intersections

#               d1  d2  d3  d4  d5
N_WIRES =     [0, 4,  5,  7,  9,  11]
N_CTRL =      [0, 2,  3,  3,  4,  5]
N_QUERY =     [0, 1,  1,  2,  3,  4]
SAME_COLOR =  [0, 0,  0,  1,  1,  1]


@register
class WireTracing(TaskFamily):
    name = "wire_tracing"
    description = "Trace tangled wires through crossings to their endpoints."
    skills = ["visual_tracking", "perception"]

    def build(self, rng, style, difficulty, noise_rng):
        n = N_WIRES[difficulty]
        W, H = 1100, 120 + 80 * n
        x0, x1 = 90, W - 90
        ys = np.linspace(80, H - 80, n)
        letters = list(string.ascii_uppercase[:n])
        palette = list(NAMED_COLORS.values())

        for _attempt in range(200):
            perm = rng.permutation(n)
            curves = []
            for i in range(n):
                ctrl = [(x0, ys[i]), (x0 + 60, ys[i])]
                for k in range(N_CTRL[difficulty]):
                    cx = x0 + (x1 - x0) * (k + 1) / (N_CTRL[difficulty] + 1) + rng.uniform(-60, 60)
                    ctrl.append((cx, rng.uniform(40, H - 40)))
                ctrl += [(x1 - 60, ys[perm[i]]), (x1, ys[perm[i]])]
                curves.append(bezier(ctrl, n=260))
            if self._well_separated(curves):
                break
        else:  # pragma: no cover - extremely unlikely
            raise RuntimeError("could not sample a legible layout")

        img, draw = canvas(W, H, style)
        lw = style.lw(4)
        order = rng.permutation(n)  # draw order decides which wire is on top at crossings
        for i in order:
            color = style.ink if SAME_COLOR[difficulty] else palette[(i + style.hue_shift) % len(palette)]
            draw.line(curves[i], fill=color, width=lw, joint="curve")
        f = style.font(26, bold=True)
        for i in range(n):
            draw.ellipse([x0 - 14, ys[i] - 14, x0 + 14, ys[i] + 14], outline=style.ink, width=3, fill=style.background)
            draw.text((x0 - 40, ys[i]), letters[i], font=f, fill=style.ink, anchor="mm")
            draw.ellipse([x1 - 14, ys[i] - 14, x1 + 14, ys[i] + 14], outline=style.ink, width=3, fill=style.background)
            draw.text((x1 + 40, ys[i]), str(i + 1), font=f, fill=style.ink, anchor="mm")
        img = finish(img, style, noise_rng)

        q = sorted(rng.choice(n, size=N_QUERY[difficulty], replace=False).tolist())
        mapping = {letters[i]: str(int(perm[i]) + 1) for i in q}
        crossings = {letters[i]: sum(polyline_intersections(curves[i], curves[j]) for j in range(n) if j != i)
                     for i in q}
        if len(q) == 1:
            prompt = (f"The image shows {n} wires. Each wire starts at a lettered terminal on the left and "
                      f"ends at a numbered terminal on the right. Which numbered terminal is wire "
                      f"{letters[q[0]]} connected to?")
            answer = AnswerSpec(kind="integer", value=int(mapping[letters[q[0]]]),
                                format_hint="a single terminal number, e.g. 3")
        else:
            qs = ", ".join(letters[i] for i in q)
            prompt = (f"The image shows {n} wires. Each wire starts at a lettered terminal on the left and "
                      f"ends at a numbered terminal on the right. For each of the wires {qs}, give the "
                      f"numbered terminal it connects to.")
            answer = AnswerSpec(kind="mapping", value=mapping, partial_credit=True,
                                format_hint="comma-separated letter=number pairs, e.g. A=3, C=1")
        return {
            "prompt": prompt,
            "answer": answer,
            "images": [img],
            "facts": {"full_mapping": {letters[i]: int(perm[i]) + 1 for i in range(n)},
                      "crossings_on_queried_wires": crossings, "n_wires": n,
                      "same_color": bool(SAME_COLOR[difficulty])},
        }

    @staticmethod
    def _well_separated(curves: list[list[tuple[float, float]]], tol: float = 5.0, max_run: int = 8) -> bool:
        """Reject layouts where two wires run within `tol` px of each other for a long stretch."""
        arrs = [np.array(c) for c in curves]
        for i in range(len(arrs)):
            for j in range(i + 1, len(arrs)):
                a, b = arrs[i][15:-15], arrs[j][15:-15]
                d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(axis=1)
                close = d < tol
                run = best = 0
                for c in close:
                    run = run + 1 if c else 0
                    best = max(best, run)
                if best > max_run:
                    return False
        return True
