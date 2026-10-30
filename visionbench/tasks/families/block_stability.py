"""Intuitive statics: will a stack of offset blocks stand, and if not, where does it fail?

For every interface, the combined centre of mass of all blocks above it must
lie over the contact region with the block below. Edge positions sit on a
half-unit ruler and masses are printed (or implied by equal density), so the
answer is exact. Configurations where any centre of mass falls within 0.15
units of a contact edge are rejected as too close to call. Stable and unstable
towers are balanced roughly 50/50.
"""

from __future__ import annotations

from visionbench.core.types import AnswerSpec
from visionbench.tasks.base import TaskFamily, register
from visionbench.tasks.render import canvas, finish

#            d1 d2 d3 d4 d5
N_BLOCKS = [0, 3, 4, 5, 6, 7]
LABELED_MASS = [0, 0, 0, 1, 1, 1]
MARGIN = 0.15


def analyse(blocks):
    """blocks: list of dict(l, r, m) bottom->top. Returns (first failing block index (0-based, the support), margins)."""
    margins = []
    fail = None
    for k in range(len(blocks) - 1):
        above = blocks[k + 1:]
        M = sum(b["m"] for b in above)
        com = sum(b["m"] * (b["l"] + b["r"]) / 2 for b in above) / M
        lo = max(blocks[k]["l"], blocks[k + 1]["l"])
        hi = min(blocks[k]["r"], blocks[k + 1]["r"])
        m = min(com - lo, hi - com)
        margins.append({"interface_on_block": k + 1, "com": round(com, 4), "contact": (lo, hi), "margin": round(m, 4)})
        if m < 0 and fail is None:
            fail = k
    return fail, margins


@register
class BlockStability(TaskFamily):
    name = "block_stability"
    description = "Decide whether a stack of offset blocks is stable and where it first fails (statics)."
    skills = ["physical_reasoning", "measurement"]

    def build(self, rng, style, difficulty, noise_rng):
        n = N_BLOCKS[difficulty]
        want_stable = bool(rng.random() < 0.45)
        for _ in range(5000):
            blocks = []
            x = float(rng.integers(4, 9))
            for i in range(n):
                w = float(rng.integers(4, 13)) / 2          # 2.0 .. 6.0 units
                if i == 0:
                    l = x
                else:
                    pl, pr = blocks[-1]["l"], blocks[-1]["r"]
                    l = float(rng.integers(int((pl - w + 0.5) * 2), int((pr - 0.5) * 2) + 1)) / 2
                m = int(rng.integers(1, 10)) if LABELED_MASS[difficulty] else w  # equal density -> mass ∝ width
                blocks.append({"l": l, "r": l + w, "m": float(m)})
            if min(b["l"] for b in blocks) < 0.5 or max(b["r"] for b in blocks) > 19.5:
                continue
            fail, margins = analyse(blocks)
            if any(abs(mg["margin"]) < MARGIN for mg in margins):
                continue
            if (fail is None) == want_stable:
                break
        else:  # pragma: no cover
            raise RuntimeError("no clear-cut tower")
        answer = "stable" if fail is None else str(fail + 1)
        img = self._render(blocks, bool(LABELED_MASS[difficulty]), style)
        img = finish(img, style, noise_rng)
        mass_txt = ("Each block's mass is printed on it." if LABELED_MASS[difficulty] else
                    "All blocks are made of the same uniform material and have the same height and depth, so a "
                    "block's mass is proportional to its width.")
        prompt = ("The side view shows rigid rectangular blocks stacked on a table, numbered from 1 (bottom) upward. "
                  "Edges line up with the ruler marks (every half unit). The blocks are glued to nothing: each block "
                  f"simply rests on the one below. {mass_txt} Ignore friction limits; a set of blocks tips over if "
                  "its combined centre of mass is not above the region where it touches the block below. "
                  "Is the tower stable as drawn? If it is, answer 'stable'. If not, give the number of the lowest "
                  "block from which the blocks above it would tip off.")
        return {
            "prompt": prompt,
            "answer": AnswerSpec(kind="text", value=answer, format_hint="'stable' or a block number"),
            "images": [img],
            "facts": {"blocks": blocks, "interfaces": margins, "first_failure_block": None if fail is None else fail + 1},
        }

    def _render(self, blocks, labeled, style):
        unit = 50
        bh = 62
        pad = 50
        W = pad * 2 + 20 * unit
        H = pad * 2 + bh * len(blocks) + 100
        img, draw = canvas(W, H, style)
        ground = H - pad - 60
        draw.rectangle([pad - 20, ground, W - pad + 20, ground + 14], fill=(120, 110, 100))
        # ruler
        f = style.font(15)
        for k in range(0, 41):
            x = pad + k * unit / 2
            tall = k % 2 == 0
            draw.line([(x, ground + 16), (x, ground + (34 if tall else 24))], fill=style.ink, width=2 if tall else 1)
            if tall:
                draw.text((x, ground + 46), str(k // 2), font=f, fill=style.ink, anchor="mm")
        fills = [(198, 160, 110), (170, 190, 215), (190, 215, 170), (225, 180, 180), (210, 195, 230),
                 (230, 215, 160), (175, 210, 210)]
        for i, b in enumerate(blocks):
            y1 = ground - i * bh
            y0 = y1 - bh
            x0, x1 = pad + b["l"] * unit, pad + b["r"] * unit
            draw.rectangle([x0, y0, x1, y1], fill=fills[i % len(fills)], outline=style.ink, width=style.lw(2))
            label = f"{i + 1}" + (f"  ({int(b['m'])} kg)" if labeled else "")
            draw.text(((x0 + x1) / 2, (y0 + y1) / 2), label, font=style.font(18, bold=True), fill=style.ink,
                      anchor="mm")
        return img
