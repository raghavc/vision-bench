"""Rough cost estimate before a run.

Image tokens use the documented approximation (width*height/750 after
resizing). Output tokens use per-effort priors that should be replaced with
measured averages from a pilot run (reported in results.json -> cost).
"""

from __future__ import annotations

from pathlib import Path

from visionbench.config import Settings
from visionbench.harness.images import fit
from visionbench.tasks import dataset as ds

OUTPUT_PRIOR = {"low": 1500, "medium": 4000, "high": 9000, "xhigh": 15000, "max": 25000}


def estimate(root: Path, settings: Settings, models: list[str], samples: int, conditions: list[str], mode: str | None):
    items = ds.load(root)
    max_side = int(settings.harness.get("max_image_side", 1568))
    img_tok = 0
    txt_tok = 0
    for it in items:
        it.load_images(root)
        for im in it.images:
            w, h = fit(im, max_side).size
            img_tok += w * h / 750
        txt_tok += len(it.prompt) / 3.5 + 150
    effort = settings.subject.get("effort", "high")
    out_per = OUTPUT_PRIOR.get(effort, 9000)
    agent_mult = 3.0 if (mode or settings.harness.get("mode")) == "agentic" else 1.0
    lines = [f"{len(items)} items, avg {img_tok / max(1, len(items)):.0f} image tokens/item, effort={effort}"]
    total = 0.0
    for m in models:
        pin, pout = settings.price(m)
        per_cond = {}
        for c in conditions:
            inp = (txt_tok + (img_tok if c == "vision" else 0)) * samples * (agent_mult if c == "vision" else 1)
            out = len(items) * samples * out_per * (agent_mult if c == "vision" else 0.7)
            per_cond[c] = (inp * pin + out * pout) / 1e6
        cost = sum(per_cond.values())
        total += cost
        lines.append(f"  {m:24s} ~${cost:,.2f}  (" + ", ".join(f"{c}: ${v:,.2f}" for c, v in per_cond.items()) + ")")
    lines.append(f"  total ~${total:,.2f} (output-token prior; refine with a pilot run)")
    return "\n".join(lines)
