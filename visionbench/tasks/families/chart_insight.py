"""Open-ended chart analysis graded by a rubric whose criteria are derived from ground truth.

This family exercises the rubric verifier. The model writes a free-form
analysis of a time series with injected anomalies (spikes, dips, level shifts,
gaps, slope changes). The generator knows exactly where each anomaly is, so
every rubric item is atomic, binary and objective ("identifies the level
shift near t=37, ±3"), and there is a critical item against fabricated
anomalies. The judge grades against facts, not taste.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from visionbench.core.types import AnswerSpec, RubricItem  # noqa: E402
from visionbench.tasks.base import TaskFamily, register  # noqa: E402
from visionbench.tasks.render import finish  # noqa: E402

TYPES = ["spike", "dip", "level_shift", "gap", "slope_change"]
DESC = {
    "spike": "a one-point upward spike",
    "dip": "a one-point downward dip",
    "level_shift": "an abrupt, persistent level shift",
    "gap": "a gap where data is missing",
    "slope_change": "a change in trend slope",
}
#             d1 d2 d3 d4 d5
N_ANOM = [0, 1, 2, 2, 3, 4]
NOISE = [0, 0.4, 0.6, 0.9, 1.1, 1.3]


@register
class ChartInsight(TaskFamily):
    name = "chart_insight"
    description = "Free-form anomaly analysis of a time-series chart, graded by a fact-derived rubric."
    skills = ["chart_reading", "analysis", "open_ended"]

    def build(self, rng, style, difficulty, noise_rng):
        T = 100
        t = np.arange(T)
        trend = rng.uniform(-0.04, 0.08)
        base = 20 + trend * t + 1.5 * np.sin(2 * np.pi * t / rng.uniform(18, 30))
        y = base + rng.normal(0, NOISE[difficulty], T)
        anomalies = []
        k = N_ANOM[difficulty]
        types = list(rng.choice(TYPES, size=k, replace=False))
        positions = sorted(rng.choice(np.arange(12, 88, 14), size=k, replace=False).tolist())
        positions = [int(p + rng.integers(0, 6)) for p in positions]
        mask = np.ones(T, bool)
        for typ, p in zip(types, positions):
            if typ == "spike":
                y[p] += rng.uniform(7, 11)
            elif typ == "dip":
                y[p] -= rng.uniform(7, 11)
            elif typ == "level_shift":
                y[p:] += rng.choice([-1, 1]) * rng.uniform(5, 8)
            elif typ == "gap":
                mask[p:p + int(rng.integers(4, 7))] = False
            elif typ == "slope_change":
                y[p:] += rng.choice([-1, 1]) * rng.uniform(0.25, 0.4) * (t[p:] - p)
            anomalies.append({"type": str(typ), "t": int(p)})
        img = self._render(t, y, mask, style)
        img = finish(img, style, noise_rng)
        rubric = [
            RubricItem(id=f"a{i}", criterion=(f"Identifies {DESC[a['type']]} at approximately t={a['t']} "
                                              f"(a location within ±4 of {a['t']} counts; the anomaly type must "
                                              "be described compatibly, exact wording not required)."), weight=2.0)
            for i, a in enumerate(anomalies)
        ]
        rubric.append(RubricItem(
            id="no_fabrication", critical=True, weight=2.0,
            criterion=("Does not claim any anomaly that is absent from the reference list. Ordinary noise or the "
                       "smooth periodic oscillation described as such is fine; calling them a spike, dip, shift, gap "
                       "or slope change is a fabrication.")))
        rubric.append(RubricItem(
            id="count", weight=1.0,
            criterion=f"States or clearly implies that there are exactly {len(anomalies)} anomalies in total."))
        prompt = ("The chart shows a sensor reading over time t = 0..99. Besides noise and a smooth periodic "
                  "oscillation, it contains a small number of anomalies (possible kinds: a one-point spike, a "
                  "one-point dip, an abrupt persistent level shift, a gap with missing data, or a change in trend "
                  "slope). List every anomaly you find with its kind and approximate t position. Be precise and "
                  "do not report anything you are not confident is an anomaly.")
        return {
            "prompt": prompt,
            "answer": AnswerSpec(kind="rubric", value=anomalies,
                                 format_hint="a short list, one anomaly per line: kind @ t≈position"),
            "images": [img],
            "rubric": rubric,
            "facts": {"anomalies": anomalies, "trend_per_step": round(float(trend), 4)},
        }

    def _render(self, t, y, mask, style):
        bg = "#%02x%02x%02x" % style.background
        plt.rcParams["font.family"] = style.font_family
        fig, ax = plt.subplots(figsize=(12, 5.4), dpi=100, facecolor=bg)
        ax.set_facecolor(bg)
        yy = np.where(mask, y, np.nan)
        ax.plot(t, yy, color="#1f6fd1", lw=1.8 * style.line_scale, marker="o", ms=3)
        ax.set_xticks(np.arange(0, 101, 10))
        ax.set_xticks(np.arange(0, 101, 2), minor=True)
        ax.grid(which="major", color="#a0a0a0", lw=0.8)
        ax.grid(which="minor", axis="x", color="#e0e0e0", lw=0.5)
        ax.set_xlim(-1, 100)
        ax.set_xlabel("t")
        ax.set_ylabel("reading")
        ax.set_title("Sensor S-14")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=bg)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
