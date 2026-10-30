"""Multi-hop quantitative reasoning over synthetic charts.

Data values sit exactly on gridlines, so an exact answer exists and can be
recovered from the pixels. Questions need several readings plus arithmetic or
comparison. Harder levels drop value labels, add series, use a secondary
y-axis (a common source of misattribution) or stacked bars (segment heights
have to be computed by subtraction). Ties are resampled away, so every
argmax/condition has one unambiguous resolution.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from visionbench.core.types import AnswerSpec  # noqa: E402
from visionbench.tasks.base import TaskFamily, register  # noqa: E402
from visionbench.tasks.render import finish  # noqa: E402

SERIES_NAMES = ["Atlas", "Borealis", "Cobalt", "Dune", "Ember", "Fjord"]
CAT_SETS = [
    [str(y) for y in range(2015, 2026)],
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    ["North", "South", "East", "West", "Central", "Coastal", "Highland", "Delta"],
]
COLORS = ["#1f6fd1", "#e0452b", "#2e9e4f", "#f2a516", "#8a4fc2", "#5a5a5a"]

#            d1       d2       d3      d4      d5
KIND = [None, "bars", "bars", "lines", "dual", "stacked"]
N_SERIES = [0, 2, 3, 4, 2, 4]
N_CATS = [0, 5, 6, 8, 7, 6]
LABELS = [0, 1, 0, 0, 0, 0]


@register
class ChartReasoning(TaskFamily):
    name = "chart_reasoning"
    description = "Multi-hop arithmetic over values read precisely from charts."
    skills = ["chart_reading", "arithmetic", "multi_hop"]

    def build(self, rng, style, difficulty, noise_rng):
        kind = KIND[difficulty]
        cats_all = CAT_SETS[int(rng.integers(len(CAT_SETS)))]
        n_c = N_CATS[difficulty]
        start = int(rng.integers(0, len(cats_all) - n_c + 1))
        cats = cats_all[start:start + n_c]
        names = list(rng.choice(SERIES_NAMES, size=N_SERIES[difficulty], replace=False))
        for _ in range(500):
            data, scale = self._data(rng, kind, len(names), n_c)
            # overlapping markers would hide a value: no two line series may coincide at a category
            if kind == "lines" and any(len(set(col)) < len(col) for col in data.T.tolist()):
                continue
            q = self._question(rng, kind, names, cats, data)
            if q is not None:
                break
        else:  # pragma: no cover
            raise RuntimeError("no tie-free question")
        prompt_q, answer, kind_ans, steps = q
        img = self._render(kind, names, cats, data, scale, style, difficulty)
        img = finish(img, style, noise_rng)
        prompt = (f"Use the chart to answer. Read values as precisely as the gridlines allow (all data values lie "
                  f"exactly on gridlines). {prompt_q}")
        spec = AnswerSpec(kind=kind_ans, value=answer, tolerance=1e-6 if kind_ans == "number" else 0.0,
                          rel_tolerance=0.0,
                          format_hint="a single number" if kind_ans in ("number", "integer") else "a single category name")
        return {
            "prompt": prompt,
            "answer": spec,
            "images": [img],
            "facts": {"chart_kind": kind, "values": {n: dict(zip(cats, map(int, data[i]))) for i, n in enumerate(names)},
                      "reasoning_steps": steps},
        }

    # ------------------------------------------------------------------ data
    def _data(self, rng, kind, n_s, n_c):
        if kind == "bars":
            return rng.integers(1, 9, size=(n_s, n_c)) * 5, {"major": 5, "minor": 5, "max": 45}
        if kind == "lines":
            return rng.integers(2, 39, size=(n_s, n_c)), {"major": 5, "minor": 1, "max": 40}
        if kind == "dual":
            left = rng.integers(2, 19, size=n_c) * 10        # bars, left axis 0..200 step 10
            right = rng.integers(1, 20, size=n_c) * 0.5        # line, right axis 0..10 step 0.5
            return np.stack([left, right]), {"major": 50, "minor": 10, "max": 200, "rmax": 10, "rminor": 0.5}
        if kind == "stacked":
            return rng.integers(1, 9, size=(n_s, n_c)) * 2, {"major": 10, "minor": 2, "max": 70}
        raise ValueError(kind)

    # ------------------------------------------------------------- questions
    def _question(self, rng, kind, names, cats, data):
        n_s, n_c = data.shape
        def tie(v):  # True if the maximum is not unique
            v = np.asarray(v)
            return (v == v.max()).sum() > 1
        if kind == "dual":
            bar, line = data
            if tie(line) or tie(bar):
                return None
            t = int(rng.integers(3))
            if t == 0:
                i = int(np.argmax(line))
                return (f"In the {self._cat_word(cats)} where the {names[1]} line (right axis) is highest, what is the "
                        f"{names[0]} bar value (left axis)?", int(bar[i]), "integer",
                        [f"argmax {names[1]} = {cats[i]}", f"{names[0]}[{cats[i]}] = {bar[i]}"])
            if t == 1:
                i = int(np.argmax(bar))
                return (f"What is the {names[1]} value (right axis) in the {self._cat_word(cats)} where the {names[0]} "
                        f"bars (left axis) are highest?", float(line[i]), "number",
                        [f"argmax {names[0]} = {cats[i]}", f"{names[1]}[{cats[i]}] = {line[i]}"])
            ratio = bar / line
            if tie(ratio):
                return None
            i = int(np.argmax(ratio))
            return (f"For which {self._cat_word(cats)} is the ratio of {names[0]} (left axis) to {names[1]} "
                    f"(right axis) the largest?", cats[i], "text",
                    [f"ratios = {dict(zip(cats, np.round(ratio, 3)))}"])
        if kind == "stacked":
            totals = data.sum(0)
            t = int(rng.integers(3))
            s = int(rng.integers(n_s))
            if t == 0:
                if tie(data[s]):
                    return None
                return (f"What is the largest {names[s]} segment value among all {self._cat_word(cats, True)}?",
                        int(data[s].max()), "integer", [f"{names[s]} = {data[s].tolist()}"])
            if t == 1:
                if tie(totals):
                    return None
                i = int(np.argmax(totals))
                return (f"In the {self._cat_word(cats)} with the largest total stack height, what is the {names[s]} "
                        "segment value?", int(data[s, i]), "integer",
                        [f"totals = {totals.tolist()}", f"{names[s]}[{cats[i]}] = {data[s, i]}"])
            a, b = rng.choice(n_s, 2, replace=False)
            mask = data[a] > data[b]
            if (data[a] == data[b]).any() or not mask.any() or mask.all():
                return None
            return (f"What is the sum of the total stack heights over the {self._cat_word(cats, True)} where the "
                    f"{names[a]} segment is larger than the {names[b]} segment?", int(totals[mask].sum()), "integer",
                    [f"where {names[a]}>{names[b]}: {[c for c, m in zip(cats, mask) if m]}"])
        # bars / lines
        t = int(rng.integers(4))
        a, b = (int(x) for x in rng.choice(n_s, 2, replace=False))
        if t == 0:
            mask = data[a] > data[b]
            if (data[a] == data[b]).any() or not mask.any() or mask.all():
                return None
            c = [i for i in range(n_s) if i not in (a, b)]
            s = c[0] if c else a
            return (f"What is the total of {names[s]} over the {self._cat_word(cats, True)} in which {names[a]} is "
                    f"greater than {names[b]}?", int(data[s][mask].sum()), "integer",
                    [f"mask = {[c for c, m in zip(cats, mask) if m]}", f"sum {names[s]}"])
        if t == 1:
            gap = np.abs(data[a] - data[b])
            if tie(gap):
                return None
            i = int(np.argmax(gap))
            return (f"In which {self._cat_word(cats)} is the absolute difference between {names[a]} and {names[b]} "
                    "the largest?", cats[i], "text", [f"gaps = {gap.tolist()}"])
        if t == 2:
            mean = data[a].mean()
            if np.isclose(data[a], mean).any():
                return None
            return (f"How many {self._cat_word(cats, True)} have a {names[a]} value strictly above the average "
                    f"{names[a]} value across all {self._cat_word(cats, True)}?", int((data[a] > mean).sum()), "integer",
                    [f"mean = {mean:.3f}"])
        diffs = np.diff(data[a])
        if tie(diffs) or diffs.max() <= 0:
            return None
        i = int(np.argmax(diffs))
        return (f"Between which two consecutive {self._cat_word(cats, True)} does {names[a]} increase the most? "
                f"Answer with the size of that increase.", int(diffs[i]), "integer",
                [f"diffs = {diffs.tolist()}", f"{cats[i]}->{cats[i + 1]}"])

    @staticmethod
    def _cat_word(cats, plural=False):
        if cats[0].isdigit():
            return "years" if plural else "year"
        if cats[0] in ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"):
            return "months" if plural else "month"
        return "regions" if plural else "region"

    # ---------------------------------------------------------------- render
    def _render(self, kind, names, cats, data, scale, style, difficulty):
        bg = "#%02x%02x%02x" % style.background
        plt.rcParams["font.family"] = style.font_family
        fig, ax = plt.subplots(figsize=(11, 6.6), dpi=100, facecolor=bg)
        ax.set_facecolor(bg)
        x = np.arange(len(cats))
        cols = COLORS[style.hue_shift:] + COLORS[:style.hue_shift]
        if kind == "bars":
            w = 0.8 / len(names)
            for i, n in enumerate(names):
                bars = ax.bar(x - 0.4 + w / 2 + i * w, data[i], w, label=n, color=cols[i], zorder=3)
                if LABELS[difficulty]:
                    ax.bar_label(bars, fontsize=9)
        elif kind == "lines":
            markers = ["o", "s", "^", "D", "v", "P"]
            for i, n in enumerate(names):
                ax.plot(x, data[i], marker=markers[i], label=n, color=cols[i], lw=2 * style.line_scale, zorder=3)
        elif kind == "stacked":
            bottom = np.zeros(len(cats))
            for i, n in enumerate(names):
                ax.bar(x, data[i], 0.6, bottom=bottom, label=n, color=cols[i], edgecolor="white", zorder=3)
                bottom += data[i]
        elif kind == "dual":
            ax.bar(x, data[0], 0.55, label=f"{names[0]} (left axis)", color=cols[0], zorder=3)
            ax2 = ax.twinx()
            ax2.plot(x, data[1], marker="o", color=cols[1], lw=2.4, label=f"{names[1]} (right axis)", zorder=4)
            ax2.set_ylim(0, scale["rmax"])
            ax2.set_yticks(np.arange(0, scale["rmax"] + 1e-9, 1))
            ax2.set_yticks(np.arange(0, scale["rmax"] + 1e-9, scale["rminor"]), minor=True)
            ax2.set_ylabel(names[1], color=cols[1])
            ax2.tick_params(axis="y", colors=cols[1])
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.9)
        ax.set_ylim(0, scale["max"])
        ax.set_yticks(np.arange(0, scale["max"] + 1e-9, scale["major"]))
        ax.set_yticks(np.arange(0, scale["max"] + 1e-9, scale["minor"]), minor=True)
        ax.grid(axis="y", which="major", color="#9a9a9a", lw=0.9, zorder=0)
        ax.grid(axis="y", which="minor", color="#d4d4d4", lw=0.6, zorder=0)
        ax.set_xticks(x, cats)
        if kind != "dual":
            ax.legend(loc="upper left", framealpha=0.9, ncol=min(4, len(names)))
        ax.set_title(f"Performance by {self._cat_word(cats)}" if kind != "dual" else "Volume vs. margin", fontsize=14)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=bg)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
