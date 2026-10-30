"""Self-contained HTML report (inline SVG, no external assets) with light/dark themes.

Charts follow the dataviz method: one axis per chart, categorical hues in a
fixed order that follows the subject (never its rank), sequential single-hue
heatmap, thin marks, recessive grid, hover tooltips on every mark, and a
table next to every chart.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
       "#1c5cab", "#184f95", "#104281", "#0d366b"]

CSS = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--surface-2:#f3f2ef;--text-primary:#0b0b0b;--text-secondary:#52514e;
--text-muted:#7a7974;--grid:#e4e3df;--axis:#b9b8b2;%LIGHT%}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface-1:#1a1a19;
--surface-2:#242422;--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#94938b;--grid:#2e2e2c;--axis:#55554f;%DARK%}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--surface-2:#242422;--text-primary:#f0efec;
--text-secondary:#c3c2b7;--text-muted:#94938b;--grid:#2e2e2c;--axis:#55554f;%DARK%}
body{margin:0}
.viz-root{background:var(--surface-1);color:var(--text-primary);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",
Inter,Helvetica,Arial,sans-serif;padding:28px 36px;max-width:1240px;margin:0 auto}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:34px 0 6px}h3{font-size:15px;margin:18px 0 6px}
.sub{color:var(--text-secondary);margin:0 0 14px}.muted{color:var(--text-muted)}
table{border-collapse:collapse;width:100%;margin:8px 0 4px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--grid)}th{color:var(--text-secondary);font-weight:600}
td.num,th.num{text-align:right}.card{background:var(--surface-2);border-radius:10px;padding:14px 16px;margin:10px 0}
.kpis{display:flex;gap:14px;flex-wrap:wrap}.kpi{background:var(--surface-2);border-radius:10px;padding:12px 16px;min-width:150px}
.kpi .v{font-size:26px;font-weight:650}.kpi .l{color:var(--text-secondary);font-size:12px}
svg text{fill:var(--text-secondary);font-size:12px}svg .ink{fill:var(--text-primary)}
svg .grid{stroke:var(--grid);stroke-width:1}svg .axis{stroke:var(--axis);stroke-width:1}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin:4px 0 6px;color:var(--text-secondary);font-size:12px}
.sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
details summary{cursor:pointer;color:var(--text-secondary)}code{font-size:12px}
.mark:hover{opacity:.8}
"""


def _vars(colors: list[str]) -> str:
    return "".join(f"--series-{i + 1}:{c};" for i, c in enumerate(colors))


def esc(x: Any) -> str:
    return html.escape(str(x))


def pct(x: float | None, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{100 * x:.{digits}f}%"


def usd(x: float | None) -> str:
    return "–" if not x else f"${x:.3f}"


def secs(x: float | None) -> str:
    return "–" if x is None else f"{x:.1f}s"


def series_var(i: int) -> str:
    return f"var(--series-{i % 8 + 1})"


# ------------------------------------------------------------------ charts
def leaderboard_svg(subjects: list[dict]) -> str:
    rows = sorted(subjects, key=lambda s: -s["accuracy"]["mean"])
    W, rh, left, right = 980, 34, 300, 70
    H = rh * len(rows) + 46
    x = lambda v: left + v * (W - left - right)
    out = [f'<svg width="{W}" height="{H}" role="img" aria-label="Accuracy with 95% confidence intervals">']
    for t in range(0, 101, 10):
        out.append(f'<line class="grid" x1="{x(t / 100)}" x2="{x(t / 100)}" y1="8" y2="{H - 30}"/>')
        out.append(f'<text x="{x(t / 100)}" y="{H - 12}" text-anchor="middle">{t}%</text>')
    order = [s["label"] for s in subjects]
    for k, s in enumerate(rows):
        y = 24 + k * rh
        a = s["accuracy"]
        lo, hi = a["ci95"]
        c = series_var(order.index(s["label"]))
        tip = f'{s["label"]}: {pct(a["mean"])} (95% CI {pct(lo)}–{pct(hi)}), n={a["n_items"]} items / {a["n_clusters"]} clusters'
        out.append(f'<text class="ink" x="{left - 12}" y="{y + 4}" text-anchor="end">{esc(s["label"])}</text>')
        if lo == lo:
            out.append(f'<line x1="{x(lo)}" x2="{x(hi)}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="2" stroke-linecap="round"/>')
        out.append(f'<g class="mark"><title>{esc(tip)}</title><circle cx="{x(a["mean"])}" cy="{y}" r="6" fill="{c}" '
                   f'stroke="var(--surface-1)" stroke-width="2"/><rect x="{left}" y="{y - rh / 2}" width="{W - left - right}" '
                   f'height="{rh}" fill="transparent"/></g>')
        out.append(f'<text class="ink" x="{W - right + 8}" y="{y + 4}">{pct(a["mean"])}</text>')
    out.append("</svg>")
    return "".join(out)


def heatmap_svg(subjects: list[dict], families: list[str], key: str = "by_family") -> str:
    cw, ch, left, top = 108, 30, 190, 130
    W = left + cw * len(subjects) + 160
    H = top + ch * len(families) + 10
    out = [f'<svg width="{W}" height="{H}" role="img" aria-label="Accuracy by task family">']
    for j, s in enumerate(subjects):
        xx = left + j * cw + cw / 2
        out.append(f'<text class="ink" transform="translate({xx - 4},{top - 8}) rotate(-35)" text-anchor="start">'
                   f'<title>{esc(s["label"])}</title>{esc(s["label"][:28])}</text>')
    for i, f in enumerate(families):
        y = top + i * ch
        out.append(f'<text class="ink" x="{left - 10}" y="{y + ch / 2 + 4}" text-anchor="end">{esc(f)}</text>')
        for j, s in enumerate(subjects):
            v = s[key].get(f, {}).get("mean")
            if v is None or v != v:
                fill, label = "var(--surface-2)", "–"
            else:
                fill = SEQ[min(len(SEQ) - 1, int(v * (len(SEQ) - 1) + 0.5))]
                label = pct(v, 0)
            dark_text = v is not None and v == v and v > 0.55
            tip = f'{f} · {s["label"]}: {label}'
            out.append(f'<g class="mark"><title>{esc(tip)}</title><rect x="{left + j * cw + 1}" y="{y + 1}" width="{cw - 2}" '
                       f'height="{ch - 2}" rx="4" fill="{fill}"/>'
                       f'<text x="{left + j * cw + cw / 2}" y="{y + ch / 2 + 4}" text-anchor="middle" '
                       f'style="fill:{"#ffffff" if dark_text else "#0b0b0b"}">{label}</text></g>')
    out.append("</svg>")
    return "".join(out)


def lines_svg(series: list[tuple[str, int, list[tuple[float, float]]]], xlabels: list[str], ylabel: str,
              diag: bool = False) -> str:
    """series: (label, color_index, [(x_index_or_value, y)])."""
    W, H, left, right, top, bottom = 820, 320, 56, 220, 16, 40
    xs = lambda v: left + v * (W - left - right)
    ys = lambda v: top + (1 - v) * (H - top - bottom)
    out = [f'<svg width="{W}" height="{H}" role="img" aria-label="{esc(ylabel)}">']
    for t in range(0, 101, 20):
        out.append(f'<line class="grid" x1="{left}" x2="{W - right}" y1="{ys(t / 100)}" y2="{ys(t / 100)}"/>')
        out.append(f'<text x="{left - 8}" y="{ys(t / 100) + 4}" text-anchor="end">{t}%</text>')
    n = max(1, len(xlabels) - 1)
    for i, lab in enumerate(xlabels):
        out.append(f'<text x="{xs(i / n)}" y="{H - 16}" text-anchor="middle">{esc(lab)}</text>')
    if diag:
        out.append(f'<line x1="{xs(0)}" y1="{ys(0)}" x2="{xs(1)}" y2="{ys(1)}" stroke="var(--axis)" stroke-dasharray="4 4"/>')
    for label, ci, pts in series:
        c = series_var(ci)
        if not pts:
            continue
        P = [(xs(px / n if not diag else px), ys(py)) for px, py in pts]
        out.append(f'<polyline fill="none" stroke="{c}" stroke-width="2" points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in P)}"/>')
        for (px, py), (a, b) in zip(pts, P):
            xl = xlabels[int(px)] if not diag else f"conf {px:.2f}"
            out.append(f'<g class="mark"><title>{esc(label)} · {esc(xl)}: {pct(py)}</title>'
                       f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4.5" fill="{c}" stroke="var(--surface-1)" stroke-width="2"/>'
                       f'<circle cx="{a:.1f}" cy="{b:.1f}" r="12" fill="transparent"/></g>')
        if len(series) <= 4 and not diag:  # calibration endpoints coincide; the legend carries identity there
            a, b = P[-1]
            out.append(f'<text class="ink" x="{a + 10}" y="{b + 4}">{esc(label[:30])}</text>')
    out.append("</svg>")
    return "".join(out)


def grouped_bars_svg(groups: list[str], cats: list[str], values: list[list[float | None]]) -> str:
    W, H, left, top, bottom = 900, 300, 56, 16, 64
    gw = (W - left - 20) / max(1, len(groups))
    bw = min(34, (gw - 24) / max(1, len(cats)))
    ys = lambda v: top + (1 - v) * (H - top - bottom)
    out = [f'<svg width="{W}" height="{H}" role="img" aria-label="Vision vs controls">']
    for t in range(0, 101, 20):
        out.append(f'<line class="grid" x1="{left}" x2="{W - 20}" y1="{ys(t / 100)}" y2="{ys(t / 100)}"/>')
        out.append(f'<text x="{left - 8}" y="{ys(t / 100) + 4}" text-anchor="end">{t}%</text>')
    for g, gname in enumerate(groups):
        gx = left + g * gw + (gw - bw * len(cats) - 2 * (len(cats) - 1)) / 2
        for k, cname in enumerate(cats):
            v = values[g][k]
            if v is None or v != v:
                continue
            x0 = gx + k * (bw + 2)
            y0 = ys(v)
            h = ys(0) - y0
            r = min(4, h / 2)
            out.append(f'<g class="mark"><title>{esc(gname)} · {esc(cname)}: {pct(v)}</title>'
                       f'<path d="M{x0},{ys(0)} V{y0 + r} Q{x0},{y0} {x0 + r},{y0} H{x0 + bw - r} Q{x0 + bw},{y0} '
                       f'{x0 + bw},{y0 + r} V{ys(0)} Z" fill="{series_var(k)}"/></g>')
        out.append(f'<text class="ink" x="{left + g * gw + gw / 2}" y="{H - 40}" text-anchor="middle">{esc(gname[:28])}</text>')
    out.append(f'<line class="axis" x1="{left}" x2="{W - 20}" y1="{ys(0)}" y2="{ys(0)}"/></svg>')
    return "".join(out)


def legend(labels: list[str], start: int = 0) -> str:
    return '<div class="legend">' + "".join(
        f'<span><span class="sw" style="background:{series_var(i + start)}"></span>{esc(l)}</span>'
        for i, l in enumerate(labels)) + "</div>"


# ------------------------------------------------------------------ page
def render(results: dict[str, Any], out_path: Path, title: str = "VisionBench report") -> Path:
    subs = results["subjects"]
    fams = results.get("families", [])
    labels = [s["label"] for s in subs]
    css = CSS.replace("%LIGHT%", _vars(SERIES_LIGHT)).replace("%DARK%", _vars(SERIES_DARK))
    h: list[str] = [f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>{esc(title)}</title>"
                    f"<meta name='viewport' content='width=device-width,initial-scale=1'><style>{css}</style></head>"
                    f"<body><div class='viz-root'>"]
    h.append(f"<h1>{esc(title)}</h1><p class='sub'>Verifiable visual reasoning · {results['n_rows']} scored attempts · "
             f"{len(fams)} task families · run <code>{esc(results['run_dir'])}</code></p>")
    if subs:
        best = max(subs, key=lambda s: s["accuracy"]["mean"])
        h.append("<div class='kpis'>")
        h.append(f"<div class='kpi'><div class='v'>{pct(best['accuracy']['mean'])}</div><div class='l'>best accuracy "
                 f"({esc(best['label'])})</div></div>")
        if "blind" in best:
            h.append(f"<div class='kpi'><div class='v'>{pct(best['blind']['mean'])}</div><div class='l'>same model, "
                     "image withheld</div></div>")
        if "pass^3" in best["pass_k"]:
            h.append(f"<div class='kpi'><div class='v'>{pct(best['pass_k']['pass^3'])}</div><div class='l'>pass^3 "
                     "(all 3 attempts right)</div></div>")
        h.append(f"<div class='kpi'><div class='v'>{best['accuracy']['n_items']}</div><div class='l'>items "
                 f"({best['accuracy']['n_clusters']} independent problems)</div></div></div>")

    # leaderboard
    h.append("<h2>Leaderboard</h2><p class='sub'>Accuracy (pass@1, sample-averaged per item) with 95% cluster-bootstrap "
             "confidence intervals. Overlapping intervals mean the ranking is not resolved.</p>")
    h.append(leaderboard_svg(subs))
    h.append("<table><tr><th>Subject</th><th class='num'>Accuracy</th><th class='num'>95% CI</th><th class='num'>± SE</th>"
             "<th class='num'>Partial credit</th><th class='num'>pass^3</th><th class='num'>Blind</th>"
             "<th class='num'>Caption-only</th><th class='num'>ECE</th><th class='num'>$ / correct</th>"
             "<th class='num'>Rank (95%)</th></tr>")
    for s in sorted(subs, key=lambda s: -s["accuracy"]["mean"]):
        a = s["accuracy"]
        rk = results.get("rank_spread", {}).get(s["label"])
        h.append(f"<tr><td>{esc(s['label'])}</td><td class='num'>{pct(a['mean'])}</td>"
                 f"<td class='num'>{pct(a['ci95'][0])}–{pct(a['ci95'][1])}</td><td class='num'>{pct(a['se'])}</td>"
                 f"<td class='num'>{pct(s['partial_credit']['mean'])}</td>"
                 f"<td class='num'>{pct(s['pass_k'].get('pass^3'))}</td>"
                 f"<td class='num'>{pct(s.get('blind', {}).get('mean'))}</td>"
                 f"<td class='num'>{pct(s.get('caption', {}).get('mean'))}</td>"
                 f"<td class='num'>{pct(s['calibration'].get('ece'))}</td>"
                 f"<td class='num'>{usd(s['cost']['usd_per_correct'])}</td>"
                 f"<td class='num'>{'' if not rk else f'{rk[0]}–{rk[1]}'}</td></tr>")
    h.append("</table>")

    # family heatmap
    if fams:
        h.append("<h2>Accuracy by task family</h2><p class='sub'>Each cell is sample-averaged accuracy; darker is higher. "
                 "Hover for values.</p>")
        h.append(heatmap_svg(subs, fams))

    # difficulty curves
    diffs = sorted({int(d) for s in subs for d in s["by_difficulty"]})
    if len(diffs) > 1:
        h.append("<h2>Accuracy vs. difficulty level</h2><p class='sub'>Difficulty is a generator parameter "
                 "(more objects, deeper rule compositions, finer resolution), not post-hoc filtering against any model.</p>")
        series = []
        for i, s in enumerate(subs):
            pts = [(diffs.index(int(d)), v["mean"]) for d, v in sorted(s["by_difficulty"].items(), key=lambda kv: int(kv[0]))]
            series.append((s["label"], i, pts))
        h.append(legend(labels))
        h.append(lines_svg(series, [f"L{d}" for d in diffs], "Accuracy by difficulty"))

    # controls
    if any("blind" in s or "caption" in s for s in subs):
        h.append("<h2>Does the model need the image?</h2><p class='sub'>The same items with the image withheld (blind) or "
                 "replaced by a dense question-agnostic caption (caption-only). A valid visual item has a large "
                 "vision–blind gap. A vision–caption gap shows reasoning that a verbal description can't carry.</p>")
        cats = ["vision", "caption-only", "blind"]
        vals = [[s["accuracy"]["mean"], s.get("caption", {}).get("mean"), s.get("blind", {}).get("mean")] for s in subs]
        h.append(legend(cats))
        h.append(grouped_bars_svg(labels, cats, vals))
        h.append("<table><tr><th>Subject</th><th class='num'>Vision − blind</th><th class='num'>95% CI</th>"
                 "<th class='num'>Vision − caption</th><th class='num'>95% CI</th></tr>")
        for s in subs:
            gb = s.get("blind", {}).get("gap_vs_vision", {})
            gc = s.get("caption", {}).get("gap_vs_vision", {})
            h.append(f"<tr><td>{esc(s['label'])}</td><td class='num'>{pct(gb.get('diff'))}</td>"
                     f"<td class='num'>{pct((gb.get('ci95') or [None])[0])}–{pct((gb.get('ci95') or [None, None])[1])}</td>"
                     f"<td class='num'>{pct(gc.get('diff'))}</td>"
                     f"<td class='num'>{pct((gc.get('ci95') or [None])[0])}–{pct((gc.get('ci95') or [None, None])[1])}</td></tr>")
        h.append("</table>")

    # calibration
    cal_series = []
    for i, s in enumerate(subs):
        bins = s["calibration"].get("bins") or []
        if bins:
            cal_series.append((s["label"], i, [(b["conf"], b["acc"]) for b in bins]))
    if cal_series:
        h.append("<h2>Calibration</h2><p class='sub'>Stated confidence vs. observed accuracy (dashed line = perfect "
                 "calibration). Points below the line are overconfident.</p>")
        h.append(legend([c[0] for c in cal_series]))
        h.append(lines_svg(cal_series, ["0", "0.2", "0.4", "0.6", "0.8", "1.0"], "Calibration", diag=True))
        h.append("<table><tr><th>Subject</th><th class='num'>ECE</th><th class='num'>Brier</th>"
                 "<th class='num'>Mean confidence</th><th class='num'>Accuracy</th><th class='num'>Overconfidence</th></tr>")
        for s in subs:
            c = s["calibration"]
            h.append(f"<tr><td>{esc(s['label'])}</td><td class='num'>{pct(c.get('ece'))}</td>"
                     f"<td class='num'>{c.get('brier', float('nan')):.3f}</td><td class='num'>{pct(c.get('mean_conf'))}</td>"
                     f"<td class='num'>{pct(c.get('accuracy'))}</td><td class='num'>{pct(c.get('overconfidence'))}</td></tr>")
        h.append("</table>")

    # comparisons
    if results["comparisons"]:
        h.append("<h2>Paired comparisons</h2><p class='sub'>Differences on shared items, clustered SE, cluster-bootstrap "
                 "CI, sign-flip permutation p-value with Holm correction. MDE = minimum detectable difference at 80% power.</p>")
        h.append("<table><tr><th>A</th><th>B</th><th class='num'>A − B</th><th class='num'>95% CI</th>"
                 "<th class='num'>p (Holm)</th><th class='num'>MDE</th><th class='num'>item corr.</th></tr>")
        for c in results["comparisons"]:
            if "diff" not in c:
                continue
            h.append(f"<tr><td>{esc(c['a'])}</td><td>{esc(c['b'])}</td><td class='num'>{pct(c['diff'])}</td>"
                     f"<td class='num'>{pct(c['ci95'][0])}–{pct(c['ci95'][1])}</td>"
                     f"<td class='num'>{c['p_holm']:.4f}</td><td class='num'>{pct(c['mde_80pct_power'])}</td>"
                     f"<td class='num'>{c['item_correlation']:.2f}</td></tr>")
        h.append("</table>")

    # error attribution & outcomes
    h.append("<h2>Outcomes and failure attribution</h2>")
    h.append("<table><tr><th>Subject</th><th class='num'>correct</th><th class='num'>incorrect</th>"
             "<th class='num'>unparseable</th><th class='num'>refusal</th><th class='num'>error</th>"
             "<th>Failure attribution (process judge)</th></tr>")
    for s in subs:
        o = s["outcomes"]
        attr = ", ".join(f"{k}: {v}" for k, v in sorted(s["error_attribution"].items(), key=lambda kv: -kv[1])) or "–"
        h.append(f"<tr><td>{esc(s['label'])}</td>" + "".join(f"<td class='num'>{o.get(k, 0)}</td>" for k in
                 ("correct", "incorrect", "unparseable", "refusal", "error")) + f"<td>{esc(attr)}</td></tr>")
    h.append("</table>")

    # reasoning analysis
    ds_root = None
    try:
        ds_root = Path(json.loads((Path(results["run_dir"]) / "manifest.json").read_text())["dataset"])
    except (OSError, KeyError, ValueError):
        pass
    h.append(reasoning_section(results, ds_root))

    # reliability & robustness
    h.append("<h2>Reliability</h2><table><tr><th>Subject</th>" + "".join(
        f"<th class='num'>{k}</th>" for k in ("pass@1", "pass@3", "pass^3", "pass^5")) +
        "<th class='num'>solved on all re-renders</th><th class='num'>mean tool calls</th><th class='num'>latency p50</th></tr>")
    for s in subs:
        rc = s.get("render_consistency", {})
        h.append(f"<tr><td>{esc(s['label'])}</td>" + "".join(
            f"<td class='num'>{pct(s['pass_k'].get(k))}</td>" for k in ("pass@1", "pass@3", "pass^3", "pass^5")) +
            f"<td class='num'>{pct(rc.get('solved_on_all_renders'))}</td>"
            f"<td class='num'>{s['cost']['mean_tool_calls']:.1f}</td>"
            f"<td class='num'>{secs(s['cost']['latency_p50'])}</td></tr>")
    h.append("</table>")

    # item quality
    h.append("<h2>Item quality audit</h2>")
    bs = results.get("blind_solvable_items", [])
    h.append(f"<p class='sub'>{len(bs)} items were answered correctly at least half the time <em>without the image</em> "
             "(candidates for removal). Items flagged by IRT, where the strongest subject always fails but others "
             "succeed, may have a bad key.</p>")
    sus = results.get("irt", {}).get("suspect_items", [])
    if bs or sus:
        h.append("<table><tr><th>uid</th><th>family</th><th>flag</th><th class='num'>value</th></tr>")
        for b in bs[:20]:
            h.append(f"<tr><td><code>{b['uid']}</code></td><td>{esc(b['family'])}</td><td>blind-solvable</td>"
                     f"<td class='num'>{pct(b['blind_acc'])}</td></tr>")
        for b in sus[:20]:
            h.append(f"<tr><td><code>{b['uid']}</code></td><td></td><td>IRT suspect key</td>"
                     f"<td class='num'>{pct(b['max_other_acc'])}</td></tr>")
        h.append("</table>")
    fd = results.get("irt", {}).get("family_mean_difficulty")
    if fd:
        h.append("<h3>Family difficulty (Rasch b, higher = harder)</h3><table><tr><th>family</th><th class='num'>b</th></tr>")
        for f, b in sorted(fd.items(), key=lambda kv: -kv[1]):
            h.append(f"<tr><td>{esc(f)}</td><td class='num'>{b:+.2f}</td></tr>")
        h.append("</table>")

    # item drilldown
    h.append("<h2>Items</h2><details><summary>Show per-item scores</summary><table><tr><th>uid</th><th>family</th>"
             "<th class='num'>L</th><th>answer</th><th>scores</th></tr>")
    for it in results.get("item_table", [])[:3000]:
        sc = " · ".join(f"{esc(k)}: {v:.2f}" for k, v in it["scores"].items())
        ans = json.dumps(it["answer"], default=str)
        h.append(f"<tr><td><code>{it['uid']}</code></td><td>{esc(it['family'])}</td><td class='num'>{it['difficulty']}</td>"
                 f"<td><code>{esc(ans[:80])}</code></td><td class='muted'>{sc}</td></tr>")
    h.append("</table></details>")
    h.append("<p class='muted' style='margin-top:30px'>Generated by VisionBench. Methodology: see METHODOLOGY.md.</p>")
    h.append("</div></body></html>")
    out_path.write_text("".join(h))
    return out_path


# --------------------------------------------------------- reasoning analysis
LOCI = ["perception", "reasoning", "instruction", "no_answer"]


def stacked_svg(rows: list[tuple[str, dict[str, float]]], cats: list[str]) -> str:
    W, rh, left, right = 900, 30, 300, 20
    H = rh * len(rows) + 30
    x = lambda v: left + v * (W - left - right)
    out = [f'<svg width="{W}" height="{H}" role="img" aria-label="Failure locus by subject">']
    for t in range(0, 101, 25):
        out.append(f'<line class="grid" x1="{x(t / 100)}" x2="{x(t / 100)}" y1="4" y2="{H - 24}"/>')
        out.append(f'<text x="{x(t / 100)}" y="{H - 8}" text-anchor="middle">{t}%</text>')
    for i, (label, dist) in enumerate(rows):
        y = 8 + i * rh
        out.append(f'<text class="ink" x="{left - 10}" y="{y + 14}" text-anchor="end">{esc(label)}</text>')
        acc = 0.0
        for k, c in enumerate(cats):
            v = dist.get(c, 0.0)
            if v <= 0:
                continue
            w = v * (W - left - right)
            out.append(f'<g class="mark"><title>{esc(label)} · {esc(c)}: {pct(v)} of failures</title>'
                       f'<rect x="{x(acc) + 1}" y="{y}" width="{max(0, w - 2)}" height="{rh - 10}" rx="3" '
                       f'fill="{series_var(k)}"/></g>')
            acc += v
    out.append("</svg>")
    return "".join(out)


def _thumb(dataset_root: Path | None, uid: str) -> str:
    if dataset_root is None:
        return ""
    p = dataset_root / "images" / f"{uid}_0.png"
    if not p.exists():
        return ""
    import base64
    import io

    from PIL import Image
    im = Image.open(p).convert("RGB")
    im.thumbnail((420, 420))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82)
    return (f"<img alt='item {uid}' style='max-width:420px;border-radius:8px;border:1px solid var(--grid)' "
            f"src='data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}'>")


def _r(d: dict | None, key: str = "rate") -> str:
    return "–" if not d else pct(d.get(key))


def reasoning_section(results: dict[str, Any], dataset_root: Path | None) -> str:
    R = results.get("reasoning")
    if not R:
        return ("<h2>Reasoning analysis</h2><p class='sub'>No reasoning critiques yet. Run "
                "<code>visionbench critique &lt;run&gt;</code> to grade how each agent reasoned, not only what it "
                "answered.</p>")
    subs = R["subjects"]
    h = ["<h2>Reasoning analysis</h2><p class='sub'>A critic given the generator's ground-truth key facts audits each "
         "agent's visible reasoning: which facts it perceived correctly, whether each step is valid, which strategy it "
         "used, and whether its correct answers were actually earned. 'Sound success' means correct <em>and</em> "
         "supported by valid reasoning. 'Unsupported' correct answers are guesses or right-for-the-wrong-reason.</p>"]
    h.append("<table><tr><th>Subject</th><th class='num'>n</th><th class='num'>Accuracy</th>"
             "<th class='num'>Sound success</th><th class='num'>Unsupported (of correct)</th>"
             "<th class='num'>Key-fact recall</th><th class='num'>Key-fact misreads</th>"
             "<th class='num'>Claim precision</th><th class='num'>Valid reasoning</th>"
             "<th class='num'>Self-verifies</th></tr>")
    for label, p in subs.items():
        h.append(f"<tr><td>{esc(label)}</td><td class='num'>{p['n']}</td><td class='num'>{_r(p['accuracy'])}</td>"
                 f"<td class='num'>{_r(p['sound_success'])}</td><td class='num'>{_r(p['unsupported_share_of_correct'])}</td>"
                 f"<td class='num'>{_r(p['key_fact_recall'], 'mean')}</td><td class='num'>{_r(p['key_fact_error_rate'], 'mean')}</td>"
                 f"<td class='num'>{_r(p['perception_precision'], 'mean')}</td><td class='num'>{_r(p['reasoning_valid'])}</td>"
                 f"<td class='num'>{_r(p['self_verification'])}</td></tr>")
    h.append("</table>")

    h.append("<h3>Where failures originate</h3><p class='sub'>Perception: misread or missed a decisive fact. "
             "Reasoning: read the image correctly but inferred or computed wrongly.</p>")
    h.append(legend(LOCI))
    h.append(stacked_svg([(lab, p["failure_locus"]) for lab, p in subs.items()], LOCI))

    types = sorted({t for p in subs.values() for t in p["error_types"]},
                   key=lambda t: -sum(p["error_types"].get(t, 0) for p in subs.values()))
    if types:
        h.append("<h3>Error taxonomy (share of failures)</h3><table><tr><th>First error</th>" +
                 "".join(f"<th class='num'>{esc(l)}</th>" for l in subs) + "</tr>")
        for t in types:
            h.append(f"<tr><td>{esc(t.replace('_', ' '))}</td>" +
                     "".join(f"<td class='num'>{pct(p['error_types'].get(t))}</td>" for p in subs.values()) + "</tr>")
        h.append("</table>")

    strategies = sorted({t for p in subs.values() for t in p["strategy"]})
    h.append("<h3>Reasoning strategies</h3><table><tr><th>Strategy</th>" +
             "".join(f"<th class='num'>{esc(l)}</th>" for l in subs) + "</tr>")
    for t in strategies:
        h.append(f"<tr><td>{esc(t.replace('_', ' '))}</td>" +
                 "".join(f"<td class='num'>{pct(p['strategy'].get(t))}</td>" for p in subs.values()) + "</tr>")
    h.append("<tr><td>accuracy when it self-verified</td>" + "".join(
        f"<td class='num'>{_r(p['accuracy_when_self_verified'])}</td>" for p in subs.values()) + "</tr>")
    h.append("<tr><td>accuracy when it did not</td>" + "".join(
        f"<td class='num'>{_r(p['accuracy_when_not_self_verified'])}</td>" for p in subs.values()) + "</tr></table>")

    fams = sorted({f for p in subs.values() for f in p["by_family"]})
    if fams:
        h.append("<h3>Sound success by family</h3><p class='sub'>Cell = sound-success rate (correct with valid, "
                 "supported reasoning); hover for key-fact recall and the most common error.</p>")
        fake = [{"label": lab, "by_family": {f: {"mean": d["sound_success"]} for f, d in p["by_family"].items()}}
                for lab, p in subs.items()]
        h.append(heatmap_svg(fake, fams))
        h.append("<table><tr><th>family</th>" + "".join(f"<th>{esc(l)}: recall · misreads · top error</th>" for l in subs)
                 + "</tr>")
        for f in fams:
            cells = []
            for p in subs.values():
                d = p["by_family"].get(f)
                if not d:
                    cells.append("<td>–</td>")
                    continue
                te = d["top_error"][0][0].replace("_", " ") if d["top_error"] else "–"
                cells.append(f"<td>{pct(d['key_fact_recall'], 0)} · {pct(d['key_fact_error_rate'], 0)} · {esc(te)}</td>")
            h.append(f"<tr><td>{esc(f)}</td>{''.join(cells)}</tr>")
        h.append("</table>")

    grounded = {l: p["evidence_grounding"] for l, p in subs.items() if p.get("evidence_grounding")}
    if grounded:
        h.append("<h3>Evidence grounding (agentic)</h3><p class='sub'>Share of gold evidence regions the agent actually "
                 "inspected with zoom, measured deterministically from its crop boxes.</p><table><tr><th>Subject</th>"
                 "<th class='num'>Evidence covered</th><th class='num'>Crops / item</th>"
                 "<th class='num'>Acc. (all evidence inspected)</th><th class='num'>Acc. (some missed)</th></tr>")
        for l, g in grounded.items():
            h.append(f"<tr><td>{esc(l)}</td><td class='num'>{pct(g['mean_coverage'])}</td>"
                     f"<td class='num'>{g['mean_crops']:.1f}</td><td class='num'>{_r(g['accuracy_full_coverage'])}</td>"
                     f"<td class='num'>{_r(g['accuracy_partial_coverage'])}</td></tr>")
        h.append("</table>")

    if R["head_to_head"]:
        h.append("<h3>Head-to-head: why the loser failed</h3><table><tr><th>A</th><th>B</th><th class='num'>A right, B wrong</th>"
                 "<th>B's errors there</th><th class='num'>B right, A wrong</th><th>A's errors there</th></tr>")
        for c in R["head_to_head"]:
            fmt = lambda d: ", ".join(f"{k.replace('_', ' ')} {pct(v, 0)}" for k, v in list(d.items())[:3]) or "–"
            h.append(f"<tr><td>{esc(c['a'])}</td><td>{esc(c['b'])}</td><td class='num'>{c['a_wins']}</td>"
                     f"<td>{esc(fmt(c['why_b_failed']))}</td><td class='num'>{c['b_wins']}</td>"
                     f"<td>{esc(fmt(c['why_a_failed']))}</td></tr>")
        h.append("</table>")

    h.append("<h3>Critiques of individual attempts</h3>")
    for label, p in subs.items():
        ex = p["exemplars"]
        h.append(f"<details><summary><b>{esc(label)}</b>: {len(ex['instructive_failures'])} instructive failures, "
                 f"{len(ex['unsupported_successes'])} unsupported successes, {len(ex['best'])} strong solutions</summary>")
        for kind, title in (("instructive_failures", "Failure"), ("unsupported_successes", "Correct but unsupported"),
                            ("best", "Sound solution")):
            for c in ex[kind]:
                h.append(f"<div class='card'><b>{title}</b> · {esc(c['family'])} L{c['difficulty']} · "
                         f"<code>{c['uid']}</code> · strategy: {esc((c['strategy'] or '').replace('_', ' '))}"
                         + (f" · first error: {esc((c['error_type'] or '').replace('_', ' '))}" if c['error_type'] not in (None, 'none') else "")
                         + f"<div style='display:flex;gap:16px;margin-top:8px;flex-wrap:wrap'>{_thumb(dataset_root, c['uid'])}"
                         f"<div style='flex:1;min-width:300px'><p>{esc(c['critique'])}</p>"
                         + (f"<p class='muted'>First wrong step: {esc(c['first_error_step'])}</p>" if c['first_error_step'] else "")
                         + f"<p><b>Strengths.</b> {esc(c['strengths'])}</p><p><b>Weaknesses.</b> {esc(c['weaknesses'])}</p>"
                         f"</div></div></div>")
        h.append("</details>")
    return "".join(h)
