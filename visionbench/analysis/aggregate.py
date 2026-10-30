"""Join attempts, verdicts and items into a results document (results.json)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from visionbench.analysis import stats
from visionbench.config import Settings
from visionbench.core.types import Attempt, TaskInstance


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def build_rows(run_dir: Path, settings: Settings) -> tuple[list[stats.Row], dict[str, TaskInstance]]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    dataset_root = Path(manifest["dataset"])
    items = {}
    with open(dataset_root / "items.jsonl") as f:
        for line in f:
            it = TaskInstance.from_record(json.loads(line))
            items[it.uid] = it
    attempts: dict[str, Attempt] = {}
    for r in _jsonl(run_dir / "attempts.jsonl"):
        a = Attempt.from_record(r)
        prev = attempts.get(a.key)
        if prev is None or (prev.error and not a.error):
            attempts[a.key] = a
    verdicts = {v["attempt_key"]: v for v in _jsonl(run_dir / "verdicts.jsonl")}
    rows = []
    for key, a in attempts.items():
        v = verdicts.get(key)
        if v is None or a.uid not in items:
            continue
        if v["outcome"] == "error":  # infrastructure failures (API/network) are retried, never scored
            continue
        pin, pout = settings.price(a.model)
        rows.append(stats.Row(
            model=a.model, mode=a.mode, condition=a.condition, uid=a.uid, cluster=a.cluster, family=a.family,
            difficulty=a.difficulty, sample=a.sample, score=float(v["score"]), partial=v.get("partial"),
            outcome=v["outcome"], confidence=a.confidence, cost=a.usage.cost(pin, pout),
            output_tokens=a.usage.output_tokens, latency=a.latency_s, tool_calls=a.tool_calls,
            error_category=v.get("error_category")))
    return rows, items


def _subset(rows, **kw):
    return [r for r in rows if all(getattr(r, k) == v for k, v in kw.items())]


def aggregate(run_dir: Path, settings: Settings) -> dict[str, Any]:
    rows, items = build_rows(run_dir, settings)
    subjects = sorted({(r.model, r.mode) for r in rows if r.condition == "vision"})
    out: dict[str, Any] = {"run_dir": str(run_dir), "subjects": [], "comparisons": [], "n_rows": len(rows)}
    vision_by_subject = {}
    for model, mode in subjects:
        vis = _subset(rows, model=model, mode=mode, condition="vision")
        vision_by_subject[f"{model} [{mode}]"] = vis
        s: dict[str, Any] = {"model": model, "mode": mode, "label": f"{model} [{mode}]"}
        s["accuracy"] = stats.summarize(vis)
        s["partial_credit"] = stats.summarize(vis, value=lambda r: r.partial if r.partial is not None else r.score)
        s["pass_k"] = stats.pass_k_table(vis)
        s["outcomes"] = dict(Counter(r.outcome for r in vis))
        s["by_family"] = {f: stats.summarize(_subset(vis, family=f)) for f in sorted({r.family for r in vis})}
        for f, fs in s["by_family"].items():
            chance = float(np.mean([items[u].facts.get("chance", 0.0) or 0.0
                                    for u in {r.uid for r in vis if r.family == f}]))
            fs["chance"] = chance
            fs["chance_normalized"] = (fs["mean"] - chance) / (1 - chance) if chance < 1 else None
        s["by_difficulty"] = {str(d): stats.summarize(_subset(vis, difficulty=d))
                              for d in sorted({r.difficulty for r in vis})}
        s["by_family_difficulty"] = {
            f: {str(d): stats.summarize(_subset(vis, family=f, difficulty=d))["mean"]
                for d in sorted({r.difficulty for r in vis if r.family == f})}
            for f in sorted({r.family for r in vis})}
        s["calibration"] = stats.calibration(vis)
        cost = sum(r.cost for r in vis)
        n_correct = sum(r.score for r in vis)
        lat = [r.latency for r in vis if r.latency]
        s["cost"] = {"total_usd": cost, "usd_per_attempt": cost / max(1, len(vis)),
                     "usd_per_correct": cost / n_correct if n_correct else None,
                     "mean_output_tokens": float(np.mean([r.output_tokens for r in vis])) if vis else 0,
                     "latency_p50": float(np.median(lat)) if lat else None,
                     "latency_p90": float(np.quantile(lat, 0.9)) if lat else None,
                     "mean_tool_calls": float(np.mean([r.tool_calls for r in vis])) if vis else 0}
        errs = Counter(r.error_category for r in vis if r.error_category)
        s["error_attribution"] = dict(errs)
        # controls (always run in direct mode)
        for cond in ("blind", "caption"):
            ctrl = _subset(rows, model=model, condition=cond)
            if ctrl:
                cs = stats.summarize(ctrl)
                cs["by_family"] = {f: stats.summarize(_subset(ctrl, family=f))["mean"]
                                   for f in sorted({r.family for r in ctrl})}
                cs["gap_vs_vision"] = stats.paired_compare(vis, ctrl)
                s[cond] = cs
        # re-render consistency: share of problems solved on *every* rendering (sample-averaged >= .5)
        by_cluster: dict[str, list[float]] = defaultdict(list)
        im = stats.item_means(vis)
        for u, (m, c, _) in im.items():
            by_cluster[c].append(m)
        multi = {c: v for c, v in by_cluster.items() if len(v) > 1}
        if multi:
            s["render_consistency"] = {
                "n_problems": len(multi),
                "solved_on_all_renders": float(np.mean([all(x >= 0.5 for x in v) for v in multi.values()])),
                "solved_on_any_render": float(np.mean([any(x >= 0.5 for x in v) for v in multi.values()])),
            }
        out["subjects"].append(s)

    # pairwise comparisons with Holm correction
    labels = list(vision_by_subject)
    pairs = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            c = stats.paired_compare(vision_by_subject[labels[i]], vision_by_subject[labels[j]])
            c.update({"a": labels[i], "b": labels[j]})
            pairs[f"{labels[i]} vs {labels[j]}"] = c
    adj = stats.holm({k: v["p_value"] for k, v in pairs.items() if "p_value" in v})
    for k, v in pairs.items():
        v["p_holm"] = adj.get(k)
        out["comparisons"].append(v)
    out["rank_spread"] = stats.rank_spread(vision_by_subject) if len(labels) > 1 else {}

    # item-level analysis: IRT, suspect keys, blind-solvable items
    resp: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r.condition == "vision":
            resp[(f"{r.model} [{r.mode}]", r.uid)].append(r.score)
    if resp:
        theta, b = stats.rasch(resp)
        out["irt"] = {"theta": theta,
                      "family_mean_difficulty": {
                          f: float(np.mean([b[u] for u in b if items[u].family == f]))
                          for f in sorted({items[u].family for u in b})},
                      "suspect_items": stats.suspect_items(resp, theta, b)}
    blind_item = stats.item_means(_subset(rows, condition="blind"))
    out["blind_solvable_items"] = sorted(
        [{"uid": u, "family": items[u].family, "blind_acc": m} for u, (m, _, _) in blind_item.items() if m >= 0.5],
        key=lambda d: -d["blind_acc"])[:50]
    out["families"] = sorted({r.family for r in rows})
    from visionbench.analysis.reasoning_profile import profiles
    out["reasoning"] = profiles(run_dir)
    out["item_table"] = _item_table(rows, items)
    return out


def _item_table(rows, items) -> list[dict[str, Any]]:
    per: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per[r.uid][f"{r.model} [{r.mode}|{r.condition}]"].append(r.score)
    table = []
    for uid, d in per.items():
        it = items[uid]
        table.append({"uid": uid, "family": it.family, "difficulty": it.difficulty,
                      "answer": it.answer.value if it.answer.kind != "rubric" else "(rubric)",
                      "scores": {k: float(np.mean(v)) for k, v in d.items()}})
    return sorted(table, key=lambda t: (t["family"], t["difficulty"], t["uid"]))
