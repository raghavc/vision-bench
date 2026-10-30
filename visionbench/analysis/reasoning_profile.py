"""Aggregate reasoning critiques into per-agent reasoning profiles and head-to-head analyses."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def _rate(xs: list[bool]) -> dict[str, float] | None:
    if not xs:
        return None
    p = sum(xs) / len(xs)
    return {"rate": p, "se": math.sqrt(p * (1 - p) / len(xs)), "n": len(xs)}


def _mean(xs: list[float | None]) -> dict[str, float] | None:
    v = [x for x in xs if x is not None]
    if not v:
        return None
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / max(1, len(v) - 1))
    return {"mean": m, "se": sd / math.sqrt(len(v)), "n": len(v)}


def _dist(xs: list[str | None]) -> dict[str, float]:
    xs = [x for x in xs if x]
    c = Counter(xs)
    return {k: v / len(xs) for k, v in c.most_common()} if xs else {}


def profiles(run_dir: Path) -> dict[str, Any] | None:
    recs = [r for r in _jsonl(run_dir / "critiques.jsonl") if "metrics" in r]
    if not recs:
        return None
    by_subj: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_subj[f"{r['model']} [{r['mode']}]"].append(r)
    out: dict[str, Any] = {"subjects": {}, "head_to_head": [], "n": len(recs)}
    for label, rs in sorted(by_subj.items()):
        m = [r["metrics"] for r in rs]
        correct = [x for x, r in zip(m, rs) if r["outcome"] == "correct"]
        failed = [x for x, r in zip(m, rs) if r["outcome"] != "correct"]
        prof: dict[str, Any] = {
            "n": len(rs),
            "accuracy": _rate([r["outcome"] == "correct" for r in rs]),
            "sound_success": _rate([x["sound_success"] for x in m]),
            "unsupported_share_of_correct": _rate([x["unsupported_success"] for x in correct]),
            "key_fact_recall": _mean([x["key_fact_recall"] for x in m]),
            "key_fact_error_rate": _mean([x["key_fact_error_rate"] for x in m]),
            "perception_precision": _mean([x["perception_precision"] for x in m]),
            "hallucinated_claims_per_attempt": _mean([float(x["hallucinated_claims"]) for x in m]),
            "reasoning_valid": _rate([x["reasoning_valid"] for x in m]),
            "self_verification": _rate([x["self_verification"] for x in m]),
            "accuracy_when_self_verified": _rate([r["outcome"] == "correct" for x, r in zip(m, rs) if x["self_verification"]]),
            "accuracy_when_not_self_verified": _rate([r["outcome"] == "correct" for x, r in zip(m, rs) if not x["self_verification"]]),
            "strategy": _dist([x["strategy"] for x in m]),
            "efficiency": _dist([x["efficiency"] for x in m]),
            "failure_locus": _dist([x["failure_locus"] for x in failed]),
            "error_types": _dist([x["error_type"] for x in failed]),
        }
        fam: dict[str, dict[str, Any]] = {}
        for f in sorted({r["family"] for r in rs}):
            fr = [r for r in rs if r["family"] == f]
            fm = [r["metrics"] for r in fr]
            fam[f] = {"n": len(fr),
                      "accuracy": sum(r["outcome"] == "correct" for r in fr) / len(fr),
                      "sound_success": sum(x["sound_success"] for x in fm) / len(fm),
                      "key_fact_recall": (_mean([x["key_fact_recall"] for x in fm]) or {}).get("mean"),
                      "key_fact_error_rate": (_mean([x["key_fact_error_rate"] for x in fm]) or {}).get("mean"),
                      "top_error": Counter(x["error_type"] for x, r in zip(fm, fr)
                                           if r["outcome"] != "correct").most_common(1)}
        prof["by_family"] = fam
        g = [r["grounding"] for r in rs if r.get("grounding") and r["grounding"].get("evidence_coverage") is not None]
        if g:
            full = [r["outcome"] == "correct" for r in rs if (r.get("grounding") or {}).get("evidence_coverage") == 1.0]
            part = [r["outcome"] == "correct" for r in rs
                    if (r.get("grounding") or {}).get("evidence_coverage") is not None
                    and r["grounding"]["evidence_coverage"] < 1.0]
            prof["evidence_grounding"] = {"mean_coverage": sum(x["evidence_coverage"] for x in g) / len(g),
                                          "accuracy_full_coverage": _rate(full), "accuracy_partial_coverage": _rate(part),
                                          "mean_crops": sum(x["n_crops"] for x in g) / len(g)}
        prof["exemplars"] = _exemplars(rs)
        out["subjects"][label] = prof

    # head-to-head: when A is right and B is wrong on the same item, why did B fail?
    labels = sorted(by_subj)
    idx = {lab: {r["uid"]: r for r in rs if r["sample"] == 0} for lab, rs in by_subj.items()}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            shared = set(idx[a]) & set(idx[b])
            a_only = [u for u in shared if idx[a][u]["outcome"] == "correct" and idx[b][u]["outcome"] != "correct"]
            b_only = [u for u in shared if idx[b][u]["outcome"] == "correct" and idx[a][u]["outcome"] != "correct"]
            out["head_to_head"].append({
                "a": a, "b": b, "shared": len(shared), "a_wins": len(a_only), "b_wins": len(b_only),
                "why_b_failed": _dist([idx[b][u]["metrics"]["error_type"] for u in a_only]),
                "why_a_failed": _dist([idx[a][u]["metrics"]["error_type"] for u in b_only]),
                "a_wins_by_family": dict(Counter(idx[a][u]["family"] for u in a_only).most_common(5)),
                "b_wins_by_family": dict(Counter(idx[b][u]["family"] for u in b_only).most_common(5)),
            })
    return out


def _exemplars(rs: list[dict], k: int = 3) -> dict[str, list[dict]]:
    def card(r):
        j = r["judge"]
        return {"uid": r["uid"], "family": r["family"], "difficulty": r["difficulty"], "outcome": r["outcome"],
                "error_type": j.get("error_type"), "strategy": j.get("strategy"), "critique": j.get("critique"),
                "strengths": j.get("strengths"), "weaknesses": j.get("weaknesses"),
                "first_error_step": j.get("first_error_step"),
                "key_fact_recall": r["metrics"]["key_fact_recall"]}
    fails = [r for r in rs if r["outcome"] != "correct"]
    # most instructive failures: varied error types, highest-confidence misreads first
    seen, worst = set(), []
    for r in sorted(fails, key=lambda r: -(r["metrics"]["key_fact_error_rate"] or 0)):
        et = r["metrics"]["error_type"]
        if et in seen:
            continue
        seen.add(et)
        worst.append(card(r))
        if len(worst) >= k:
            break
    unsupported = [card(r) for r in rs if r["metrics"]["unsupported_success"]][:k]
    sound = [card(r) for r in sorted(rs, key=lambda r: -(r["difficulty"] or 0)) if r["metrics"]["sound_success"]][:2]
    return {"instructive_failures": worst, "unsupported_successes": unsupported, "best": sound}
