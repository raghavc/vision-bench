"""Judge meta-evaluation: how reliable is the LLM judge, measured against programmatic truth?

Programmatically verified attempts give free gold labels. We run the
answer-equivalence judge on a stratified sample (equal correct/incorrect) and
report accuracy, sensitivity, specificity, Cohen's kappa and agreement on
the "incorrect" class separately. A lenient judge can look good on overall
agreement while waving through wrong answers.
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any

from visionbench.config import Settings
from visionbench.core.types import Attempt, TaskInstance
from visionbench.llm import Claude, RateGate
from visionbench.verifiers.judge import Judges


def _jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


async def judge_audit(run_dir: Path, settings: Settings, n: int = 150, seed: int = 0) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    items = {}
    for line in (Path(manifest["dataset"]) / "items.jsonl").read_text().splitlines():
        it = TaskInstance.from_record(json.loads(line))
        items[it.uid] = it
    attempts = {Attempt.from_record(r).key: Attempt.from_record(r) for r in _jsonl(run_dir / "attempts.jsonl")}
    verdicts = [v for v in _jsonl(run_dir / "verdicts.jsonl")
                if v["verifier"] == "programmatic" and v["outcome"] in ("correct", "incorrect")
                and v["attempt_key"] in attempts and attempts[v["attempt_key"]].final_text]
    rnd = random.Random(seed)
    pos = [v for v in verdicts if v["outcome"] == "correct"]
    neg = [v for v in verdicts if v["outcome"] == "incorrect"]
    rnd.shuffle(pos)
    rnd.shuffle(neg)
    sample = pos[: n // 2] + neg[: n - min(len(pos), n // 2)]
    judges = Judges(Claude(settings), settings)
    gate = RateGate(int(settings.run.get("concurrency", 8)))

    async def one(v):
        a = attempts[v["attempt_key"]]
        async with gate:
            out = await judges.equivalent(items[a.uid], a.final_text)
        return v, out

    results = await asyncio.gather(*[one(v) for v in sample])
    tp = fp = tn = fn = 0
    disagreements = []
    for v, out in results:
        gold = v["outcome"] == "correct"
        pred = bool(out.get("equivalent"))
        tp += gold and pred
        tn += (not gold) and (not pred)
        fp += (not gold) and pred
        fn += gold and (not pred)
        if gold != pred:
            disagreements.append({"uid": v["uid"], "gold": gold, "judge": pred, "parsed": v.get("parsed"),
                                  "judge_extracted": out.get("extracted_final_answer"),
                                  "judge_analysis": out.get("analysis", "")[:400]})
    total = max(1, tp + tn + fp + fn)
    acc = (tp + tn) / total
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / total ** 2
    kappa = (acc - pe) / (1 - pe) if pe < 1 else float("nan")
    return {"n": total, "accuracy": acc, "cohen_kappa": kappa,
            "sensitivity_on_correct": tp / max(1, tp + fn),
            "specificity_on_incorrect": tn / max(1, tn + fp),
            "false_accept_rate": fp / max(1, tn + fp),
            "judge_model": judges.model, "judge_usage": judges.usage.__dict__,
            "disagreements": disagreements}
