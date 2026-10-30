"""Run orchestration: subjects x items x conditions x samples -> attempts -> verdicts.

Runs are resumable. Attempts and verdicts are appended to JSONL as they
complete and keyed by (model, mode, condition, uid, sample), so an
interrupted run continues where it stopped and a re-verify never re-queries
the subject.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import anthropic

from visionbench.config import Settings
from visionbench.core.types import Attempt, TaskInstance, Verdict
from visionbench.harness.agent import SubjectRunner
from visionbench.harness.mock import mock_attempt
from visionbench.llm import BudgetExceeded, Claude, RateGate
from visionbench.tasks import dataset as ds
from visionbench.tasks.base import REGISTRY, load_all
from visionbench.tasks.key_facts import crop_grounding, key_facts
from visionbench.verifiers import answers
from visionbench.verifiers.judge import Judges
from visionbench.verifiers.reasoning import attempt_view, derive


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


class Run:
    def __init__(self, settings: Settings, dataset_root: Path, run_dir: Path, *, models: list[str], mode: str,
                 conditions: list[str], samples: int, families: list[str] | None = None, limit: int | None = None,
                 per_family: int | None = None):
        self.settings = settings
        self.dataset_root = dataset_root
        self.run_dir = run_dir
        self.models = models
        self.mode = mode
        self.conditions = conditions
        self.samples = samples
        self.families = families
        self.limit = limit
        self.per_family = per_family
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "ledger").mkdir(exist_ok=True)
        self.attempts_path = run_dir / "attempts.jsonl"
        self.verdicts_path = run_dir / "verdicts.jsonl"
        self.critiques_path = run_dir / "critiques.jsonl"
        self._lock = asyncio.Lock()
        self._claude: Claude | None = None

    @property
    def claude(self) -> Claude:
        if self._claude is None:
            self._claude = Claude(self.settings)
        return self._claude

    def items(self) -> list[TaskInstance]:
        items = ds.load(self.dataset_root, self.families, self.limit)
        if self.per_family:
            # stratified subset: spread the first N items of each family across difficulty levels
            by_fam: dict[str, list[TaskInstance]] = {}
            for it in sorted(items, key=lambda i: (i.family, i.seed, i.difficulty, i.render_seed)):
                by_fam.setdefault(it.family, []).append(it)
            items = []
            for fam_items in by_fam.values():
                levels = sorted({i.difficulty for i in fam_items})
                picked, k = [], 0
                while len(picked) < min(self.per_family, len(fam_items)):
                    lvl = levels[(k * 2 + len(levels) // 2) % len(levels)]
                    cand = [i for i in fam_items if i.difficulty == lvl and i not in picked]
                    if cand:
                        picked.append(cand[0])
                    k += 1
                items += picked
        for it in items:
            it.load_images(self.dataset_root)
        return items

    def write_manifest(self, n_items: int) -> None:
        safe = {k: v for k, v in self.settings.raw.items() if k != "anthropic"}
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                    cwd=Path(__file__).resolve().parents[2]).stdout.strip()
        except OSError:
            commit = ""
        manifest = {
            "dataset": str(self.dataset_root), "models": self.models, "mode": self.mode,
            "conditions": self.conditions, "samples": self.samples, "families": self.families,
            "n_items": n_items, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": commit,
            "settings": safe, "dataset_manifest": json.loads((self.dataset_root / "manifest.json").read_text())
            if (self.dataset_root / "manifest.json").exists() else None,
        }
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    # --------------------------------------------------------------- attempts
    async def collect(self, progress: bool = True) -> None:
        items = self.items()
        self.write_manifest(len(items))
        done = {Attempt.from_record(r).key for r in _read_jsonl(self.attempts_path) if not r.get("error")
                or str(r.get("error", "")).startswith("refusal")}
        all_mock = all(m.startswith("mock:") for m in self.models)
        runner = None if all_mock else SubjectRunner(self.claude, self.settings, self.run_dir / "ledger")
        gate = RateGate(int(self.settings.run.get("concurrency", 8)))
        jobs = []
        for model in self.models:
            for inst in items:
                for cond in self.conditions:
                    if self.mode == "agentic" and cond == "caption":
                        continue
                    for k in range(self.samples):
                        key = f"{model}|{self.mode if cond == 'vision' else 'direct'}|{cond}|{inst.uid}|{k}"
                        if key not in done:
                            jobs.append((model, inst, cond, k))
        total, finished, t0 = len(jobs), 0, time.time()
        if progress:
            print(f"[run] {total} attempts to collect ({len(done)} already done)", file=sys.stderr)

        stop = {"budget": False}

        async def one(model, inst, cond, k):
            nonlocal finished
            async with gate:
                if stop["budget"]:
                    return
                try:
                    if model.startswith("mock:"):
                        att = mock_attempt(inst, model, self.mode if cond == "vision" else "direct", cond, k)
                    elif self.mode == "agentic" and cond == "vision":
                        att = await runner.run_agentic(inst, model, k)
                    else:
                        att = await runner.run_direct(inst, model, cond, k)
                except BudgetExceeded as e:
                    if not stop["budget"]:
                        print(f"[run] {e}; stopping. Completed attempts are saved; resume later.", file=sys.stderr)
                    stop["budget"] = True
                    return
                except anthropic.APIStatusError as e:
                    att = Attempt(uid=inst.uid, family=inst.family, cluster=inst.cluster, difficulty=inst.difficulty,
                                  model=model, mode=self.mode if cond == "vision" else "direct", condition=cond,
                                  sample=k, error=f"api_{e.status_code}: {str(e)[:300]}")
                    if e.status_code in (401, 403):
                        raise
                except Exception as e:  # keep the run alive; record the failure
                    att = Attempt(uid=inst.uid, family=inst.family, cluster=inst.cluster, difficulty=inst.difficulty,
                                  model=model, mode=self.mode if cond == "vision" else "direct", condition=cond,
                                  sample=k, error=f"{type(e).__name__}: {e}"[:500])
                    traceback.print_exc(limit=3, file=sys.stderr)
            async with self._lock:
                with open(self.attempts_path, "a") as f:
                    f.write(json.dumps(att.to_record(), default=str) + "\n")
                finished += 1
                if progress and (finished % max(10, total // 20) == 0 or finished == total):
                    rate = finished / max(1e-6, time.time() - t0)
                    print(f"[run] {finished}/{total} attempts ({rate:.2f}/s)", file=sys.stderr)

        await asyncio.gather(*[one(*j) for j in jobs])

    # ---------------------------------------------------------------- verify
    def latest_attempts(self) -> dict[str, Attempt]:
        out: dict[str, Attempt] = {}
        for r in _read_jsonl(self.attempts_path):
            a = Attempt.from_record(r)
            prev = out.get(a.key)
            if prev is None or (prev.error and not a.error):
                out[a.key] = a
        return out

    async def verify(self, progress: bool = True, process_judge: bool | None = None) -> None:
        load_all()
        items = {it.uid: it for it in self.items()}
        attempts = self.latest_attempts()
        have = {r["attempt_key"] for r in _read_jsonl(self.verdicts_path)}
        todo = [a for k, a in attempts.items() if k not in have and a.uid in items]
        judges = None
        needs_judge = any(items[a.uid].answer.kind == "rubric" or items[a.uid].facts.get("lenient_equivalence")
                          for a in todo) or \
            (process_judge if process_judge is not None else bool(self.settings.run.get("process_judge", False)))
        if needs_judge:
            try:
                judges = Judges(self.claude, self.settings)
                judges.cost_log = self.run_dir / "judge_costs.jsonl"
            except RuntimeError as e:  # no API key: rubric items stay unverified until one is configured
                print(f"[verify] judges unavailable ({e}); skipping rubric-graded items", file=sys.stderr)
                todo = [a for a in todo if items[a.uid].answer.kind != "rubric"]
        gate = RateGate(int(self.settings.run.get("concurrency", 8)))
        do_process = process_judge if process_judge is not None else bool(self.settings.run.get("process_judge", False))
        if progress:
            print(f"[verify] {len(todo)} attempts to verify", file=sys.stderr)

        async def one(a: Attempt):
            inst = items[a.uid]
            async with gate:
                try:
                    v = await self._verdict(a, inst, judges)
                except Exception as e:  # leave it unverified; a later `verify` retries it
                    print(f"[verify] {a.key}: {type(e).__name__}: {str(e)[:160]}", file=sys.stderr)
                    return
                if do_process and judges and v.outcome in ("incorrect", "unparseable") and a.condition == "vision":
                    try:
                        diag = await judges.diagnose(inst, a.final_text, a.thinking_summary)
                        v.error_category = diag.get("category")
                        v.details["diagnosis"] = diag
                    except anthropic.APIError as e:
                        v.details["diagnosis_error"] = str(e)[:200]
            async with self._lock:
                with open(self.verdicts_path, "a") as f:
                    f.write(json.dumps(v.to_record(), default=str) + "\n")

        await asyncio.gather(*[one(a) for a in todo])
        if judges and progress:
            print(f"[verify] judge usage: {judges.usage}", file=sys.stderr)

    async def _verdict(self, a: Attempt, inst: TaskInstance, judges: Judges | None) -> Verdict:
        if a.error:
            outcome = "refusal" if a.error.startswith("refusal") else "error"
            return Verdict(attempt_key=a.key, uid=a.uid, outcome=outcome, score=0.0, details={"error": a.error})
        spec = inst.answer
        if spec.kind == "rubric":
            if a.condition == "blind" and not self.settings.run.get("judge_blind_rubric", False):
                # an image-free answer can't locate anomalies; skip the judge spend and score it as a failure
                return Verdict(attempt_key=a.key, uid=a.uid, outcome="incorrect", score=0.0,
                               details={"note": "blind rubric item not judged"}, verifier="rubric_skipped_blind")
            response = a.raw_answer if (a.mode == "agentic" and a.raw_answer) else a.final_text
            if not response.strip():
                return Verdict(attempt_key=a.key, uid=a.uid, outcome="unparseable", score=0.0)
            g = await judges.grade_rubric(inst, response)
            return Verdict(attempt_key=a.key, uid=a.uid, outcome="correct" if g["passed"] else "incorrect",
                           score=float(g["passed"]), partial=g["score"], details=g, verifier="rubric_judge")
        if a.raw_answer is None:
            return Verdict(attempt_key=a.key, uid=a.uid, outcome="unparseable", score=0.0)
        fam = REGISTRY.get(inst.family)
        outcome, score, partial, parsed, details = answers.check(spec, a.raw_answer, fam, inst)
        verifier = "programmatic"
        if outcome != "correct" and inst.facts.get("lenient_equivalence") and judges is not None:
            # external free-form answers: reference-guided equivalence check (e.g. "1/2" vs "0.5 m")
            eq = await judges.equivalent(inst, a.final_text or a.raw_answer)
            details["programmatic_outcome"] = outcome
            details["equivalence"] = eq
            if eq.get("equivalent") is True:
                outcome, score, verifier = "correct", 1.0, "programmatic+equivalence_judge"
        return Verdict(attempt_key=a.key, uid=a.uid, outcome=outcome, score=score, partial=partial,
                       parsed=parsed, details=details, verifier=verifier)

    # -------------------------------------------------------------- critique
    async def critique(self, all_samples: bool = False, limit: int | None = None, progress: bool = True,
                       models: list[str] | None = None) -> None:
        """Reasoning critique of vision-condition attempts (sample 0 per item unless all_samples)."""
        items = {it.uid: it for it in self.items()}
        attempts = self.latest_attempts()
        verdicts = {r["attempt_key"]: r for r in _read_jsonl(self.verdicts_path)}
        have = {r["attempt_key"] for r in _read_jsonl(self.critiques_path)}
        todo = [a for k, a in attempts.items()
                if k in verdicts and k not in have and a.condition == "vision" and a.uid in items
                and not a.model.startswith(("mock:", "human:")) and (all_samples or a.sample == 0)
                and (models is None or a.model in models) and verdicts[k]["outcome"] != "error"]
        todo.sort(key=lambda a: (a.model, a.family, a.uid))
        if limit:
            todo = todo[:limit]
        judges = Judges(self.claude, self.settings)
        judges.cost_log = self.run_dir / "judge_costs.jsonl"
        gate = RateGate(int(self.settings.run.get("concurrency", 8)))
        if progress:
            print(f"[critique] {len(todo)} attempts to critique", file=sys.stderr)
        done = 0

        async def one(a: Attempt):
            nonlocal done
            inst = items[a.uid]
            v = verdicts[a.key]
            kf = key_facts(inst)
            async with gate:
                try:
                    j = await judges.critique(inst, kf, attempt_view(a), v["outcome"] == "correct")
                except anthropic.APIError as e:
                    j = {"_error": str(e)[:300]}
            rec = {"attempt_key": a.key, "uid": a.uid, "model": a.model, "mode": a.mode, "family": a.family,
                   "difficulty": a.difficulty, "sample": a.sample, "outcome": v["outcome"], "key_facts": kf,
                   "judge": j, "grounding": crop_grounding(a.trajectory, inst), "judge_model": judges.model}
            if not any(k.startswith("_") for k in j):
                rec["metrics"] = derive(j, len(kf), v["outcome"])
            async with self._lock:
                with open(self.critiques_path, "a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
                done += 1
                if progress and (done % max(5, len(todo) // 20) == 0 or done == len(todo)):
                    print(f"[critique] {done}/{len(todo)}", file=sys.stderr)

        await asyncio.gather(*[one(a) for a in todo])
        if progress:
            print(f"[critique] judge usage: {judges.usage}", file=sys.stderr)
