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
from visionbench.llm import Claude, RateGate
from visionbench.tasks import dataset as ds
from visionbench.tasks.base import REGISTRY, load_all
from visionbench.verifiers import answers
from visionbench.verifiers.judge import Judges


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


class Run:
    def __init__(self, settings: Settings, dataset_root: Path, run_dir: Path, *, models: list[str], mode: str,
                 conditions: list[str], samples: int, families: list[str] | None = None, limit: int | None = None):
        self.settings = settings
        self.dataset_root = dataset_root
        self.run_dir = run_dir
        self.models = models
        self.mode = mode
        self.conditions = conditions
        self.samples = samples
        self.families = families
        self.limit = limit
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "ledger").mkdir(exist_ok=True)
        self.attempts_path = run_dir / "attempts.jsonl"
        self.verdicts_path = run_dir / "verdicts.jsonl"
        self._lock = asyncio.Lock()
        self._claude: Claude | None = None

    @property
    def claude(self) -> Claude:
        if self._claude is None:
            self._claude = Claude(self.settings)
        return self._claude

    def items(self) -> list[TaskInstance]:
        items = ds.load(self.dataset_root, self.families, self.limit)
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

        async def one(model, inst, cond, k):
            nonlocal finished
            async with gate:
                try:
                    if model.startswith("mock:"):
                        att = mock_attempt(inst, model, self.mode if cond == "vision" else "direct", cond, k)
                    elif self.mode == "agentic" and cond == "vision":
                        att = await runner.run_agentic(inst, model, k)
                    else:
                        att = await runner.run_direct(inst, model, cond, k)
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
        needs_judge = any(items[a.uid].answer.kind == "rubric" for a in todo) or \
            (process_judge if process_judge is not None else bool(self.settings.run.get("process_judge", False)))
        if needs_judge:
            try:
                judges = Judges(self.claude, self.settings)
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
                v = await self._verdict(a, inst, judges)
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
        return Verdict(attempt_key=a.key, uid=a.uid, outcome=outcome, score=score, partial=partial,
                       parsed=parsed, details=details)
