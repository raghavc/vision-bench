"""LLM-as-judge verifiers, used only where programmatic checking is impossible.

Design follows the reliability findings of APEX, VisualToolBench and
ResearchRubrics:
  * atomic, binary criteria, judged one criterion per call;
  * reference-guided: the judge sees ground-truth facts from the generator;
  * k independent samples with a majority vote, and per-criterion agreement logged;
  * critical criteria gate the pass/fail outcome;
  * structured JSON output (no free-text parsing).
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections import Counter
from typing import Any

from visionbench.config import Settings
from visionbench.core.types import RubricItem, TaskInstance, Usage
from visionbench.llm import Claude

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "verdict": {"type": "string", "enum": ["MET", "NOT_MET"]},
    },
    "required": ["analysis", "verdict"],
    "additionalProperties": False,
}

EQUIV_SCHEMA = {
    "type": "object",
    "properties": {
        "extracted_final_answer": {"type": "string"},
        "analysis": {"type": "string"},
        "equivalent": {"type": "boolean"},
    },
    "required": ["extracted_final_answer", "analysis", "equivalent"],
    "additionalProperties": False,
}

PROCESS_SCHEMA = {
    "type": "object",
    "properties": {
        "first_error": {"type": "string"},
        "category": {"type": "string", "enum": [
            "perception",          # misread / missed something in the image
            "reasoning",           # read the image correctly but drew a wrong inference / computation
            "instruction_following",  # ignored a rule or the required answer format
            "incomplete",          # gave up, ran out of budget, or no final answer
            "other",
        ]},
        "explanation": {"type": "string"},
    },
    "required": ["first_error", "category", "explanation"],
    "additionalProperties": False,
}

RUBRIC_SYSTEM = (
    "You are a meticulous, impartial grader for a visual-reasoning benchmark. You judge exactly ONE rubric "
    "criterion against a candidate response. You are given reference facts produced by the program that generated "
    "the task; treat them as ground truth. Judge only what the criterion asks. Do not reward length, confidence, "
    "or style, and do not penalize terse wording. If the response is ambiguous about the criterion, the verdict is "
    "NOT_MET. Write a brief analysis, then the verdict."
)

EQUIV_SYSTEM = (
    "You check whether a candidate's final answer to a visual-reasoning question is equivalent to the reference "
    "answer. Extract the candidate's single final answer (ignore intermediate work). Equivalent means the same "
    "value or entity, allowing trivial formatting differences (units, capitalization, '37.5' vs '75/2', 'A=3, B=1' "
    "vs 'B: 1, A: 3'). It is NOT equivalent if the candidate hedges between options, gives a different value, or "
    "gives no final answer."
)

PROCESS_SYSTEM = (
    "You diagnose why a model answered a visual-reasoning task incorrectly. You are given the task, the ground-truth "
    "facts the task was generated from (what is really in the image), the correct answer, and the model's visible "
    "reasoning. Find the FIRST point where the reasoning diverges from the facts and classify it: 'perception' if "
    "the model misread, missed or hallucinated something visible in the image; 'reasoning' if its reading of the "
    "image matches the facts but its inference or arithmetic is wrong; 'instruction_following' if it ignored a rule "
    "stated in the prompt or the answer format; 'incomplete' if it never reached an answer. Be concrete."
)


def _image_blocks(instance: TaskInstance, show: bool, max_side: int) -> list[dict[str, Any]]:
    if not show:
        return []
    from visionbench.harness.images import encode_image
    return [encode_image(im, max_side)[0] for im in instance.images]


class Judges:
    def __init__(self, claude: Claude, settings: Settings):
        self.claude = claude
        j = settings.judge
        self.model = j.get("model", "claude-opus-5-5")
        self.effort = j.get("effort", "high")
        self.max_tokens = int(j.get("max_tokens", 16000))
        self.samples = int(j.get("samples", 3))
        self.fallbacks = bool(j.get("fallbacks", True))
        self.show_image = bool(j.get("show_image", True))
        self.max_side = int(settings.harness.get("max_image_side", 1568))
        self.usage = Usage()

    async def _call(self, system: str, content: list[dict[str, Any]], schema: dict[str, Any]) -> dict[str, Any]:
        res = await self.claude.create(
            model=self.model, system=system, max_tokens=self.max_tokens, effort=self.effort,
            messages=[{"role": "user", "content": content}], json_schema=schema, fallbacks=self.fallbacks,
            thinking_display="omitted",
        )
        self.usage.add(res.usage)
        if res.stop_reason == "refusal":
            return {"_refusal": res.refusal_category}
        try:
            return json.loads(res.text)
        except json.JSONDecodeError:
            return {"_unparsed": res.text[:500]}

    # ------------------------------------------------------------- rubric
    async def grade_rubric(self, instance: TaskInstance, response: str) -> dict[str, Any]:
        imgs = _image_blocks(instance, self.show_image, self.max_side)
        facts = json.dumps(instance.facts, default=str, indent=1)

        async def one(item: RubricItem) -> dict[str, Any]:
            content = imgs + [{"type": "text", "text": (
                f"<task>\n{instance.prompt}\n</task>\n\n<reference_facts>\n{facts}\n</reference_facts>\n\n"
                f"<criterion id=\"{item.id}\">\n{item.criterion}\n</criterion>\n\n"
                f"<candidate_response>\n{response}\n</candidate_response>")}]
            outs = await asyncio.gather(*[self._call(RUBRIC_SYSTEM, content, VERDICT_SCHEMA)
                                          for _ in range(self.samples)])
            votes = [o.get("verdict") for o in outs if o.get("verdict")]
            c = Counter(votes)
            met = c.get("MET", 0) > c.get("NOT_MET", 0)
            agreement = (max(c.values()) / len(votes)) if votes else 0.0
            return {"id": item.id, "met": met, "votes": dict(c), "agreement": agreement,
                    "weight": item.weight, "critical": item.critical,
                    "analysis": next((o.get("analysis") for o in outs if o.get("verdict") == ("MET" if met else "NOT_MET")), "")}

        items = await asyncio.gather(*[one(it) for it in instance.rubric])
        total_w = sum(i["weight"] for i in items) or 1.0
        score = sum(i["weight"] for i in items if i["met"]) / total_w
        critical_ok = all(i["met"] for i in items if i["critical"])
        passed = critical_ok and score >= 0.8 - 1e-9
        return {"score": score if critical_ok else 0.0, "raw_score": score, "passed": passed,
                "critical_ok": critical_ok, "items": items,
                "mean_agreement": sum(i["agreement"] for i in items) / max(1, len(items))}

    # -------------------------------------------------------- equivalence
    async def equivalent(self, instance: TaskInstance, response: str) -> dict[str, Any]:
        ref = json.dumps(instance.answer.value, default=str)
        content = [{"type": "text", "text": (
            f"<question>\n{instance.prompt}\n</question>\n<reference_answer>{ref}</reference_answer>\n"
            f"<candidate_response>\n{response[-6000:]}\n</candidate_response>")}]
        return await self._call(EQUIV_SYSTEM, content, EQUIV_SCHEMA)

    # ------------------------------------------------------------ process
    async def diagnose(self, instance: TaskInstance, response: str, thinking: str) -> dict[str, Any]:
        imgs = _image_blocks(instance, self.show_image, self.max_side)
        facts = json.dumps(instance.facts, default=str, indent=1)
        ref = json.dumps(instance.answer.value, default=str)
        content = imgs + [{"type": "text", "text": (
            f"<task>\n{instance.prompt}\n</task>\n<ground_truth_facts>\n{facts}\n</ground_truth_facts>\n"
            f"<correct_answer>{ref}</correct_answer>\n"
            f"<model_reasoning_summary>\n{thinking[-12000:]}\n</model_reasoning_summary>\n"
            f"<model_response>\n{response[-8000:]}\n</model_response>")}]
        return await self._call(PROCESS_SYSTEM, content, PROCESS_SCHEMA)


def b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode()
