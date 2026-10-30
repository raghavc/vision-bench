"""Subject runners: direct (single call) and agentic (tool loop with zoom / code execution).

Every attempt returns an `Attempt` with the full trajectory (messages, tool
calls, crops saved to disk as an image ledger), token usage, latency, and
stop reason. The loops are append-only (assistant content is passed back
unchanged), as preserved thinking requires.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from visionbench.config import Settings
from visionbench.core.types import Attempt, TaskInstance, Usage
from visionbench.harness.images import crop_zoom, encode_image
from visionbench.harness.prompting import CAPTION_PROMPT, SYSTEM_AGENT, SYSTEM_DIRECT, user_text
from visionbench.llm import Claude
from visionbench.verifiers.answers import extract_answer

ZOOM_TOOL = {
    "name": "zoom",
    "description": (
        "Return a crop of one of the task images at high resolution. Coordinates are pixels in the ORIGINAL image "
        "(its size is stated in the task), with (0,0) at the top-left. Use this to read small text, count "
        "carefully, or follow thin lines."),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "image_index": {"type": "integer", "description": "0-based index of the image"},
            "x0": {"type": "number"}, "y0": {"type": "number"},
            "x1": {"type": "number"}, "y1": {"type": "number"},
        },
        "required": ["image_index", "x0", "y0", "x1", "y1"],
        "additionalProperties": False,
    },
}

SUBMIT_TOOL = {
    "name": "submit_answer",
    "description": "Submit your single final answer. Call exactly once, at the end.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "Final answer in the required format"},
            "confidence": {"type": "number", "description": "Probability (0-100) that the answer is exactly correct"},
        },
        "required": ["answer", "confidence"],
        "additionalProperties": False,
    },
}

CODE_TOOL = {"type": "code_execution_20260521", "name": "code_execution"}


class SubjectRunner:
    def __init__(self, claude: Claude, settings: Settings, ledger_dir: Path):
        self.claude = claude
        s, h = settings.subject, settings.harness
        self.effort = s.get("effort", "high")
        self.max_tokens = int(s.get("max_tokens", 32000))
        self.fallbacks = bool(s.get("fallbacks", False))
        self.max_side = int(h.get("max_image_side", 1568))
        self.max_turns = int(h.get("max_turns", 12))
        self.zoom = bool(h.get("zoom_tool", True))
        self.code = bool(h.get("code_execution", False))
        self.ledger_dir = ledger_dir
        self.caption_model = settings.judge.get("model", "claude-opus-5-5")
        self._captions: dict[str, tuple[str, Usage]] = {}

    # -------------------------------------------------------------- helpers
    def _image_content(self, inst: TaskInstance) -> tuple[list[dict], list[list[int]]]:
        blocks, sizes = [], []
        for k, im in enumerate(inst.images):
            block, sent = encode_image(im, self.max_side)
            if len(inst.images) > 1:
                blocks.append({"type": "text", "text": f"Image {k}:"})
            blocks.append(block)
            sizes.append([im.width, im.height, sent[0], sent[1]])
        return blocks, sizes

    async def caption(self, inst: TaskInstance) -> tuple[str, Usage]:
        """Question-agnostic dense caption, cached per item (caption-only control)."""
        if inst.uid in self._captions:
            return self._captions[inst.uid]
        imgs, _ = self._image_content(inst)
        res = await self.claude.create(model=self.caption_model, max_tokens=8000, effort="high",
                                       messages=[{"role": "user", "content": imgs + [{"type": "text", "text": CAPTION_PROMPT}]}],
                                       fallbacks=True, thinking_display="omitted")
        self._captions[inst.uid] = (res.text, res.usage)
        return res.text, res.usage

    # --------------------------------------------------------------- direct
    async def run_direct(self, inst: TaskInstance, model: str, condition: str, sample: int) -> Attempt:
        att = Attempt(uid=inst.uid, family=inst.family, cluster=inst.cluster, difficulty=inst.difficulty,
                      model=model, mode="direct", condition=condition, sample=sample)
        content: list[dict] = []
        caption = None
        if condition == "vision":
            imgs, sizes = self._image_content(inst)
            content += imgs
            att.trajectory.append({"event": "images", "sizes_orig_and_sent": sizes})
        elif condition == "caption":
            caption, cu = await self.caption(inst)
            att.usage.add(cu)
            att.trajectory.append({"event": "caption", "text": caption})
        content.append({"type": "text", "text": user_text(inst, "direct", condition, caption)})
        t0 = time.monotonic()
        res = await self.claude.create(model=model, system=SYSTEM_DIRECT, max_tokens=self.max_tokens,
                                       effort=self.effort, messages=[{"role": "user", "content": content}],
                                       fallbacks=self.fallbacks)
        att.latency_s = time.monotonic() - t0
        att.usage.add(res.usage)
        att.turns = 1
        att.stop_reason = res.stop_reason
        att.final_text = res.text
        att.thinking_summary = res.thinking
        if res.model_served != model:
            att.trajectory.append({"event": "served_by_other_model", "model": res.model_served})
        if res.stop_reason == "refusal":
            att.error = f"refusal:{res.refusal_category}"
            return att
        ans, conf, method = extract_answer(res.text)
        att.raw_answer, att.confidence = ans, conf
        att.trajectory.append({"event": "final", "extraction": method, "request_id": res.request_id})
        return att

    # -------------------------------------------------------------- agentic
    async def run_agentic(self, inst: TaskInstance, model: str, sample: int) -> Attempt:
        att = Attempt(uid=inst.uid, family=inst.family, cluster=inst.cluster, difficulty=inst.difficulty,
                      model=model, mode="agentic", condition="vision", sample=sample)
        imgs, sizes = self._image_content(inst)
        size_txt = "; ".join(f"image {k}: original {w}x{h} px (shown to you at {sw}x{sh})"
                             for k, (w, h, sw, sh) in enumerate(sizes))
        tools: list[dict[str, Any]] = [SUBMIT_TOOL]
        if self.zoom:
            tools.insert(0, ZOOM_TOOL)
        container = None
        extra: list[dict] = []
        if self.code:
            tools.append(CODE_TOOL)
            for k, im in enumerate(inst.images):
                import io
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                f = await self.claude.client.files.upload(file=(f"image_{k}.png", buf.getvalue(), "image/png"))
                extra.append({"type": "container_upload", "file_id": f.id})
        system = SYSTEM_AGENT.format(code=(" and run Python (PIL, numpy available; the images are uploaded as "
                                           "image_0.png, ...) with code execution") if self.code else "")
        text = user_text(inst, "agentic", "vision") + f"\n\n[{size_txt}]"
        messages: list[dict[str, Any]] = [{"role": "user", "content": imgs + extra + [{"type": "text", "text": text}]}]
        att.trajectory.append({"event": "images", "sizes_orig_and_sent": sizes})
        t0 = time.monotonic()
        crops = 0
        for turn in range(self.max_turns):
            res = await self.claude.create(model=model, system=system, max_tokens=self.max_tokens,
                                           effort=self.effort, messages=messages, tools=tools,
                                           fallbacks=self.fallbacks, container=container)
            att.usage.add(res.usage)
            container = res.container_id or container  # keep uploaded files across turns
            att.turns = turn + 1
            att.stop_reason = res.stop_reason
            att.thinking_summary += ("\n" if att.thinking_summary else "") + res.thinking
            att.final_text = res.text or att.final_text
            att.trajectory.append({"event": "assistant", "turn": turn, "text": res.text[-4000:],
                                   "tool_calls": [{"name": t.name, "input": t.input} for t in res.tool_uses],
                                   "server_tools": [b.type for b in res.content if b.type.endswith("tool_result")],
                                   "request_id": res.request_id})
            if res.stop_reason == "refusal":
                att.error = f"refusal:{res.refusal_category}"
                break
            messages.append({"role": "assistant", "content": res.content})
            if res.stop_reason == "pause_turn":
                continue
            submit = next((t for t in res.tool_uses if t.name == "submit_answer"), None)
            if submit is not None:
                att.raw_answer = str(submit.input.get("answer", ""))
                c = submit.input.get("confidence")
                att.confidence = max(0.0, min(1.0, float(c) / 100)) if isinstance(c, (int, float)) else None
                att.tool_calls += 1
                break
            if not res.tool_uses:
                # ended without submitting: accept a tagged answer, otherwise nudge once
                ans, conf, method = extract_answer(res.text)
                if ans is not None:
                    att.raw_answer, att.confidence = ans, conf
                    break
                messages.append({"role": "user", "content": [{"type": "text", "text":
                                 "Please call submit_answer with your final answer now."}]})
                continue
            results = []
            for t in res.tool_uses:
                att.tool_calls += 1
                if t.name == "zoom":
                    try:
                        inp = t.input
                        im = inst.images[int(inp["image_index"])]
                        crop = crop_zoom(im, (inp["x0"], inp["y0"], inp["x1"], inp["y1"]), self.max_side)
                        crops += 1
                        rel = f"{inst.uid}_{model}_{sample}_{crops}.png".replace("/", "_")
                        crop.save(self.ledger_dir / rel)
                        block, _ = encode_image(crop, self.max_side)
                        results.append({"type": "tool_result", "tool_use_id": t.id, "content": [
                            {"type": "text", "text": f"Crop of image {inp['image_index']} region "
                                                     f"({inp['x0']:.0f},{inp['y0']:.0f})-({inp['x1']:.0f},{inp['y1']:.0f}), "
                                                     f"scaled to {crop.width}x{crop.height}."}, block]})
                        att.trajectory.append({"event": "crop", "file": rel, "box": [inp["x0"], inp["y0"], inp["x1"], inp["y1"]],
                                               "image_index": inp["image_index"]})
                    except (ValueError, IndexError, KeyError, TypeError) as e:
                        results.append({"type": "tool_result", "tool_use_id": t.id, "is_error": True,
                                        "content": f"Error: {e}"})
                else:
                    results.append({"type": "tool_result", "tool_use_id": t.id, "is_error": True,
                                    "content": f"Unknown tool {t.name}"})
            messages.append({"role": "user", "content": results})
        else:
            att.error = "max_turns_exhausted"
        att.latency_s = time.monotonic() - t0
        return att


def attempt_json(att: Attempt) -> str:
    return json.dumps(att.to_record(), default=str)
