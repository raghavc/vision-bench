"""Async model access shared by the harness and the judges.

Model ids without a prefix go to the Anthropic API (official SDK). Ids prefixed
with `or:` (e.g. `or:google/gemini-4-argon`) go to OpenRouter's
OpenAI-compatible endpoint, which allows cross-vendor comparisons. OpenRouter
calls are metered with the provider-reported cost, and a hard budget stops
new calls once it is reached.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import json

import anthropic

from visionbench.config import Settings
from visionbench.core.types import Usage

FALLBACK_BETA = "server-side-fallback-2026-07-01"
OR_PREFIX = "or:"


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class LLMResult:
    content: list[Any]
    text: str
    thinking: str
    stop_reason: str
    usage: Usage
    latency_s: float
    model_served: str
    request_id: str | None = None
    refusal_category: str | None = None
    tool_uses: list[Any] = field(default_factory=list)
    container_id: str | None = None


class Claude:
    """Routes each call to Anthropic or OpenRouter by model id; tracks OpenRouter spend against a budget."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._anthropic: anthropic.AsyncAnthropic | None = None
        self._openrouter = None
        o = settings.section("openrouter")
        self.or_budget = float(o.get("budget_usd", 0) or 0)
        self.or_spent = 0.0
        self.or_max_tokens = int(o.get("max_tokens", 16000))
        self._spend_lock = asyncio.Lock()

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        if self._anthropic is None:
            if not self.settings.api_key:
                raise RuntimeError(
                    "No Anthropic API key. Put it in config/settings.toml ([anthropic] api_key = \"sk-ant-...\") "
                    "or export ANTHROPIC_API_KEY.")
            a = self.settings.section("anthropic")
            self._anthropic = anthropic.AsyncAnthropic(
                api_key=self.settings.api_key,
                max_retries=int(a.get("max_retries", 4)),
                timeout=float(a.get("timeout_seconds", 900)),
            )
        return self._anthropic

    @property
    def openrouter(self):
        if self._openrouter is None:
            import openai
            o = self.settings.section("openrouter")
            if not o.get("api_key"):
                raise RuntimeError("No OpenRouter key. Put it in config/settings.toml ([openrouter] api_key = \"sk-or-...\").")
            self._openrouter = openai.AsyncOpenAI(
                api_key=o["api_key"], base_url=o.get("base_url", "https://openrouter.ai/api/v1"),
                max_retries=int(o.get("max_retries", 4)), timeout=float(o.get("timeout_seconds", 600)),
                default_headers={"HTTP-Referer": "https://github.com/raghavc/vision-bench", "X-Title": "VisionBench"})
        return self._openrouter

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        effort: str | None = None,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
        fallbacks: bool = False,
        thinking_display: str = "summarized",
        container: str | None = None,
    ) -> LLMResult:
        if model.startswith(OR_PREFIX):
            if tools:
                raise NotImplementedError("agentic tool loops are Anthropic-only for now; use --mode direct")
            return await self._create_openrouter(model[len(OR_PREFIX):], messages, max_tokens, effort, system,
                                                 json_schema)
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "thinking": {"type": "adaptive", "display": thinking_display},
        }
        output_config: dict[str, Any] = {}
        if effort:
            output_config["effort"] = effort
        if json_schema:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        if output_config:
            params["output_config"] = output_config
        if system:
            params["system"] = system
        if tools:
            params["tools"] = tools
        if container:
            params["container"] = container

        t0 = time.monotonic()
        if fallbacks:
            async with self.client.beta.messages.stream(betas=[FALLBACK_BETA], fallbacks="default", **params) as s:
                msg = await s.get_final_message()
        else:
            async with self.client.messages.stream(**params) as s:
                msg = await s.get_final_message()
        latency = time.monotonic() - t0

        text = "".join(b.text for b in msg.content if b.type == "text")
        thinking = "\n".join(getattr(b, "thinking", "") or "" for b in msg.content if b.type == "thinking")
        u = msg.usage
        usage = Usage(
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        refusal = None
        if msg.stop_reason == "refusal":
            sd = getattr(msg, "stop_details", None)
            refusal = getattr(sd, "category", None) or "unspecified"
        return LLMResult(
            content=list(msg.content),
            text=text,
            thinking=thinking,
            stop_reason=msg.stop_reason or "",
            usage=usage,
            latency_s=latency,
            model_served=msg.model,
            request_id=getattr(msg, "_request_id", None),
            refusal_category=refusal,
            tool_uses=[b for b in msg.content if b.type == "tool_use"],
            container_id=getattr(getattr(msg, "container", None), "id", None),
        )

    # ---------------------------------------------------------- OpenRouter
    async def _create_openrouter(self, model, messages, max_tokens, effort, system, json_schema) -> LLMResult:
        if self.or_budget and self.or_spent >= self.or_budget:
            raise BudgetExceeded(f"OpenRouter budget ${self.or_budget:.2f} reached (spent ${self.or_spent:.4f})")
        oa_msgs: list[dict[str, Any]] = []
        if system:
            oa_msgs.append({"role": "system", "content": system})
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                oa_msgs.append({"role": m["role"], "content": content})
                continue
            parts = []
            for b in content:
                if b["type"] == "text":
                    parts.append({"type": "text", "text": b["text"]})
                elif b["type"] == "image":
                    src = b["source"]
                    parts.append({"type": "image_url",
                                  "image_url": {"url": f"data:{src['media_type']};base64,{src['data']}"}})
            oa_msgs.append({"role": m["role"], "content": parts})
        extra: dict[str, Any] = {"usage": {"include": True}}
        if effort:
            extra["reasoning"] = {"effort": {"xhigh": "high", "max": "high"}.get(effort, effort)}
        kwargs: dict[str, Any] = {}
        if json_schema:
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "output", "strict": True, "schema": json_schema}}
        t0 = time.monotonic()
        resp = await self.openrouter.chat.completions.create(
            model=model, messages=oa_msgs, max_tokens=min(max_tokens, self.or_max_tokens), extra_body=extra, **kwargs)
        latency = time.monotonic() - t0
        if not resp.choices:
            raise RuntimeError(f"OpenRouter returned no choices: {str(resp)[:300]}")
        ch = resp.choices[0]
        msg = ch.message
        text = msg.content or ""
        if json_schema and text and not text.lstrip().startswith("{"):
            # some providers wrap JSON in prose or fences; keep the outermost object
            a, b = text.find("{"), text.rfind("}")
            if a != -1 and b > a:
                text = text[a:b + 1]
        reasoning = getattr(msg, "reasoning", None) or (msg.model_extra or {}).get("reasoning") or ""
        u = resp.usage
        cost = float((u.model_extra or {}).get("cost") or 0.0) if u else 0.0
        usage = Usage(input_tokens=(u.prompt_tokens or 0) if u else 0,
                      output_tokens=(u.completion_tokens or 0) if u else 0, cost_usd=cost)
        async with self._spend_lock:
            self.or_spent += cost
        finish = ch.finish_reason or ""
        stop = {"stop": "end_turn", "length": "max_tokens", "content_filter": "refusal"}.get(finish, finish)
        return LLMResult(content=[], text=text, thinking=str(reasoning), stop_reason=stop, usage=usage,
                         latency_s=latency, model_served=f"{OR_PREFIX}{resp.model}", request_id=resp.id,
                         refusal_category="content_filter" if stop == "refusal" else None)


class RateGate:
    """Bounded concurrency shared across a run."""

    def __init__(self, n: int):
        self.sem = asyncio.Semaphore(n)

    async def __aenter__(self):
        await self.sem.acquire()
        return self

    async def __aexit__(self, *exc):
        self.sem.release()
