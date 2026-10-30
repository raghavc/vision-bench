"""Thin async wrapper over the Anthropic SDK shared by the harness and the judges."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic

from visionbench.config import Settings
from visionbench.core.types import Usage

FALLBACK_BETA = "server-side-fallback-2026-07-01"


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
    def __init__(self, settings: Settings):
        if not settings.api_key:
            raise RuntimeError(
                "No Anthropic API key. Put it in config/settings.toml ([anthropic] api_key = \"sk-ant-...\") "
                "or export ANTHROPIC_API_KEY.")
        a = settings.section("anthropic")
        self.client = anthropic.AsyncAnthropic(
            api_key=settings.api_key,
            max_retries=int(a.get("max_retries", 4)),
            timeout=float(a.get("timeout_seconds", 900)),
        )

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


class RateGate:
    """Bounded concurrency shared across a run."""

    def __init__(self, n: int):
        self.sem = asyncio.Semaphore(n)

    async def __aenter__(self):
        await self.sem.acquire()
        return self

    async def __aexit__(self, *exc):
        self.sem.release()
