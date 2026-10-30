"""Reasoning critique: grade *how* each agent reasoned, and build per-agent reasoning profiles.

Per attempt, the critic (an LLM judge given the generator's ground-truth key
facts) returns structured judgments. From these we derive metrics that are
then aggregated per agent:

  perception_precision   share of the agent's concrete image claims that are correct
  key_fact_recall        share of decisive facts the agent stated or used correctly
  key_fact_error_rate    share of mentioned key facts the agent got wrong (misreads)
  reasoning_valid        steps follow from the agent's own premises
  sound_success          correct AND reasoning valid AND answer supported
  unsupported_success    correct but not earned (guess, or wrong steps that happened to land right)
  failure_locus          perception (misread a key fact) / reasoning (premises right, inference wrong) / other

Agentic attempts also get deterministic evidence grounding: did the crops
cover the gold evidence boxes?
"""

from __future__ import annotations

from typing import Any


PERCEPTUAL = {"perception_misread", "perception_omission", "hallucination", "attribute_binding", "tracking",
              "counting_enumeration"}
REASONING = {"spatial_transform", "arithmetic", "rule_misapplication", "premature_commitment"}


def derive(j: dict[str, Any], n_key_facts: int, outcome: str) -> dict[str, Any]:
    obs = j.get("observations") or []
    n_ok = sum(o.get("verdict") == "correct" for o in obs)
    n_bad = sum(o.get("verdict") == "incorrect" for o in obs)
    kf = {int(k.get("index", -1)): k.get("status") for k in (j.get("key_facts") or [])}
    statuses = [kf.get(i, "not_mentioned") for i in range(n_key_facts)]
    stated_ok = statuses.count("stated_correctly")
    stated_bad = statuses.count("stated_incorrectly")
    correct = outcome == "correct"
    valid = bool(j.get("reasoning_valid"))
    supported = bool(j.get("answer_supported"))
    # Locus comes from the critic's FIRST error. Key-fact misreads are only a fallback: key facts include
    # derived conclusions (e.g. the final gear direction), so any wrong answer would otherwise look perceptual.
    et = j.get("error_type") or "none"
    if correct:
        locus = None
    elif outcome in ("unparseable", "refusal", "error") or et == "gave_up":
        locus = "no_answer"
    elif et in PERCEPTUAL:
        locus = "perception"
    elif et in REASONING:
        locus = "reasoning"
    elif et == "instruction_misread":
        locus = "instruction"
    else:
        locus = "perception" if stated_bad > 0 else "reasoning"
    return {
        "n_claims": len(obs),
        "perception_precision": n_ok / (n_ok + n_bad) if (n_ok + n_bad) else None,
        "hallucinated_claims": n_bad,
        "key_fact_recall": stated_ok / n_key_facts if n_key_facts else None,
        "key_fact_error_rate": stated_bad / (stated_ok + stated_bad) if (stated_ok + stated_bad) else None,
        "reasoning_valid": valid,
        "answer_supported": supported,
        "sound_success": correct and valid and supported,
        "unsupported_success": correct and not (valid and supported),
        "failure_locus": locus,
        "error_type": j.get("error_type"),
        "strategy": j.get("strategy"),
        "self_verification": bool(j.get("self_verification")),
        "efficiency": j.get("efficiency"),
    }


def attempt_view(att) -> dict[str, Any]:
    """The parts of an attempt the critic sees (no ground truth leaks from here)."""
    calls = []
    for e in att.trajectory:
        if e.get("event") == "assistant":
            calls += e.get("tool_calls") or []
    return {"answer": att.raw_answer, "response": att.final_text, "thinking": att.thinking_summary,
            "tool_calls": calls}
