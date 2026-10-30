"""Answer extraction and deterministic checking for every AnswerSpec kind.

The output protocol is strict: the final answer goes inside <answer>...</answer>
(or the agentic submit_answer tool). An answer that can't be parsed into the
expected type is scored "unparseable", separately from "incorrect", so
formatting failures are never confused with reasoning failures. Scoring
counts both as wrong.
"""

from __future__ import annotations

import re
from typing import Any

from visionbench.core.types import AnswerSpec

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
CONF_RE = re.compile(r"<confidence>\s*([0-9]+(?:\.[0-9]+)?)\s*%?\s*</confidence>", re.S | re.I)
FILLER = {"the", "block", "face", "object", "objects", "color", "colour", "letter", "number", "is",
          "answer", "final", "it", "on", "terminal", "wire", "of", "and"}


def extract_answer(text: str) -> tuple[str | None, float | None, str]:
    """Return (answer, confidence 0..1, method). Uses the *last* <answer> block."""
    conf = None
    m = CONF_RE.findall(text or "")
    if m:
        try:
            conf = max(0.0, min(1.0, float(m[-1]) / 100.0))
        except ValueError:
            conf = None
    found = ANSWER_RE.findall(text or "")
    if found:
        return found[-1].strip(), conf, "tag"
    m2 = re.findall(r"final answer\s*[:\-]\s*(.+)", text or "", re.I)
    if m2:
        return m2[-1].strip(), conf, "fallback"
    return None, conf, "none"


# ----------------------------------------------------------------- parsers
def parse_number(s: str) -> float | None:
    s = str(s).strip().replace(",", "").replace("$", "").replace("−", "-")
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    fracs = re.findall(r"(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", s)
    if len(fracs) == 1 and len(nums) == 2:
        den = float(fracs[0][1])
        return float(fracs[0][0]) / den if den else None
    if len(nums) == 1:
        return float(nums[0])
    if len(nums) > 1:
        # "x = 12 ... so 37.5": ambiguous unless all equal
        vals = {float(n) for n in nums}
        return vals.pop() if len(vals) == 1 else None
    return None


def norm_token(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[\"'`*_.()\[\]{}]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_text(s: str) -> str:
    toks = [t for t in re.split(r"[\s,;:!?.]+", norm_token(s)) if t and t not in FILLER]
    return " ".join(toks)


def parse_tokens(s: str) -> list[str]:
    s = re.sub(r"\band\b", ",", str(s), flags=re.I)
    return [norm_token(t) for t in re.split(r"[,;\n]+|\s{2,}", s) if norm_token(t)] or \
        [norm_token(t) for t in s.split() if norm_token(t)]


def parse_mapping(s: str) -> dict[str, str]:
    out = {}
    for k, v in re.findall(r"([A-Za-z0-9_\-]+)\s*(?:=|:|->|→)\s*([A-Za-z0-9_.\-]+(?:-[A-Za-z]+)?)", str(s)):
        out[norm_token(k)] = norm_token(v)
    return out


def parse_grid(s: str) -> list[list[int]] | None:
    rows = []
    for line in str(s).strip().splitlines():
        digits = re.findall(r"\d", line)
        if digits:
            rows.append([int(d) for d in digits])
    return rows or None


# ------------------------------------------------------------------ checks
def check(spec: AnswerSpec, raw: str, family=None, instance=None) -> tuple[str, float, float | None, Any, dict]:
    """Return (outcome, score, partial, parsed, details). outcome in correct|incorrect|unparseable."""
    k = spec.kind
    truth = spec.value
    if k in ("integer", "number"):
        x = parse_number(raw)
        if x is None:
            return "unparseable", 0.0, None, None, {}
        if k == "integer" and spec.tolerance == 0:
            ok = abs(x - round(x)) < 1e-9 and int(round(x)) == int(truth)
        else:
            tol = max(spec.tolerance, spec.rel_tolerance * abs(float(truth)), 1e-9)
            ok = abs(x - float(truth)) <= tol
        return ("correct" if ok else "incorrect"), float(ok), None, x, {"abs_error": abs(x - float(truth))}
    if k == "choice":
        m = re.findall(r"\b([A-Z])\b", str(raw).upper())
        if len(set(m)) != 1:
            return "unparseable", 0.0, None, None, {}
        ok = m[0] == str(truth).upper()
        return ("correct" if ok else "incorrect"), float(ok), None, m[0], {}
    if k == "text":
        a, t = norm_text(raw), norm_text(truth)
        if not a:
            return "unparseable", 0.0, None, None, {}
        ok = a == t
        return ("correct" if ok else "incorrect"), float(ok), None, a, {}
    if k == "sequence":
        toks = parse_tokens(raw)
        ok = toks == [norm_token(t) for t in truth]
        return ("correct" if ok else "incorrect"), float(ok), None, toks, {}
    if k == "set":
        got = {norm_token(t) for t in parse_tokens(raw)} - {""}
        want = {norm_token(t) for t in truth}
        if not got and want:
            # "none" is a legitimate answer only when the truth is empty
            return "unparseable", 0.0, None, None, {}
        tp = len(got & want)
        prec = tp / len(got) if got else 1.0
        rec = tp / len(want) if want else 1.0
        f1 = 0.0 if tp == 0 else 2 * prec * rec / (prec + rec)
        ok = got == want
        return ("correct" if ok else "incorrect"), float(ok), f1, sorted(got), \
            {"precision": prec, "recall": rec, "missing": sorted(want - got), "extra": sorted(got - want)}
    if k == "mapping":
        got = parse_mapping(raw)
        want = {norm_token(a): norm_token(b) for a, b in truth.items()}
        if not got:
            return "unparseable", 0.0, None, None, {}
        per = {key: got.get(key) == val for key, val in want.items()}
        frac = sum(per.values()) / len(per)
        ok = all(per.values())
        return ("correct" if ok else "incorrect"), float(ok), frac, got, {"per_key": per}
    if k == "grid":
        g = parse_grid(raw)
        if g is None or any(len(r) != len(g[0]) for r in g):
            return "unparseable", 0.0, None, g, {}
        same_shape = len(g) == len(truth) and len(g[0]) == len(truth[0])
        if not same_shape:
            return "incorrect", 0.0, 0.0, g, {"shape": [len(g), len(g[0])], "want_shape": [len(truth), len(truth[0])]}
        cells = [a == b for ra, rb in zip(g, truth) for a, b in zip(ra, rb)]
        acc = sum(cells) / len(cells)
        ok = acc == 1.0
        return ("correct" if ok else "incorrect"), float(ok), acc, g, {"cell_accuracy": acc}
    if k == "custom":
        if family is None:
            raise ValueError("custom answer kind needs the task family")
        score, details = family.check(instance, raw)
        if details.get("valid") is False and not details.get("moves") and not str(raw).strip():
            return "unparseable", 0.0, None, raw, details
        return ("correct" if score >= 1.0 else "incorrect"), float(score >= 1.0), score, raw, details
    raise ValueError(f"kind {k} is not programmatically checkable")
