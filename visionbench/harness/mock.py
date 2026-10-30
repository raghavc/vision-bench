"""Offline mock subjects for pipeline tests and oracle self-checks (no API calls).

  mock:oracle       always submits the ground truth (formatted like a model would)
  mock:noisy-0.7    correct with probability 0.7 (per item difficulty-adjusted), else a plausible wrong answer
  mock:guess        always a plausible wrong/random answer
"""

from __future__ import annotations

import json
import random
import zlib

from visionbench.core.types import Attempt, TaskInstance


def format_truth(inst: TaskInstance) -> str:
    spec = inst.answer
    v = spec.value
    k = spec.kind
    if k in ("integer",):
        return str(int(v))
    if k == "number":
        return f"{float(v):g}"
    if k in ("text", "choice"):
        return str(v)
    if k == "set":
        return ", ".join(v)
    if k == "sequence":
        return ", ".join(map(str, v))
    if k == "mapping":
        return ", ".join(f"{a}={b}" for a, b in v.items())
    if k == "grid":
        return "\n".join(" ".join(map(str, r)) for r in v)
    if k == "custom":
        if inst.family == "gear_train":
            return v["direction"] if v["direction"] == "stationary" else f"{v['direction']} {v['rpm']:g}"
        if inst.family == "maze_path":
            return solve_maze(inst)
    if k == "rubric":
        return "\n".join(f"{a['type']} @ t≈{a['t']}" for a in v)
    return json.dumps(v)


def solve_maze(inst: TaskInstance) -> str:
    from collections import deque

    from visionbench.tasks.families.maze_path import DIRS
    F = inst.facts

    def path(a, b):
        prev = {a: None}
        q = deque([a])
        while q:
            p = q.popleft()
            for d in F["open"][f"{p[0]},{p[1]}"]:
                n = (p[0] + DIRS[d][0], p[1] + DIRS[d][1])
                if n not in prev:
                    prev[n] = (p, d)
                    q.append(n)
        s, p = "", b
        while prev[p]:
            p, d = prev[p]
            s = d + s
        return s

    pts = [tuple(F["start"])] + [tuple(F["keys"][i]) for i in F["optimal_key_order"]] + [tuple(F["goal"])]
    return "".join(path(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def wrong_answer(inst: TaskInstance, rnd: random.Random) -> str:
    spec = inst.answer
    if spec.kind in ("integer", "number") and isinstance(spec.value, (int, float)):
        return str(spec.value + rnd.choice([-2, -1, 1, 2, 3]))
    if spec.kind == "set":
        return ", ".join(spec.value[:-1]) or "Z"
    if spec.kind == "mapping":
        items = list(spec.value.items())
        return ", ".join(f"{a}={'x' if i == 0 else b}" for i, (a, b) in enumerate(items))
    if spec.kind == "grid":
        g = [list(r) for r in spec.value]
        g[0][0] = (g[0][0] + 1) % 10
        return "\n".join(" ".join(map(str, r)) for r in g)
    if spec.kind == "custom" and inst.family == "gear_train":
        return "clockwise 1"
    if spec.kind == "custom" and inst.family == "maze_path":
        return "UUUU"
    return "unknown"


def mock_attempt(inst: TaskInstance, model: str, mode: str, condition: str, sample: int) -> Attempt:
    rnd = random.Random(zlib.crc32(f"{model}|{inst.uid}|{condition}|{sample}".encode()))
    kind = model.split(":", 1)[1]
    if kind == "oracle":
        p = 1.0
    elif kind.startswith("noisy-"):
        p = float(kind.split("-")[1]) * (1.15 - 0.1 * inst.difficulty)
    else:
        p = 0.0
    if condition == "blind":
        p *= 0.15
    elif condition == "caption":
        p *= 0.6
    ok = rnd.random() < p
    ans = format_truth(inst) if ok else wrong_answer(inst, rnd)
    conf = min(1.0, max(0.0, (0.8 if ok else 0.6) + rnd.uniform(-0.2, 0.2)))
    text = f"Observations...\n<answer>{ans}</answer>\n<confidence>{conf * 100:.0f}</confidence>"
    return Attempt(uid=inst.uid, family=inst.family, cluster=inst.cluster, difficulty=inst.difficulty, model=model,
                   mode=mode, condition=condition, sample=sample, raw_answer=ans, confidence=conf, final_text=text,
                   stop_reason="end_turn", turns=1, latency_s=rnd.uniform(2, 20))
