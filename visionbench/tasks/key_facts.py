"""Decisive ground-truth facts per item, used to grade an agent's reasoning, not just its answer.

Each family turns the facts its generator recorded into a short, numbered list
of human-readable statements: the facts a correct solution must perceive or
derive. The reasoning critic checks each one against the agent's stated
observations. That yields key-fact recall (what the agent saw) and key-fact
error (what it misread), independent of whether its final answer was right.
"""

from __future__ import annotations

import heapq
from typing import Any, Callable

from visionbench.core.types import TaskInstance

MAX_FACTS = 14


def _wire(f, inst):
    q = inst.answer.value if isinstance(inst.answer.value, dict) else None
    if q is None:
        letter = next(iter(f["crossings_on_queried_wires"]))
        q = {letter: inst.answer.value}
    out = [f"{f['n_wires']} wires; wires are {'all the same colour' if f['same_color'] else 'differently coloured'}"]
    for k, v in q.items():
        out.append(f"Wire {k} ends at terminal {v} (it crosses other wires {f['crossings_on_queried_wires'][k]} times)")
    return out


def _gear(f, inst):
    g = f["gears"]
    path = []
    cur = f["target"]
    while cur is not None:
        path.append(cur)
        cur = g[cur]["parent"]
    path.reverse()
    out = [f"Gear {path[0]} is the driver ({g[path[0]]['teeth']} teeth)"]
    link_txt = {"mesh": "meshes directly with", "open_belt": "is connected by an open (uncrossed) belt to",
                "crossed_belt": "is connected by a crossed belt to", "gap": "does NOT touch (visible gap)"}
    for a, b in zip(path, path[1:]):
        out.append(f"Gear {b} ({g[b]['teeth']} teeth) {link_txt[g[b]['link']]} gear {a}")
    m = f["motion"][f["target"]]
    out.append(f"Gear {f['target']}: " + ("stationary (not driven)" if m is None else
                                          f"{'clockwise' if m['cw'] else 'counterclockwise'} at {m['rpm']:g} rpm"))
    decoys = [k for k, v in g.items() if v["link"] == "gap"]
    if decoys:
        out.append("Gears not driven (separated by a gap): " + ", ".join(decoys))
    return out


def _maze(f, inst):
    out = [f"{f['size']}x{f['size']} maze; S at (row {f['start'][0]}, col {f['start'][1]}); "
           f"G at (row {f['goal'][0]}, col {f['goal'][1]})"]
    for i, k in enumerate(f["keys"]):
        out.append(f"Key cell K{'' if len(f['keys']) == 1 else i + 1} at (row {k[0]}, col {k[1]})")
    out.append(f"The shortest valid route has exactly {f['optimal_length']} moves")
    return out


def _grid(f, inst):
    ti = f["test_input"]
    to = inst.answer.value
    return [f"Hidden rule: {f['program_text']}",
            f"Test input is {len(ti)} rows x {len(ti[0])} columns",
            f"Correct output is {len(to)} rows x {len(to[0])} columns"]


def _chart(f, inst):
    out = [f"Chart type: {f['chart_kind']}"]
    for s, vals in f["values"].items():
        out.append(f"{s}: " + ", ".join(f"{k}={v}" for k, v in vals.items()))
    out += [f"Step: {s}" for s in f.get("reasoning_steps", [])]
    return out


def _counting(f, inst):
    preds = f["predicates"]

    def match(s):
        for p in preds:
            hit = s.get(p["attr"]) == p["value"]
            if p["negated"]:
                hit = not hit
            if not hit:
                return False
        return True
    hits = [s for s in f["shapes"] if match(s)]
    out = [f"{f['total_shapes']} shapes in total; {len(hits)} satisfy the query"]
    for s in hits[:10]:
        out.append(f"Matching: {s['size']} {s['fill']} {s['color']} {s['shape']} near (x={s['x']:.0f}, y={s['y']:.0f})")
    return out


def _lattice(f, inst):
    out = ["Vertices (in order): " + ", ".join(f"({x},{y})" for x, y in f["vertices"]),
           f"Polygon area = {f['twice_area'] / 2:g}", f"Boundary lattice points = {f['boundary_points']}",
           f"Interior lattice points = {f['interior_points']}"]
    if f.get("hole"):
        x0, y0, x1, y1 = f["hole"]
        out.append(f"Rectangular hole from ({x0},{y0}) to ({x1},{y1}), area {(x1 - x0) * (y1 - y0)}")
    return out


def _cube(f, inst):
    v = f.get("view")
    if not v:
        return [f"Correct answer: {inst.answer.value}"]
    return [f"Visible in the view: top={v['top']}, front={v['front']}, right={v['right']}",
            f"Hidden faces: back={v['back']}, left={v['left']}, bottom={v['bottom']}",
            f"Opposite pairs: {v['top']}–{v['bottom']}, {v['front']}–{v['back']}, {v['left']}–{v['right']}"]


def _clock(f, inst):
    return [f"Start clock shows {f['start_hhmm']}",
            f"End clock {'(a mirror image) actually shows' if f['mirrored_end'] else 'shows'} {f['end_hhmm']}",
            f"Elapsed time = {f['elapsed_min']} minutes"]


def _receipt(f, inst):
    out = []
    voids = [ln for ln in f["lines"] if ln["void"]]
    if voids:
        out.append("Voided (not charged): " + "; ".join(f"{ln['qty']} x {ln['name']} @ {ln['price']}" for ln in voids))
    out.append("Category totals (non-voided): " + ", ".join(f"{k}={v}" for k, v in f["category_totals"].items()))
    out.append(f"Subtotal {f['subtotal']}; discount {f['discount']}; tax {f['tax']}; total {f['total']}")
    return out


def _persp(f, inst):
    ego = f.get("ego_after") or f.get("ego") or f.get("ego1")
    out = []
    if "new_pos" in f:
        out.append(f"After turning {f['turn']} and moving {f['steps']}, R1 is at column {f['new_pos'][0]}, "
                   f"row {f['new_pos'][1]}")
    else:
        h = f.get("heading") or f.get("heading1")
        out.append(f"R1 faces {h}")
    if "heading2" in f:
        out.append(f"R2 faces {f['heading2']}")
    for k, (fw, rt) in sorted(ego.items()):
        out.append(f"{k} is {'in front of' if fw > 0 else 'behind'} and to the {'right' if rt > 0 else 'left'} "
                   f"of R1{' (after moving)' if 'new_pos' in f else ''}")
    if "ego2" in f:
        for k, (fw, rt) in sorted(f["ego2"].items()):
            out.append(f"{k} is to R2's {'right' if rt > 0 else 'left'}")
    return out


def _search(f, inst):
    boxes = f["evidence_boxes"]
    return [f"Target: code {f['target_code']}" + (f" in {f['target_color']}" if f["target_color"] else ""),
            f"Exactly {len(boxes)} matching tags among {f['n_tags']}",
            "Matching tag locations (x,y): " + ", ".join(f"({b[0]:.0f},{b[1]:.0f})" for b in boxes)]


def _graph(f, inst):
    adj: dict[str, list[tuple[str, int]]] = {}
    for a, b, w in f["edges"]:
        adj.setdefault(a, []).append((b, w))
        adj.setdefault(b, []).append((a, w))
    src, dst, avoid = f["source"], f["target"], f.get("avoid")
    dist, prev = {src: 0}, {src: None}
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj.get(u, []):
            if v == avoid:
                continue
            if v not in dist or d + w < dist[v]:
                dist[v], prev[v] = d + w, u
                heapq.heappush(pq, (d + w, v))
    path, cur = [], dst
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    w = {(a, b): x for a, b, x in f["edges"]} | {(b, a): x for a, b, x in f["edges"]}
    out = ["A shortest path: " + " → ".join(path) + f" (length {dist[dst]})"]
    out += [f"Edge {a}–{b} has length {w[(a, b)]}" for a, b in zip(path, path[1:])]
    if "n_shortest" in f and "avoid" not in f:
        out.append(f"Number of distinct shortest paths: {f['n_shortest']}")
    return out


def _blocks(f, inst):
    out = []
    for m in f["interfaces"]:
        lo, hi = m["contact"]
        ok = m["margin"] >= 0
        out.append(f"On top of block {m['interface_on_block']}: centre of mass of the blocks above at x={m['com']:.2f}, "
                   f"contact region [{lo}, {hi}] → {'supported' if ok else 'tips'}")
    out.append("Tower is stable" if f["first_failure_block"] is None else
               f"First failure: blocks above block {f['first_failure_block']} tip off")
    return out[:MAX_FACTS]


def _insight(f, inst):
    return [f"Anomaly: {a['type'].replace('_', ' ')} at t≈{a['t']}" for a in f["anomalies"]] + \
        [f"There are exactly {len(f['anomalies'])} anomalies"]


def _external(f, inst):
    if inst.answer.kind == "rubric":
        return [f"Reference answer: {f.get('golden_response', '')}"] + [r.criterion for r in inst.rubric]
    return [f"Correct answer: {inst.answer.value}"] + ([f"Category: {f['category']}"] if f.get("category") else [])


KEY_FACTS: dict[str, Callable[[dict, TaskInstance], list[str]]] = {
    "wire_tracing": _wire, "gear_train": _gear, "maze_path": _maze, "grid_rules": _grid,
    "chart_reasoning": _chart, "counting": _counting, "lattice_geometry": _lattice, "cube_folding": _cube,
    "clock_reasoning": _clock, "receipt_reasoning": _receipt, "perspective_taking": _persp,
    "visual_search": _search, "graph_paths": _graph, "block_stability": _blocks, "chart_insight": _insight,
}


def key_facts(inst: TaskInstance) -> list[str]:
    fn = KEY_FACTS.get(inst.family, _external)
    try:
        facts = fn(inst.facts, inst)
    except (KeyError, TypeError, IndexError, StopIteration) as e:  # never block a critique on a formatting bug
        facts = [f"(key facts unavailable: {type(e).__name__})", f"Correct answer: {inst.answer.value}"]
    return facts[:MAX_FACTS]


def crop_grounding(trajectory: list[dict[str, Any]], inst: TaskInstance) -> dict[str, Any] | None:
    """Deterministic evidence grounding for agentic runs: did the agent's crops cover the gold evidence?"""
    boxes = inst.facts.get("evidence_boxes")
    crops = [e["box"] for e in trajectory if e.get("event") == "crop"]
    if not boxes:
        return {"n_crops": len(crops)} if crops else None

    def ioa(b, c):  # fraction of evidence box b inside crop c
        x0, y0 = max(b[0], min(c[0], c[2])), max(b[1], min(c[1], c[3]))
        x1, y1 = min(b[2], max(c[0], c[2])), min(b[3], max(c[1], c[3]))
        inter = max(0, x1 - x0) * max(0, y1 - y0)
        return inter / max(1e-9, (b[2] - b[0]) * (b[3] - b[1]))
    covered = [any(ioa(b, c) >= 0.5 for c in crops) for b in boxes]
    W, H = inst.facts.get("image_size", [None, None])
    area = (sum(abs(c[2] - c[0]) * abs(c[3] - c[1]) for c in crops) / (W * H)) if W and crops else None
    return {"n_crops": len(crops), "evidence_coverage": sum(covered) / len(covered),
            "crop_area_fraction": area, "n_evidence": len(boxes)}
