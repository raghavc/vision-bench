"""Weighted-graph reasoning from a drawn network diagram.

The model has to read the topology (which nodes connect), associate each weight
label with the right edge (labels sit on the edges, and edges cross at higher
levels), then run shortest-path reasoning. Ground truth comes from Dijkstra.
Variants: shortest distance, number of distinct shortest paths, and shortest
distance avoiding a node.
"""

from __future__ import annotations

import heapq
import math
import string

import numpy as np

from visionbench.core.types import AnswerSpec
from visionbench.tasks.base import TaskFamily, register
from visionbench.tasks.render import canvas, finish

#           d1 d2  d3  d4  d5
N_NODES = [0, 6, 8, 10, 12, 14]
K_NEAR = [0, 2, 2, 3, 3, 3]
ALLOW_CROSS = [0, 0, 0, 1, 1, 1]
QUESTION = [None, "dist", "dist", "dist", "count", "avoid"]


def seg_intersect(p1, p2, p3, p4) -> bool:
    def o(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if len({p1, p2, p3, p4}) < 4:
        return False
    return o(p3, p4, p1) * o(p3, p4, p2) < 0 and o(p1, p2, p3) * o(p1, p2, p4) < 0


def dijkstra(adj, src, banned=None):
    dist = {src: 0}
    count = {src: 1}
    pq = [(0, src)]
    done = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in done:
            continue
        done.add(u)
        for v, w in adj[u]:
            if v == banned:
                continue
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v], count[v] = nd, count[u]
                heapq.heappush(pq, (nd, v))
            elif nd == dist[v]:
                count[v] += count[u]
    return dist, count


@register
class GraphPaths(TaskFamily):
    name = "graph_paths"
    description = "Shortest-path reasoning over a drawn weighted graph (Dijkstra-verified)."
    skills = ["diagram_parsing", "planning", "arithmetic"]

    def build(self, rng, style, difficulty, noise_rng):
        W, H = 1100, 820
        n = N_NODES[difficulty]
        names = list(string.ascii_uppercase[:n])
        for _ in range(500):
            pts = self._points(rng, n, W, H)
            edges = self._edges(rng, pts, K_NEAR[difficulty], bool(ALLOW_CROSS[difficulty]))
            if edges is None:
                continue
            adj = {i: [] for i in range(n)}
            for a, b, w in edges:
                adj[a].append((b, w))
                adj[b].append((a, w))
            if len(dijkstra(adj, 0)[0]) < n:
                continue
            q = self._question(rng, QUESTION[difficulty], adj, names, n)
            if q is not None:
                break
        else:  # pragma: no cover
            raise RuntimeError("graph generation failed")
        text, value, facts = q
        img = self._render(pts, edges, names, style, W, H)
        img = finish(img, style, noise_rng)
        facts.update({"edges": [(names[a], names[b], w) for a, b, w in edges]})
        return {
            "prompt": "The diagram shows an undirected graph; the number on each edge is its length. " + text,
            "answer": AnswerSpec(kind="integer", value=int(value), format_hint="a single integer"),
            "images": [img],
            "facts": facts,
        }

    def _points(self, rng, n, W, H):
        pts = []
        while len(pts) < n:
            p = (float(rng.uniform(80, W - 80)), float(rng.uniform(80, H - 80)))
            if all(math.dist(p, q) > 150 for q in pts):
                pts.append(p)
        return pts

    def _edges(self, rng, pts, k, allow_cross):
        n = len(pts)
        cand = set()
        for i in range(n):
            near = sorted(range(n), key=lambda j: math.dist(pts[i], pts[j]))[1:k + 1]
            for j in near:
                cand.add((min(i, j), max(i, j)))
        if allow_cross:
            for _ in range(2):
                i, j = rng.choice(n, 2, replace=False)
                cand.add((int(min(i, j)), int(max(i, j))))
        edges = []
        for a, b in sorted(cand):
            # edges must not pass too close to a third node (would look connected)
            if any(self._pt_seg(pts[c], pts[a], pts[b]) < 34 for c in range(n) if c not in (a, b)):
                continue
            if not allow_cross and any(seg_intersect(pts[a], pts[b], pts[c], pts[d]) for c, d, _ in edges):
                continue
            edges.append((a, b, int(rng.integers(1, 16))))
        return edges if len(edges) >= n else None

    @staticmethod
    def _pt_seg(p, a, b):
        ax, ay = a
        bx, by = b
        px, py = p
        vx, vy = bx - ax, by - ay
        t = max(0, min(1, ((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy)))
        return math.hypot(ax + t * vx - px, ay + t * vy - py)

    def _question(self, rng, q, adj, names, n):
        s, t = (int(x) for x in rng.choice(n, 2, replace=False))
        dist, count = dijkstra(adj, s)
        hops = self._hops(adj, s, t)
        if hops < 3:
            return None
        if q == "dist":
            return (f"What is the length of the shortest path from {names[s]} to {names[t]}?", dist[t],
                    {"source": names[s], "target": names[t], "distance": dist[t], "n_shortest": count[t]})
        if q == "count":
            if count[t] < 2:
                return None
            return (f"How many different shortest paths are there from {names[s]} to {names[t]}?", count[t],
                    {"source": names[s], "target": names[t], "distance": dist[t], "n_shortest": count[t]})
        # avoid a node that lies on every shortest path, so the detour matters
        cands = [v for v in range(n) if v not in (s, t)]
        rng.shuffle(cands)
        for v in cands:
            d2, _ = dijkstra(adj, s, banned=v)
            if t in d2 and d2[t] > dist[t]:
                return (f"What is the length of the shortest path from {names[s]} to {names[t]} that does not "
                        f"pass through {names[v]}?", d2[t],
                        {"source": names[s], "target": names[t], "avoid": names[v], "distance_unrestricted": dist[t],
                         "distance_avoiding": d2[t]})
        return None

    @staticmethod
    def _hops(adj, s, t):
        from collections import deque
        seen = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            for v, _ in adj[u]:
                if v not in seen:
                    seen[v] = seen[u] + 1
                    q.append(v)
        return seen.get(t, 0)

    def _render(self, pts, edges, names, style, W, H):
        img, draw = canvas(W, H, style)
        for a, b, _ in edges:
            draw.line([pts[a], pts[b]], fill=(90, 90, 90), width=style.lw(3))
        f = style.font(20, bold=True)
        boxes = []
        for a, b, w in edges:
            # place the label at the first spot along its edge that is clear of other labels and other edges
            best = None
            for t in (0.5, 0.42, 0.58, 0.35, 0.65, 0.28, 0.72):
                x = pts[a][0] + t * (pts[b][0] - pts[a][0])
                y = pts[a][1] + t * (pts[b][1] - pts[a][1])
                box = (x - 17, y - 14, x + 17, y + 14)
                free = all(box[2] < o[0] or box[0] > o[2] or box[3] < o[1] or box[1] > o[3] for o in boxes)
                clear = min((self._pt_seg((x, y), pts[c], pts[d]) for c, d, _ in edges if (c, d) != (a, b)),
                            default=99)
                if free and clear > 24:
                    best = (x, y, box)
                    break
                if best is None and free:
                    best = (x, y, box)
            x, y, box = best if best else (x, y, box)
            boxes.append(box)
            draw.rounded_rectangle(box, radius=6, fill=(255, 255, 255), outline=(150, 150, 150))
            draw.text((x, y), str(w), font=f, fill=(170, 30, 30), anchor="mm")
        for i, p in enumerate(pts):
            draw.ellipse([p[0] - 24, p[1] - 24, p[0] + 24, p[1] + 24], fill=(214, 228, 248), outline=style.ink,
                         width=3)
            draw.text(p, names[i], font=style.font(24, bold=True), fill=style.ink, anchor="mm")
        return img
