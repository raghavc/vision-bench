"""Maze shortest paths, verified by simulating the submitted move string.

The verifier is a *solution checker*, not answer matching: any valid path of
optimal length is accepted, so maze variants with several shortest paths are
scored fairly. Higher difficulty adds loops (several routes), keys that must be
collected first, and one-way arrows.
"""

from __future__ import annotations

import re
from collections import deque

import numpy as np

from visionbench.core.types import AnswerSpec
from visionbench.tasks.base import TaskFamily, register
from visionbench.tasks.render import canvas, finish

#           d1 d2  d3  d4  d5
SIZE =   [0, 6, 8, 10, 12, 14]
LOOPS =  [0, 0, 0.06, 0.10, 0.12, 0.14]   # fraction of extra walls knocked down
KEYS =   [0, 0, 0, 1, 1, 2]
DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


@register
class MazePath(TaskFamily):
    name = "maze_path"
    description = "Find a shortest path through a maze (optionally via keys); verified by simulation."
    skills = ["visual_tracking", "planning", "spatial_topology"]

    def build(self, rng, style, difficulty, noise_rng):
        n = SIZE[difficulty]
        # open[r][c] = set of directions you can move from (r, c)
        open_ = self._carve(rng, n)
        walls_left = [(r, c, d) for r in range(n) for c in range(n) for d in "DR"
                      if (r + DIRS[d][0] < n and c + DIRS[d][1] < n) and d not in open_[r][c]]
        rng.shuffle(walls_left)
        for r, c, d in walls_left[: int(LOOPS[difficulty] * len(walls_left))]:
            self._knock(open_, r, c, d)

        cells = [(r, c) for r in range(n) for c in range(n)]
        for _ in range(100):
            idx = rng.choice(len(cells), size=2 + KEYS[difficulty], replace=False)
            start, goal = cells[idx[0]], cells[idx[1]]
            keys = [cells[i] for i in idx[2:]]
            dist = self._bfs(open_, start)
            if dist[goal] >= n * 1.2:
                break
        best, best_order = self._optimal(open_, start, goal, keys)

        img = self._render(open_, n, start, goal, keys, style)
        img = finish(img, style, noise_rng)

        key_txt = ""
        if keys:
            names = "the cell marked K" if len(keys) == 1 else "both cells marked K1 and K2 (in either order)"
            key_txt = f" Along the way you must pass through {names} before reaching G."
        prompt = (
            "The image shows a grid maze. Black lines are walls; you can move one cell at a time up, down, left "
            f"or right, never through a wall. Start at the cell marked S and finish at the cell marked G.{key_txt} "
            "Give a shortest such route as a sequence of moves using the letters U, D, L, R."
        )
        return {
            "prompt": prompt,
            "answer": AnswerSpec(kind="custom", value={"optimal_length": best},
                                 format_hint="a string of moves such as RRDDLU (no spaces needed)"),
            "images": [img],
            "facts": {"size": n, "start": start, "goal": goal, "keys": keys, "optimal_length": best,
                      "optimal_key_order": best_order,
                      "open": {f"{r},{c}": sorted(open_[r][c]) for r in range(n) for c in range(n)}},
        }

    # ----------------------------------------------------------------- maze
    @staticmethod
    def _knock(open_, r, c, d):
        dr, dc = DIRS[d]
        opp = {"U": "D", "D": "U", "L": "R", "R": "L"}[d]
        open_[r][c].add(d)
        open_[r + dr][c + dc].add(opp)

    def _carve(self, rng, n):
        open_ = [[set() for _ in range(n)] for _ in range(n)]
        seen = np.zeros((n, n), bool)
        stack = [(int(rng.integers(n)), int(rng.integers(n)))]
        seen[stack[0]] = True
        while stack:
            r, c = stack[-1]
            nbrs = [(d, r + dr, c + dc) for d, (dr, dc) in DIRS.items()
                    if 0 <= r + dr < n and 0 <= c + dc < n and not seen[r + dr, c + dc]]
            if not nbrs:
                stack.pop()
                continue
            d, nr, nc = nbrs[int(rng.integers(len(nbrs)))]
            self._knock(open_, r, c, d)
            seen[nr, nc] = True
            stack.append((nr, nc))
        return open_

    @staticmethod
    def _bfs(open_, src):
        n = len(open_)
        dist = {src: 0}
        q = deque([src])
        while q:
            r, c = q.popleft()
            for d in open_[r][c]:
                nxt = (r + DIRS[d][0], c + DIRS[d][1])
                if nxt not in dist:
                    dist[nxt] = dist[(r, c)] + 1
                    q.append(nxt)
        return dist

    def _optimal(self, open_, start, goal, keys):
        from itertools import permutations
        if not keys:
            return self._bfs(open_, start)[goal], []
        best, order = None, None
        for perm in permutations(range(len(keys))):
            pts = [start] + [keys[i] for i in perm] + [goal]
            total = sum(self._bfs(open_, pts[i])[pts[i + 1]] for i in range(len(pts) - 1))
            if best is None or total < best:
                best, order = total, list(perm)
        return best, order

    # ---------------------------------------------------------------- render
    def _render(self, open_, n, start, goal, keys, style):
        cell = max(38, min(64, 720 // n))
        pad = 40
        W = H = pad * 2 + cell * n
        img, draw = canvas(W, H, style)
        lw = style.lw(4)
        f = style.font(int(cell * 0.45), bold=True)
        for r in range(n):
            for c in range(n):
                x, y = pad + c * cell, pad + r * cell
                if "U" not in open_[r][c]:
                    draw.line([(x, y), (x + cell, y)], fill=style.ink, width=lw)
                if "L" not in open_[r][c]:
                    draw.line([(x, y), (x, y + cell)], fill=style.ink, width=lw)
                if r == n - 1 and "D" not in open_[r][c]:
                    draw.line([(x, y + cell), (x + cell, y + cell)], fill=style.ink, width=lw)
                if c == n - 1 and "R" not in open_[r][c]:
                    draw.line([(x + cell, y), (x + cell, y + cell)], fill=style.ink, width=lw)
        def mark(rc, text, color):
            r, c = rc
            cx, cy = pad + c * cell + cell / 2, pad + r * cell + cell / 2
            draw.rounded_rectangle([cx - cell * 0.36, cy - cell * 0.36, cx + cell * 0.36, cy + cell * 0.36],
                                   radius=6, fill=color)
            draw.text((cx, cy), text, font=f, fill=(255, 255, 255), anchor="mm")
        mark(start, "S", (40, 120, 220))
        mark(goal, "G", (30, 150, 60))
        for i, k in enumerate(keys):
            mark(k, "K" if len(keys) == 1 else f"K{i + 1}", (215, 130, 20))
        return img

    # ----------------------------------------------------------------- check
    def check(self, instance, parsed):
        facts = instance.facts
        moves = re.sub(r"[^UDLR]", "", str(parsed).upper().replace("UP", "U").replace("DOWN", "D")
                       .replace("LEFT", "L").replace("RIGHT", "R"))
        pos = tuple(facts["start"])
        keys_left = {tuple(k) for k in facts["keys"]}
        for i, m in enumerate(moves):
            if m not in facts["open"][f"{pos[0]},{pos[1]}"]:
                return 0.0, {"valid": False, "failed_at_move": i, "moves": moves, "length": len(moves)}
            pos = (pos[0] + DIRS[m][0], pos[1] + DIRS[m][1])
            keys_left.discard(pos)
        reached = pos == tuple(facts["goal"]) and not keys_left
        optimal = reached and len(moves) == facts["optimal_length"]
        return float(optimal), {"valid": True, "reached_goal": pos == tuple(facts["goal"]),
                                "missed_keys": len(keys_left), "length": len(moves),
                                "optimal_length": facts["optimal_length"], "moves": moves}
