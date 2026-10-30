"""Cube nets: mental folding and 3-D rotation, rendered with correct chirality.

The net is folded by "rolling" a cube across it, which tracks every face's
outward normal and the orientation of its printed glyph. A view of the folded
cube is drawn in true isometric projection, with glyphs affine-warped onto the
visible faces. Questions are free-form (no multiple-choice guessing floor):
the opposite face, a hidden face given two visible faces (needs chirality),
all hidden faces, and the top face after a sequence of rotations.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from PIL import Image, ImageDraw

from visionbench.core.types import AnswerSpec
from visionbench.tasks.base import TaskFamily, register
from visionbench.tasks.render import NAMED_COLORS, canvas, finish, font

# The 11 cube nets, as (row, col) cells.
NETS = [
    [(0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (2, 1)],
    [(0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (2, 2)],
    [(0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (2, 3)],
    [(0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0)],
    [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (2, 3)],
    [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0)],
    [(0, 2), (1, 0), (1, 1), (1, 2), (1, 3), (2, 1)],  # 1-4-1 variants above, 2-3-1 / 3-3 / 2-2-2 below
    [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 3)],
    [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)],
    [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4)],
    [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 1)],
]

FACE_COLORS = ["red", "blue", "green", "yellow", "purple", "orange"]
# No mirror symmetry (chirality stays visible) and no two glyphs alike under rotation ("7" was dropped
# in v2: rotated 180 degrees it reads as an "L").
LETTERS = ["F", "G", "J", "L", "P", "Q", "R", "4"]

DIR_VEC = {(0, 1): np.array([1, 0, 0]), (0, -1): np.array([-1, 0, 0]),
           (1, 0): np.array([0, -1, 0]), (-1, 0): np.array([0, 1, 0])}
UP = np.array([0, 0, 1])
PAPER_UP = np.array([0, 1, 0])
VIEW_DIRS = {"top": np.array([0, 0, 1]), "bottom": np.array([0, 0, -1]), "front": np.array([0, -1, 0]),
             "back": np.array([0, 1, 0]), "right": np.array([1, 0, 0]), "left": np.array([-1, 0, 0])}


def roll_matrix(d: np.ndarray) -> np.ndarray:
    """Rotation taking direction d -> +z and +z -> -d (cube rolling under the paper toward d)."""
    a = np.cross(d, UP)
    src = np.stack([d, UP, a], 1)
    dst = np.stack([UP, -d, a], 1)
    return (dst @ np.linalg.inv(src)).round().astype(int)


def all_rotations() -> list[np.ndarray]:
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            m = np.zeros((3, 3), int)
            for i, (p, s) in enumerate(zip(perm, signs)):
                m[i, p] = s
            if round(np.linalg.det(m)) == 1:
                mats.append(m)
    return mats


ROTATIONS = all_rotations()


def fold(net: list[tuple[int, int]]) -> dict[tuple, dict]:
    """Return {local_normal: {"cell": idx, "up": local_up_vector}} for the folded cube."""
    cells = {c: i for i, c in enumerate(net)}
    faces: dict[tuple, dict] = {}
    stack = [(net[0], np.eye(3, dtype=int))]
    seen = {net[0]}
    while stack:
        cell, R = stack.pop()
        n_local = tuple(int(v) for v in R.T @ UP)
        faces[n_local] = {"cell": cells[cell], "up": tuple(int(v) for v in R.T @ PAPER_UP)}
        for (dr, dc), vec in DIR_VEC.items():
            nb = (cell[0] + dr, cell[1] + dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append((nb, roll_matrix(vec) @ R))
    assert len(faces) == 6, "invalid net"
    return faces


# Isometric projection, viewer at (+x, -y, +z)
SCR_R = np.array([1, 1, 0]) / math.sqrt(2)
SCR_U = np.array([-1, 1, 2]) / math.sqrt(6)

#            d1          d2           d3          d4          d5
QUESTION = [None, "opposite", "hidden_one", "hidden_one", "hidden_all", "rotations"]
GLYPHS = [0, 0, 0, 1, 1, 1]


@register
class CubeFolding(TaskFamily):
    name = "cube_folding"
    version = "2"
    description = "Fold a cube net mentally; reason about hidden faces and rotations (chirality-correct)."
    skills = ["spatial_3d", "mental_rotation"]

    def build(self, rng, style, difficulty, noise_rng):
        net = NETS[int(rng.integers(len(NETS)))]
        # random rotation/reflection of the net layout on paper (still a valid net)
        net = self._transform_net(net, rng)
        faces = fold(net)
        glyph = bool(GLYPHS[difficulty])
        names = [str(x) for x in (rng.permutation(LETTERS)[:6] if glyph else rng.permutation(FACE_COLORS))]
        cell_rot = [int(rng.integers(4)) if glyph else 0 for _ in range(6)]  # glyph rotation on paper (x90°)
        Q = ROTATIONS[int(rng.integers(len(ROTATIONS)))]

        def face_at(view_dir, Qm):
            for n_local, info in faces.items():
                if tuple(Qm @ np.array(n_local)) == tuple(view_dir):
                    return info
            raise AssertionError

        def name_at(view_dir, Qm=None):
            return names[face_at(view_dir, Q if Qm is None else Qm)["cell"]]

        q = QUESTION[difficulty]
        images = []
        net_img = self._render_net(net, names, cell_rot, glyph, style)
        word = "letter" if glyph else "colour"
        if q == "opposite":
            i = int(rng.integers(6))
            n_local = next(n for n, f in faces.items() if f["cell"] == i)
            opp = faces[tuple(-v for v in n_local)]
            prompt = (f"The net shown folds into a cube (printed side outward). Which {word} ends up on the face "
                      f"opposite the {names[i]} face?")
            answer = names[opp["cell"]]
            images = [net_img]
            spec = AnswerSpec(kind="text", value=answer, format_hint=f"a single {word}")
            facts = {}
        else:
            view = self._render_view(faces, names, cell_rot, glyph, Q, style, hide_right=(q.startswith("hidden_one")))
            images = [self._side_by_side(net_img, view, style)]
            if q == "hidden_one":
                prompt = (f"The net on the left folds into a cube (printed side outward). On the right is a view of "
                          f"that cube showing its top face and front face; the right-hand face is hidden. Which "
                          f"{word} is on the face marked '?'")
                spec = AnswerSpec(kind="text", value=name_at(VIEW_DIRS["right"]), format_hint=f"a single {word}")
            elif q == "hidden_all":
                prompt = (f"The net on the left folds into a cube (printed side outward). On the right is a view of "
                          f"that cube showing its top, front and right faces. Which {word}s are on the hidden back, "
                          f"left and bottom faces?")
                val = {k: name_at(VIEW_DIRS[k]) for k in ("back", "left", "bottom")}
                spec = AnswerSpec(kind="mapping", value=val, partial_credit=True,
                                  format_hint="back=X, left=Y, bottom=Z")
            else:  # rotations
                ops = self._ops(rng, 3)
                Qm = Q
                for _, M in ops:
                    Qm = M @ Qm
                desc = "; then ".join(t for t, _ in ops)
                prompt = (f"The net on the left folds into a cube (printed side outward). On the right is a view of "
                          f"that cube showing its top, front and right faces. Starting from that position: {desc}. "
                          f"Which {word} is now on top, and which is now in front?")
                val = {"top": name_at(VIEW_DIRS["top"], Qm), "front": name_at(VIEW_DIRS["front"], Qm)}
                spec = AnswerSpec(kind="mapping", value=val, partial_credit=True, format_hint="top=X, front=Y")
            facts = {"view": {k: name_at(v) for k, v in VIEW_DIRS.items()}}
        images = [finish(im, style, noise_rng) for im in images]
        facts.update({"net_cells": net, "names": names})
        return {"prompt": prompt, "answer": spec, "images": images, "facts": facts}

    @staticmethod
    def _ops(rng, k):
        pool = [
            ("tip the cube forward a quarter turn (the top face rolls toward you to become the front)"),
            ("tip the cube backward a quarter turn (the front face rolls up to become the top)"),
            ("turn the cube a quarter turn about the vertical axis so the left face comes to the front"),
            ("turn the cube a quarter turn about the vertical axis so the right face comes to the front"),
            ("roll the cube a quarter turn to the right (the top face becomes the right face)"),
        ]
        maps = {
            0: (VIEW_DIRS["top"], VIEW_DIRS["front"]),
            1: (VIEW_DIRS["front"], VIEW_DIRS["top"]),
            2: (VIEW_DIRS["left"], VIEW_DIRS["front"]),
            3: (VIEW_DIRS["right"], VIEW_DIRS["front"]),
            4: (VIEW_DIRS["top"], VIEW_DIRS["right"]),
        }
        out = []
        for _ in range(k):
            i = int(rng.integers(len(pool)))
            src, dst = maps[i]
            M = next(R for R in ROTATIONS if tuple(R @ src) == tuple(dst)
                     and (R @ np.cross(src, dst) == np.cross(src, dst)).all())
            out.append((pool[i], M))
        return out

    @staticmethod
    def _transform_net(net, rng):
        pts = np.array(net)
        k = int(rng.integers(4))
        for _ in range(k):
            pts = np.stack([pts[:, 1], -pts[:, 0]], 1)
        if rng.random() < 0.5:
            pts[:, 1] = -pts[:, 1]
        pts -= pts.min(0)
        order = rng.permutation(6)
        return [tuple(int(v) for v in pts[i]) for i in order]

    # ---------------------------------------------------------------- render
    def _texture(self, name, glyph, rot, size=200):
        tex = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(tex)
        if glyph:
            d.rectangle([0, 0, size, size], fill=(245, 238, 220, 255), outline=(30, 30, 30, 255), width=6)
            d.text((size / 2, size / 2), name, font=font(int(size * 0.62), "DejaVu Sans", True),
                   fill=(30, 30, 30, 255), anchor="mm")
            # small bar marks the glyph's top edge (helps orientation reasoning)
            tex = tex.rotate(-90 * rot)
        else:
            d.rectangle([0, 0, size, size], fill=NAMED_COLORS[name] + (255,), outline=(30, 30, 30, 255), width=6)
        return tex

    def _render_net(self, net, names, cell_rot, glyph, style):
        cs = 110
        rows = max(r for r, _ in net) + 1
        cols = max(c for _, c in net) + 1
        W, H = cols * cs + 80, rows * cs + 80
        img, _ = canvas(W, H, style)
        for i, (r, c) in enumerate(net):
            tex = self._texture(names[i], glyph, cell_rot[i], cs).convert("RGBA")
            img.paste(tex, (40 + c * cs, 40 + r * cs), tex)
        return img

    def _render_view(self, faces, names, cell_rot, glyph, Q, style, hide_right=False):
        W = H = 460
        img, draw = canvas(W, H, style)
        scale = 150
        cx, cy = W / 2, H / 2 + 10

        def proj(p):
            return (cx + scale * float(p @ SCR_R), cy - scale * float(p @ SCR_U))

        for view_name in ("top", "front", "right"):
            vd = VIEW_DIRS[view_name]
            n_local, info = next((n, f) for n, f in faces.items() if tuple(Q @ np.array(n)) == tuple(vd))
            up_view = Q @ np.array(info["up"])
            # glyph rotation on the paper rotates its texture's up direction within the face plane
            rot = cell_rot[info["cell"]]
            right_view = np.cross(up_view, vd)
            for _ in range(rot):
                up_view, right_view = right_view, -up_view
            if hide_right and view_name == "right":
                c3 = vd * 0.5
                corners = [proj(c3 + a * 0.5 * u + b * 0.5 * v) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))
                           for u, v in [(np.cross(vd, [0, 0, 1]) if abs(vd[2]) < 1 else np.array([1, 0, 0]),
                                         np.array([0, 0, 1]) if abs(vd[2]) < 1 else np.array([0, 1, 0]))]]
                draw.polygon(corners, fill=(225, 225, 225), outline=(30, 30, 30), width=4)
                c2 = proj(c3)
                draw.text(c2, "?", font=font(64, "DejaVu Sans", True), fill=(30, 30, 30), anchor="mm")
                continue
            tex = self._texture(names[info["cell"]], glyph, 0, 200)
            # affine map texture (u right, v down) -> screen
            c3 = vd * 0.5
            o = np.array(proj(c3 - 0.5 * right_view + 0.5 * up_view))       # texture (0,0) = top-left
            ex = np.array(proj(c3 + 0.5 * right_view + 0.5 * up_view)) - o  # +u
            ey = np.array(proj(c3 - 0.5 * right_view - 0.5 * up_view)) - o  # +v
            A = np.array([[ex[0], ey[0]], [ex[1], ey[1]]]) / 200.0
            Ainv = np.linalg.inv(A)
            # output (X,Y) -> input (u,v) = Ainv @ ((X,Y) - o)
            a, b = Ainv[0]
            c, d = Ainv[1]
            coeffs = (a, b, -(a * o[0] + b * o[1]), c, d, -(c * o[0] + d * o[1]))
            warped = tex.transform((W, H), Image.AFFINE, coeffs, resample=Image.BICUBIC)
            img.paste(warped, (0, 0), warped)
        return img

    def _side_by_side(self, a, b, style):
        W = a.width + b.width + 60
        H = max(a.height, b.height) + 40
        img, draw = canvas(W, H, style)
        img.paste(a, (20, (H - a.height) // 2))
        img.paste(b, (a.width + 40, (H - b.height) // 2))
        draw.line([(a.width + 30, 20), (a.width + 30, H - 20)], fill=(180, 180, 180), width=2)
        return img
