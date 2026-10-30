"""Building, saving and loading datasets of task instances.

Splits follow the vendor practice of a public dev set plus a private held-out
set:
  * dev:  seeds 0..N-1, reproducible by anyone from this repo;
  * test: seeds derived with HMAC(secret, i), so the items can't be
    reproduced without the secret in settings.toml. Rotating the secret
    gives a fresh, uncontaminated test set.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from visionbench.core.types import TaskInstance
from visionbench.tasks.base import get_family, load_all


@dataclass
class DatasetSpec:
    name: str
    families: list[str]
    difficulties: list[int]
    per_cell: int
    split: str = "dev"            # dev | test
    rerenders: int = 0            # extra style re-renders per problem (robustness / consistency)
    start: int = 0
    meta: dict = field(default_factory=dict)


def seed_for(split: str, i: int, secret: str | None) -> int:
    if split == "dev":
        return i
    if not secret:
        raise ValueError("test split needs [run] private_seed_secret in settings.toml")
    digest = hmac.new(secret.encode(), f"vb-test-{i}".encode(), hashlib.sha256).digest()
    return 10_000_000 + int.from_bytes(digest[:6], "big") % 10**12


def build(spec: DatasetSpec, out_dir: Path, secret: str | None = None, progress=None) -> Path:
    load_all()
    root = out_dir / spec.name
    (root / "images").mkdir(parents=True, exist_ok=True)
    items_path = root / "items.jsonl"
    n = 0
    t0 = time.time()
    with open(items_path, "w") as f:
        for fam_name in spec.families:
            fam = get_family(fam_name)
            for d in spec.difficulties:
                if d not in fam.difficulties:
                    continue
                for i in range(spec.start, spec.start + spec.per_cell):
                    seed = seed_for(spec.split, i, secret)
                    for r in range(spec.rerenders + 1):
                        inst = fam.generate(seed, d, render_seed=r)
                        save_instance(inst, root)
                        f.write(json.dumps(inst.to_record(), default=str) + "\n")
                        n += 1
                        if progress:
                            progress(n)
    manifest = {
        "spec": spec.__dict__, "n_items": n, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "build_seconds": round(time.time() - t0, 1),
        "family_versions": {f: get_family(f).version for f in spec.families},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return root


def save_instance(inst: TaskInstance, root: Path) -> None:
    paths = []
    for k, im in enumerate(inst.images):
        rel = f"images/{inst.uid}_{k}.png"
        im.save(root / rel, format="PNG", optimize=True)
        paths.append(rel)
    inst.image_paths = paths


def load(root: Path, families: Iterable[str] | None = None, limit: int | None = None) -> list[TaskInstance]:
    fams = set(families) if families else None
    out = []
    with open(root / "items.jsonl") as f:
        for line in f:
            inst = TaskInstance.from_record(json.loads(line))
            if fams and inst.family not in fams:
                continue
            out.append(inst)
            if limit and len(out) >= limit:
                break
    return out
