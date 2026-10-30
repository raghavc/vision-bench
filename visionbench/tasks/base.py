"""Task family interface and registry."""

from __future__ import annotations

import zlib
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from visionbench.core.types import AnswerSpec, RubricItem, TaskInstance
from visionbench.tasks.render import Style


class TaskFamily(ABC):
    """A parameterized generator of verifiable visual-reasoning problems.

    `generate(seed, difficulty, render_seed)` is a pure function: the same
    arguments always give the same instance, so a dataset can be rebuilt
    from its manifest, and a private split is just a private seed range.
    """

    name: str = ""
    version: str = "1"
    description: str = ""
    skills: list[str] = []
    difficulties: tuple[int, ...] = (1, 2, 3, 4, 5)

    def generate(self, seed: int, difficulty: int, render_seed: int = 0) -> TaskInstance:
        if difficulty not in self.difficulties:
            raise ValueError(f"{self.name}: difficulty {difficulty} not in {self.difficulties}")
        base = zlib.crc32(f"{self.name}:{self.version}".encode())
        rng = np.random.default_rng([base, seed, difficulty])
        style = Style.plain() if render_seed == 0 else Style.sample(
            np.random.default_rng([base, seed, difficulty, render_seed, 7]))
        noise_rng = np.random.default_rng([base, seed, render_seed, 11])
        out = self.build(rng, style, difficulty, noise_rng)
        return TaskInstance(
            family=self.name,
            version=self.version,
            seed=seed,
            difficulty=difficulty,
            render_seed=render_seed,
            prompt=out["prompt"],
            answer=out["answer"],
            images=out["images"],
            facts=out.get("facts", {}),
            rubric=out.get("rubric", []),
            skills=out.get("skills", list(self.skills)),
        )

    @abstractmethod
    def build(self, rng: np.random.Generator, style: Style, difficulty: int,
              noise_rng: np.random.Generator) -> dict[str, Any]:
        """Return {prompt, answer: AnswerSpec, images: [PIL], facts, rubric?, skills?}."""

    def check(self, instance: TaskInstance, parsed: Any) -> tuple[float, dict[str, Any]]:
        """Solution checker for answer kind 'custom'. Returns (score in [0,1], details)."""
        raise NotImplementedError(f"{self.name} has no custom checker")


REGISTRY: dict[str, TaskFamily] = {}


def register(cls: type[TaskFamily]) -> type[TaskFamily]:
    inst = cls()
    if not inst.name:
        raise ValueError(f"{cls.__name__} has no name")
    REGISTRY[inst.name] = inst
    return cls


def get_family(name: str) -> TaskFamily:
    if not REGISTRY:
        load_all()
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown task family {name!r}; known: {sorted(REGISTRY)}") from None


def load_all() -> dict[str, TaskFamily]:
    from visionbench.tasks import families  # noqa: F401  (registers on import)
    return REGISTRY


__all__ = ["TaskFamily", "AnswerSpec", "RubricItem", "register", "get_family", "load_all", "REGISTRY"]
