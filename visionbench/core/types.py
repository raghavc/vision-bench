"""Core data model shared by Task, Harness and Verifier.

A TaskInstance is fully self-describing: prompt, images, the ground-truth
answer spec, the intermediate facts the generator used (for process
verification), an optional rubric, and provenance (family, version, seeds).
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from PIL import Image

AnswerKind = Literal[
    "integer",   # exact integer
    "number",    # float with abs/rel tolerance
    "choice",    # one label out of `choices`
    "text",      # normalized exact string
    "sequence",  # ordered list of tokens
    "set",       # unordered set of tokens
    "mapping",   # dict token -> token
    "grid",      # 2-D integer grid
    "custom",    # family-specific solution checker (e.g. any valid shortest path)
    "rubric",    # open-ended, graded by the rubric judge
]


@dataclass
class AnswerSpec:
    kind: AnswerKind
    value: Any
    format_hint: str
    tolerance: float = 0.0          # absolute, for "number"
    rel_tolerance: float = 0.0      # relative, for "number"
    choices: list[str] | None = None
    partial_credit: bool = False    # sequence/set/mapping/grid: report fractional score too


@dataclass
class RubricItem:
    id: str
    criterion: str
    weight: float = 1.0
    critical: bool = False          # failing a critical item fails the whole response


@dataclass
class TaskInstance:
    family: str
    version: str
    seed: int
    difficulty: int
    prompt: str
    answer: AnswerSpec
    images: list[Image.Image] = field(default_factory=list, repr=False)
    image_paths: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    rubric: list[RubricItem] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    render_seed: int = 0
    # Instances that share a cluster share the same latent problem (re-renders);
    # statistics resample clusters, not items.
    cluster: str = ""
    uid: str = ""

    def __post_init__(self) -> None:
        if not self.uid:
            self.uid = make_uid(self.family, self.version, self.seed, self.difficulty, self.render_seed)
        if not self.cluster:
            self.cluster = make_uid(self.family, self.version, self.seed, self.difficulty, 0)

    # ---- image helpers -------------------------------------------------
    def load_images(self, base: Path) -> list[Image.Image]:
        if not self.images and self.image_paths:
            self.images = [Image.open(base / p).convert("RGB") for p in self.image_paths]
        return self.images

    def png_bytes(self, i: int = 0) -> bytes:
        buf = io.BytesIO()
        self.images[i].save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    # ---- serialization -------------------------------------------------
    def to_record(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("images")
        return d

    @classmethod
    def from_record(cls, d: dict[str, Any]) -> "TaskInstance":
        d = dict(d)
        d["answer"] = AnswerSpec(**d["answer"])
        d["rubric"] = [RubricItem(**r) for r in d.get("rubric", [])]
        return cls(**d)


def make_uid(*parts: Any) -> str:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return h[:16]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens

    def cost(self, price_in: float, price_out: float) -> float:
        # Cache writes bill at 1.25x input, reads at 0.1x.
        billed_in = self.input_tokens + 1.25 * self.cache_write_tokens + 0.1 * self.cache_read_tokens
        return (billed_in * price_in + self.output_tokens * price_out) / 1e6


@dataclass
class Attempt:
    """One harness attempt of one subject on one item (one of k samples)."""

    uid: str
    family: str
    cluster: str
    difficulty: int
    model: str
    mode: str                 # direct | agentic
    condition: str            # vision | blind
    sample: int
    raw_answer: str | None = None
    confidence: float | None = None
    final_text: str = ""
    thinking_summary: str = ""
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""
    error: str | None = None
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    turns: int = 0
    tool_calls: int = 0

    def to_record(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, d: dict[str, Any]) -> "Attempt":
        d = dict(d)
        d["usage"] = Usage(**d.get("usage", {}))
        return cls(**d)

    @property
    def key(self) -> str:
        return f"{self.model}|{self.mode}|{self.condition}|{self.uid}|{self.sample}"


Outcome = Literal["correct", "incorrect", "unparseable", "refusal", "error"]


@dataclass
class Verdict:
    attempt_key: str
    uid: str
    outcome: Outcome
    score: float                     # 1/0 for binary kinds; weighted rubric fraction otherwise
    partial: float | None = None     # fractional credit, when the kind supports it
    parsed: Any = None
    details: dict[str, Any] = field(default_factory=dict)
    verifier: str = "programmatic"
    error_category: str | None = None  # from the process judge, on failures

    @property
    def correct(self) -> bool:
        return self.outcome == "correct"

    def to_record(self) -> dict[str, Any]:
        d = asdict(self)
        d["parsed"] = json.loads(json.dumps(d["parsed"], default=str))
        return d

    @classmethod
    def from_record(cls, d: dict[str, Any]) -> "Verdict":
        return cls(**d)
