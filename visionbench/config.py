"""Settings loading: config/settings.toml, falling back to the example file and env vars."""

from __future__ import annotations

import copy
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
SETTINGS_PATH = CONFIG_DIR / "settings.toml"
EXAMPLE_PATH = CONFIG_DIR / "settings.example.toml"


@dataclass
class Settings:
    raw: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None

    def section(self, name: str) -> dict[str, Any]:
        return self.raw.get(name, {})

    @property
    def api_key(self) -> str | None:
        key = self.section("anthropic").get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        return key or None

    @property
    def subject(self) -> dict[str, Any]:
        return self.section("subject")

    @property
    def judge(self) -> dict[str, Any]:
        return self.section("judge")

    @property
    def harness(self) -> dict[str, Any]:
        return self.section("harness")

    @property
    def run(self) -> dict[str, Any]:
        return self.section("run")

    def price(self, model: str) -> tuple[float, float]:
        p = self.section("prices").get(model)
        if not p:
            return (0.0, 0.0)
        return (float(p["input"]), float(p["output"]))

    def path(self, key: str) -> Path:
        p = Path(self.run.get(key, key))
        return p if p.is_absolute() else ROOT / p

    def with_overrides(self, overrides: dict[str, dict[str, Any]]) -> "Settings":
        raw = copy.deepcopy(self.raw)
        for sec, values in overrides.items():
            raw.setdefault(sec, {}).update({k: v for k, v in values.items() if v is not None})
        return Settings(raw=raw, source=self.source)


def load_settings(path: str | Path | None = None) -> Settings:
    candidates = [Path(path)] if path else [SETTINGS_PATH, EXAMPLE_PATH]
    for p in candidates:
        if p.exists():
            with open(p, "rb") as f:
                return Settings(raw=tomllib.load(f), source=p)
    raise FileNotFoundError(f"No settings file found (looked in {', '.join(map(str, candidates))})")
