# VisionBench

Verifiable visual-reasoning evaluation built around three components:

- **Task** — procedurally generated, contamination-resistant problems with exact ground truth (plus public benchmark adapters).
- **Harness** — runs a model on each task (direct or agentic with zoom/code tools), with budgets, trajectories and controls.
- **Verifier** — programmatic solution checkers first; rubric LLM-judge only where answers are open-ended.

Work in progress — full docs coming.
