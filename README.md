# VisionBench

A verifiable **visual-reasoning** evaluation: no preference votes, just tasks whose answers can be checked. It has three parts:

- **Task.** 15 procedurally generated families with exact, solver-computed ground truth (wire tracing, gear trains, mazes, ARC-style rule induction, multi-hop charts, compositional counting, lattice geometry, cube folding, mirrored clocks, receipts, egocentric perspective, high-res visual search, graph shortest paths, statics, rubric-graded chart analysis). Difficulty runs from 1 to 5, items can be re-rendered without changing the answer, and there is a private HMAC-seeded test split. Importers for public benchmarks (VisuLogic, BabyVision, MMSI-Bench, VisualPuzzles, O3-Bench, Scale AI's HSS, EMMA, ZeroBench, MathVision) put real images through the same pipeline.
- **Harness.** Runs Claude models in direct or agentic mode (zoom tool, optional server-side code execution), with blind and caption-only controls, k samples, budgets, a full trajectory and image ledger, and resumable runs.
- **Verifier.** Programmatic checkers for every answer type, including solution checkers that accept any valid optimal answer. LLM judges (rubric, equivalence, failure attribution) are used only where they're needed, and a judge audit measures them against programmatic truth.

Results include clustered error bars, paired significance tests, pass^k reliability, calibration, IRT item analysis and a self-contained HTML report. See [METHODOLOGY.md](METHODOLOGY.md) for the design and the research behind it.

## Setup

```bash
uv venv && uv pip install -e ".[dev]" huggingface_hub pyarrow
cp config/settings.example.toml config/settings.toml   # git-ignored
# paste your key:  [anthropic] api_key = "sk-ant-..."
visionbench selftest          # offline: every family's ground truth round-trips through its verifier
```

## Quick start

```bash
visionbench families                                  # what's in the suite
visionbench preview cube_folding -d 5 --facts         # look at one item + its ground truth

visionbench build dev-small --per-cell 4 --rerenders 1   # 15 families x 5 levels x 4 x 2 renders
visionbench estimate dev-small --models claude-opus-5-5  # rough cost before spending

visionbench run dev-small --models claude-opus-5-5,claude-sonnet-5-5 \
    --samples 3 --conditions vision,blind,caption       # collect -> verify -> report
visionbench run dev-small --models claude-opus-5-5 --mode agentic   # tool-use track

visionbench verify <run> --process-judge   # perception-vs-reasoning error attribution
visionbench judge-audit <run>              # how trustworthy is the LLM judge?
open runs/<run>/report.html
```

### Public benchmarks

```bash
visionbench import visulogic --limit 300     # also: babyvision, mmsi, visualpuzzles, o3bench, hss, emma, zerobench*, mathvision
visionbench run ext-visulogic-300 --models claude-opus-5-5 --samples 3
```

\* ZeroBench is gated: accept its terms on Hugging Face and set `[huggingface] token`.

### Offline pipeline test

`mock:oracle`, `mock:noisy-0.7` and `mock:guess` are built-in fake subjects. They exercise the whole pipeline without API calls:

```bash
visionbench run dev-small --models mock:oracle,mock:noisy-0.7 --conditions vision,blind
```

## Layout

```
config/settings.example.toml   API key, subjects, judge, harness, run, prices
visionbench/tasks/families/    the 15 generators (Task)
visionbench/external/          public benchmark importers (Task)
visionbench/harness/           subject runner, prompts, images, mock subjects, estimate (Harness)
visionbench/verifiers/         answer checkers, LLM judges, judge audit (Verifier)
visionbench/analysis/          statistics, aggregation, HTML report
tests/                         determinism, oracle round-trip, verifiers, statistics, end-to-end
```
