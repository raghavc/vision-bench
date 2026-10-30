# VisionBench showcase run: findings (2026-10-30)

**Setup**
- **Suite:** 150 procedurally generated items (15 families × 5 difficulty levels × 2), dev split.
- **Real-image track:** 25-item stratified samples of VisuLogic and BabyVision.
- **Models:** five vision models from four vendors, called through OpenRouter at `reasoning.effort = medium` with identical prompts. Images were resized to a maximum side of 1568 px.
- **Judge:** GPT-6.1-Terra served as rubric judge, equivalence judge and reasoning critic.
- **Cost:** about **$13.40** in total for 1,771 scored attempts, plus the judges, critiques and audits.

The full interactive report is `runs/showcase/report.html` (local); every number is also in `runs/showcase/results.json`.

## 1. Leaderboard

| Model | Accuracy | 95% CI | Rank (95%) | Image withheld | pass^3 | ECE | Mean confidence | Cost (150 items) |
|---|---|---|---|---|---|---|---|---|
| Gemini-4-Argon | **88.9%** | 84.0–93.8 | 1–2 | 8.7% | **83.3%** | 8.3% | 96.3% | $2.12 |
| GPT-6.1-Terra | 85.9% | 80.2–91.1 | 1–2 | 6.7% | 75.0% | 9.9% | 94.3% | $0.73 |
| Claude Haiku 5.5 | 76.6% | 69.8–82.8 | 3–4 | 8.0% | 53.8% | **3.8%** | 76.1% | $0.24 |
| GPT-6-Luna | 71.3% | 64.0–78.0 | 3–5 | 8.0% | 57.1% | 22.5% | 92.4% | $0.19 |
| Qwen3.8-Flash | 68.8% | 61.5–76.2 | 4–5 | 9.4% | 57.9% | 11.2% | 79.1% | $0.71 |
| *Nemotron-3.5-Super-VL (free; separate run, n=129)* | *62.0%* | *53.5–70.5* | — | — | — | — | — | $0 |

How to read the columns:
- **Accuracy:** pass@1 averaged over samples per item, with cluster-bootstrap confidence intervals.
- **pass^3:** the share of items where all 3 attempts were correct, measured on 30 items.
- **Rank (95%):** bootstrap rank interval.

**The paired comparisons resolve three tiers** (sign-flip permutation tests, Holm-corrected):
- **Top tier ≈ Gemini-4-Argon, GPT-6.1-Terra.** The gap is +3.0 points, with CI −1.8 to +8.0 and p = 0.54, so the two are statistically tied.
- **The top tier beats everyone below.**
  - Gemini over Haiku: +12.3 points, CI +5.8 to +18.7, p = 0.004.
  - Terra over Haiku: +9.3, p = 0.007.
  - Each top model over Luna and Qwen: +14.6 to +20.6, p < 0.001.
- **Haiku 5.5 ≥ GPT-6-Luna ≈ Qwen3.8-Flash.** Haiku over Qwen is +7.4 with a CI excluding 0, but it's not significant after Holm correction (p = 0.10).

## 2. The eval measures vision, not language priors

With the image withheld, **every model collapses to 6.7–9.4%**. The vision − blind gap is **+60 to +80 points**, with clustered CIs:
- Gemini: +80.2 (73.3–86.7)
- Terra: +79.2 (72.8–85.4)
- Haiku: +68.6 (60.9–75.8)
- Luna: +63.3 (54.7–71.3)
- Qwen: +59.8 (52.2–67.6)

**Known weakness, flagged by the control:** `block_stability` is 30–60% solvable blind. Its answer space is "stable" or a block number, and about 45% of towers are stable. `counting` is partly guessable too, because answers are small integers. Planned fix for v2: ask for a numeric centre-of-mass position, which can't be guessed.

## 3. Difficulty works, and the models degrade differently

| Model | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| Gemini-4-Argon | 100 | 96.7 | 87.8 | 80.0 | 80.0 |
| GPT-6.1-Terra | 96.7 | 86.7 | 79.4 | 80.0 | **86.7** |
| Claude Haiku 5.5 | 90.0 | 86.7 | 71.7 | 76.7 | 57.8 |
| GPT-6-Luna | 83.3 | 76.7 | 57.8 | 70.0 | 68.9 |
| Qwen3.8-Flash | 89.7 | 75.9 | 65.0 | 56.7 | 57.8 |

Every model is near the ceiling at L1, and the spread widens with difficulty. **GPT-6.1-Terra is the most robust at the hardest level:** it scores higher at L5 than Gemini. Haiku and Qwen lose about 32 points from L1 to L5.

**Where the headroom is.** Rasch item difficulty for each family (higher = harder):

| Family | Rasch b | Accuracy range |
|---|---|---|
| `visual_search` | +2.42 | 10–53% |
| `maze_path` | +1.27 | 20–75% |
| `grid_rules` | +0.79 | 5–80% |
| `chart_insight` | +0.35 | |
| `wire_tracing` | −0.11 | 40–100% |
| `cube_folding` | −0.32 | 42–90% |

These six families separate models. **Saturated families** (all ≥ 90%: `chart_reasoning`, `graph_paths`, `lattice_geometry`, `receipt_reasoning` v2, `counting`) need harder generator levels (L6+) before they discriminate. That is an actionable result, not a hidden one.

Two notable family-level gaps:
- **Grid rule induction:** Qwen3.8-Flash scores 5% here, against 50–80% for the others.
- **Wire tracing:** Terra and Gemini score 90–100%, the rest 40–50%.

## 4. Reliability (pass^3) shows what pass@1 hides

On 30 items with 3 samples each, Gemini keeps 83% when all three attempts must be right, and Terra keeps 75%. Haiku, Luna and Qwen drop to **54–58%**, 15–23 points below their pass@1. The smaller models are right *sometimes* on many items that the large ones solve every time.

## 5. Calibration

| Model | Mean stated confidence | Accuracy | ECE |
|---|---|---|---|
| Claude Haiku 5.5 | 76% | 74% | **3.8%** |
| GPT-6-Luna | 92% | 70% | 22.5% |
| Gemini, Terra | 94–96% | 85–88% | about 8–10% |

Haiku is the best calibrated. GPT-6-Luna is the most overconfident. Gemini and Terra are mildly overconfident.

## 6. How each agent reasons (reasoning critique, 129 attempts)

A critic is given the generator's ground-truth key facts and audits each agent's visible reasoning. Its test-retest agreement is 90–100% (§8).

| | Gemini | Terra | Haiku | Luna |
|---|---|---|---|---|
| Correct answers that were *not* earned (guess or wrong steps) | **12%** | 19% | 30% | 16% |
| Key facts misread (of facts mentioned) | **4%** | 8% | 16% | 16% |
| Precision of image claims | 93% | 93% | 82% | 89% |
| Valid reasoning | **87%** | 77% | 63% | 67% |
| Failures starting with perception vs reasoning | 75 / 25 | **50 / 50** | 70 / 20 (10 no answer) | 82 / 9 (9 instruction) |
| Accuracy when it self-verified vs not | 60 vs 92 | 71 vs 91 | 69 vs 64 | 75 vs 62 |

Qwen has only 9 critiques: the run was stopped by provider stalls (§9). Read its 50% unsupported-success rate as indicative only.

- **Perception is the bottleneck.** For every model except Terra, 70–100% of failures start with misreading the image, which matches Scale's VisualToolBench finding of 70–80%. Terra perceives well enough that half its remaining errors are *reasoning* errors: premature commitment and spatial transforms.
- **Weaker models' accuracy overstates their competence.** 30% of Haiku's correct answers were not supported by valid reasoning, against 12% for Gemini.
- **Self-verification behaves differently by model.** It helps weaker models (Haiku +5 points, Luna +13). Strong models self-verify mainly when unsure, so their self-verified attempts are *less* accurate (Gemini 60% vs 92%). That is a signal of uncertainty, not a cause of errors.
- **Head-to-head.** When Gemini beat Luna, Luna's failures were 57% perception misreads. When Terra beat Haiku, Haiku's failures were misreads (43%) and tracking errors (29%).

## 7. Real images: agreement at the top, one divergence

| Model | Procedural | VisuLogic (chance 25%) | BabyVision |
|---|---|---|---|
| Gemini-4-Argon | 88.9 | **68** | 88 |
| GPT-6.1-Terra | 85.9 | 48 | **92** |
| Claude Haiku 5.5 | 76.6 | 28 | 40 |
| GPT-6-Luna | 71.3 | 44 | 60 |
| Qwen3.8-Flash | 68.8 | 48 | 56 |

The top two models agree across all three suites; the rank correlation with the external average is ρ ≈ 0.68. **Haiku is the outlier:** 3rd on the synthetic suite, but last on both real-image sets, near chance on VisuLogic. This is the "synthetic-render skill" risk the methodology warned about, now measured. With n = 25 per external set the confidence intervals are wide (±20 points), so it's a signal to follow up at larger n, not a verdict.

## 8. The eval audited itself, and these are the defects it found

A benchmark is only as good as its keys. Four mechanisms ran on live data.

**Cross-vendor consensus audit.** It flags items where models from two or more vendors give the *identical* wrong answer. In the first pass it flagged 9 items; each one was investigated by hand.

| Finding | Verdict | Action |
|---|---|---|
| `grid_rules` (v1): four models from three vendors agreed on a per-object "hollow out" reading | **Item defect**: the DSL treated touching objects of different colours as one mass | v2: **shadow rules** (alternative readings) added to the uniqueness proof |
| `receipt_reasoning` L5: three vendors read the PRICE column as line totals | **Item defect**: the header was ambiguous | v2: `UNIT PRICE` header, "quantity x unit price" stated |
| `receipt_reasoning` L3: answers off by 1 cent | **Item defect**: per-line vs per-category rounding | v2: category-level rounding stated; 1-cent tolerance |
| `cube_folding` L5: a "P" read as "d" | **Item defect**: a rotated P *is* a d (and a rotated 7 is an L) | v2/v3: glyph set restricted and stated in the prompt |
| `gear_train` L5: two belts' segments lined up into one apparent belt | **Layout defect** | v2: at most one belt per gear |
| `block_stability` L4: three vendors checked the centre of mass against the block's full width, not the contact region | **Key correct**: a shared physics misconception | kept; reported as a finding |
| `cube_folding` L2: two vendors gave the mirror-image answer | **Key correct** (folded by hand): a shared chirality error | kept |

After the fixes, 7 items remain flagged. They are either verified shared misconceptions, or coincidences on small integer answer spaces (counts, maze lengths) that the audit can't tell apart from agreement.

**Verifier spot-check.** Hand-reviewing "incorrect" verdicts found one verifier false negative: a correctly rounded gear speed (42.4 for 720/17) was rejected by a 0.1% tolerance. It's fixed: the gear checker now accepts correctly rounded answers and the prompt asks for 2 decimals.

**Rubric audit.** `chart_insight` scored about 10% for every model, *with 100% judge agreement*. The judge was consistent; the rubric was unfair, since a gradual slope change had a ±4 tolerance and a mislocated real anomaly counted as a fabrication. v2 uses per-type tolerances and separates mislocated from fabricated anomalies.

**Judge reliability.**
- The **equivalence judge** agreed with programmatic truth on 60 of 60 class-balanced attempts: accuracy 1.00, Cohen's κ = 1.00, false-accept rate 0%.
- The **reasoning critic's** test-retest agreement over 20 attempts was 90% on reasoning validity, 95% on answer support, error type and strategy, 96% on key-fact statuses, and 100% on self-verification.

## 9. Engineering issues surfaced by the live run (all fixed)

- **Stalled requests.** OpenRouter keeps idle connections alive while an upstream provider hangs, so HTTP read timeouts never fire; requests stalled for 40+ minutes. There is now a hard wall-clock timeout per call, with retry.
- **Credit reservation.** OpenRouter reserves `max_tokens × price` for every in-flight request. Fanning out the rubric judge (criteria × votes, 16k cap) hit 402 errors. Judge concurrency is now bounded, with token caps per judge type and backoff on 402/429.
- **Outages vs wrong answers.** Infrastructure errors are excluded from scoring and retried on resume; they were never counted as wrong answers.
- **Statistics bugs fixed:**
  - Rank intervals now use only subjects with comparable coverage; a partial subject had shrunk the shared item set.
  - The failure locus now comes from the critic's first-error type. Key facts include derived conclusions, which had made every failure look perceptual.

## 10. Limitations

- **Small sample sizes.** There are 150 procedural items, each an independent problem (no re-renders in this run), so the minimum detectable difference between two models is about 7–10 points. The external sets have n = 25 each. A publishable run should use the private test split at about 3,000 items (see METHODOLOGY §5).
- **One effort setting.** All models ran at `reasoning.effort = medium` through OpenRouter; providers may map that setting differently.
- **Judge family.** The judge and critic are an OpenAI model, so GPT-6.1-Terra was critiqued by its own model. Critiques are grounded in generator facts, but a cross-family critic should be added.
- **Visible reasoning only.** The critic sees visible reasoning and provider summaries, never a raw chain of thought.
- **Inconclusive caption control.** The caption-only control ran on only 21 items for Gemini and 2 for Terra before its budget cap, so no conclusion is drawn from it.
- **Not yet run:** the agentic (zoom) mode was not exercised on OpenRouter, and there is no human baseline yet (`visionbench human-form` is ready).

## 11. Addendum: why high single-item accuracy doesn't carry over to long tasks

This analysis was run after the report, on the same data, with no new API calls.

**By task type** (sample 0, vision):

| Model | Fully specified, short chain | Rule induction / 3-D (grid_rules, cube_folding) | Find-what-matters / long tracking (visual_search, maze_path, wire_tracing) |
|---|---|---|---|
| Gemini-4-Argon | 100% | 85% | 63% |
| GPT-6.1-Terra | 100% | 75% | 63% |
| Claude Haiku 5.5 | 96% | 65% | 47% |
| GPT-6-Luna | 96% | 70% | 27% |
| Qwen3.8-Flash | 97% | 22% | 30% |

"Fully specified, short chain" covers charts, graphs, lattice geometry, receipts, counting, gears and perspective.

**Compounding.** A synthetic long task chains k items, each from a different family, and counts as solved only if every step is correct. Item outcomes are resampled within each family, steps are treated as independent, and each figure uses 4,000 draws.

| Steps (k) | Gemini | Terra | Haiku | Luna | Qwen |
|---|---|---|---|---|---|
| 1 | 88% | 87% | 78% | 71% | 69% |
| 5 | 53% | 47% | 25% | 16% | 13% |
| 15 | 10% | 6% | 1% | 0% | 0% |

The k = 15 figure falls below the uniform-difficulty estimate (0.88¹⁵ ≈ 15%) because the hard families dominate the product. Real tasks propagate early errors, so these numbers are optimistic. This is the main reason strong single-item scores sit alongside low success on long, multi-criterion tasks (APEX averages about 15 rubric criteria per task).
