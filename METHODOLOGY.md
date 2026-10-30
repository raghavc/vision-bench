# VisionBench methodology

VisionBench measures **visual reasoning**: whether a model can extract the right facts from pixels and reason over them to an answer that can be checked. It is not a preference arena. No votes or "which answer looks better": every headline number comes from a verifier.

The design comes from a survey of frontier multimodal benchmarks and eval vendors (MMMU-Pro, MathVerse, CharXiv, BLINK, VisuLogic, ZeroBench, EMMA, VisualPuzzles, HLE, MME-CoT, BabyVision, MMSI-Bench, O3-Bench, TIR-Bench, Scale SEAL / VisualToolBench / HSS, Mercor APEX, Epoch AI, Artificial Analysis, LMArena). Their recurring failure modes, and the response in VisionBench:

| Failure mode seen in the field | Evidence | VisionBench response |
|---|---|---|
| Items solvable without the image | MMStar: GeminiPro gets 42.7% on MMMU blind; TsT stress tests | Every run includes a **blind** control and an optional **caption-only** control. Blind-solvable items are flagged. |
| Multiple-choice guessing floor and saturation | VisuLogic's 25% floor; BLINK and MMVP | Procedural tasks are **free-form**, with typed answers (integer, grid, path, set, mapping). MC external sets report **chance-normalized** accuracy. |
| Wrong answer keys | HLE: 18–29% flawed; FrontierMath: about 1/3 | Ground truth is **computed by the generator** (a solver, simulator or Dijkstra), never hand-typed. IRT flags suspect keys in external sets. |
| Contamination of static public sets | Saturation within 12–24 months | Generators are **seeded and parameterized**: unlimited fresh items, plus an HMAC-seeded **private test split**. Rotating the secret gives a new test set. |
| Ambiguous items | Too-close-to-call layouts | Each family has an **ambiguity filter**: uniqueness proof for ARC rules, legibility checks for wires, angular margins for perspective, centre-of-mass margins for statics, no ties in chart argmaxes. |
| LLM judges are noisy and lenient | VBVR-Pro (5–93% score flips); MAM-AI | Programmatic verification wherever possible. Judges only grade rubric items, which are atomic, binary, reference-guided, one criterion per call, k-sample majority vote and critical-gated. A **judge audit** reports agreement, κ and false-accept rate against programmatic truth. |
| No error bars, pooled samples | Miller (2024), "Adding Error Bars to Evals" | Samples are averaged per item first. **Cluster-robust SEs** and cluster bootstrap CIs. Paired model differences with permutation p-values and Holm correction. Rank intervals. MDE. |
| Lucky single samples | ZeroBench pass@5 vs pass^5 | k samples per item, with unbiased **pass@k** and **pass^k** (all-k-correct reliability). |
| Resolution as a hidden confounder | O3-Bench: about 10 points from the image cap | The image cap is a fixed, logged setting (`max_image_side`). The sent size is recorded per attempt. High-res tasks are a family of their own. |
| Ungrounded agent success | ViEBench: 57% of correct answers backed by the wrong crop | The agentic harness logs every crop to an image ledger. Visual-search items carry gold evidence boxes for grounding analysis. |

## 1. Task

### 1.1 Procedural families (15)

Each family is a pure function `generate(seed, difficulty, render_seed) -> TaskInstance`. The problem RNG and the style RNG are separate. A re-render with a different `render_seed` changes the palette, fonts, stroke widths, paper tone and sensor noise but keeps the answer. Re-renders share a **cluster**, which drives both the clustered statistics and the **render-consistency** metric (solved on every rendering).

| Family | Construct | Ground truth | Difficulty knobs |
|---|---|---|---|
| `wire_tracing` | continuous visual tracking through crossings | permutation from the generator | wires 4→11, control points, single-colour wires, multi-wire mapping |
| `gear_train` | physical/causal propagation through a topology | kinematics: mesh reverses, crossed belt reverses, speed ∝ teeth ratio | idlers, branches, open/crossed belts, near-miss (gapped) decoy gears |
| `maze_path` | planning over a perceived graph | BFS optimum; **solution checker** accepts any optimal path | size 6→14, loops (several routes), key cells to visit |
| `grid_rules` | abstract rule induction (ARC-style) from images only | DSL program; **uniqueness proof**: every same-depth program consistent with the examples predicts the same test output | program depth 1→3, grid size, example count |
| `chart_reasoning` | multi-hop quantitative chart reading | values on gridlines; no ties | value labels removed, minor-gridline precision, dual axis, stacked segments |
| `counting` | compositional counting with attribute binding | object list | count 12→44, occlusion ≤25%, negation, region predicate |
| `lattice_geometry` | measurement and coordinate reading, then exact geometry | shoelace / Pick's theorem | vertex count, non-convexity, sparse axis labels, holes |
| `cube_folding` | 3-D mental folding and rotation, with chirality | rolling-cube fold of the net, rotation group | opposite face → hidden face (chirality) → all hidden faces → face after a rotation sequence |
| `clock_reasoning` | analog time reading, elapsed time | constructed times | no numerals, Roman numerals, second-hand distractor, **mirror-image** clock |
| `receipt_reasoning` | document OCR plus policy arithmetic | exact decimal arithmetic (half-up cents) | voided lines, no printed line totals, category discount, category tax, rotation |
| `perspective_taking` | allocentric → egocentric transforms | vector geometry with 15° decision margins | diagonal headings, turn-and-move, two robots |
| `visual_search` | high-resolution search for tiny confusable targets | placed-tag list with evidence boxes | image width 1600→3600 px, font 22→12 px, colour-conditioned targets |
| `graph_paths` | diagram parsing (edge–label association) plus shortest paths | Dijkstra | node count, crossing edges, count of shortest paths, node avoidance |
| `block_stability` | intuitive statics | centre of mass above the contact region at every interface, with margin | block count, printed masses vs. equal density |
| `chart_insight` | open-ended analysis (rubric track) | injected anomaly list → rubric of atomic criteria and a critical no-fabrication item | anomaly count, noise |

**Construct validity.** Each family targets one named skill, listed in `skills` on every item. Scores are reported **per family**; the headline average is never the only number. Within each family, difficulty comes from generator parameters, not from filtering against particular models. Adversarial filtering against a few models biases rankings against those models.

### 1.2 Public benchmark track

The procedural suite is complemented by public, human-built benchmarks (`visionbench import <source>`), selected for reasoning headroom in 2026:

- VisuLogic
- BabyVision
- MMSI-Bench
- VisualPuzzles
- O3-Bench (high-res, tool-friendly)
- Scale AI's Humanity's Sixth Sense (image subset, rubric-graded)
- EMMA
- ZeroBench (gated)
- MathVision-testmini, as a near-saturated **calibration anchor**

Imports use stratified sampling by category with a fixed seed, pin the dataset commit SHA, and store chance level per item. Data stays local and is never committed; each dataset's license and terms apply. HSS media are non-commercial research only.

Saturated or knowledge-heavy sets (MMMU, MathVista, CharXiv, V*) are intentionally not in the default suite.

## 2. Harness

- **Direct mode:** one call. A neutral system prompt, the image(s) at a fixed resolution cap, and the answer protocol: brief key observations, then `<answer>` and then `<confidence>` (0–100). The same prompt is used for every model.
- **Agentic mode:** a tool loop with
  - `zoom`: crops in original-pixel coordinates, upsampled up to 4×;
  - optional Anthropic server-side `code_execution`, with the images uploaded through the Files API;
  - `submit_answer`.

  Budgets: `max_turns` plus `max_tokens` per turn. The conversation is append-only, as preserved thinking requires. Every crop is saved to the run's `ledger/`.
- **Controls:**
  - `blind`: the image is withheld and the model must still answer.
  - `caption`: a separate call writes a question-agnostic dense caption, and the subject answers from the text alone. This measures how much of the task can be carried by verbalization.
- **Refusals** are scored as their own outcome. Server-side refusal fallback is **off for subjects**, since it would silently substitute another model's answer, and on for judges.
- **Runs are resumable.** Attempts and verdicts append to JSONL and are keyed by (model, mode, condition, item, sample). The manifest records the settings (minus secrets), git commit and dataset manifest.

## 3. Verifier

The verifiers are applied in order of trust:

1. **Programmatic checkers** for each answer kind:
   - integer and number, with tolerances;
   - choice;
   - normalized text;
   - set, with F1 partial credit;
   - mapping, with per-key partial credit;
   - grid, with cell-accuracy partial credit;
   - custom family checkers: maze simulation (any optimal path accepted) and gear direction plus rpm.

   Answers that can't be parsed are labeled `unparseable`, separately from `incorrect`.
2. **Reference-guided equivalence judge**, used only for free-form answers in *external* sets, after the programmatic check fails ("75/2" vs "37.5 m²").
3. **Rubric judge**, for open-ended items (`chart_insight`, HSS):
   - one criterion per call, k samples with a majority vote;
   - reference facts in context;
   - critical criteria gate the pass;
   - per-criterion agreement is logged.
4. **Process judge** (optional): on failures, compares the model's visible reasoning with the generator's ground-truth facts and labels the first error as `perception`, `reasoning`, `instruction_following` or `incomplete`. This gives a perception-vs-reasoning breakdown.
5. **Judge audit** (`visionbench judge-audit`): runs the equivalence judge on a balanced sample of programmatically verified attempts and reports accuracy, Cohen's κ, sensitivity, specificity and false-accept rate.

6. **Reasoning critique** (`visionbench critique <run>`): grades *how* each agent reasoned, for correct and incorrect attempts alike.
   - **Key facts:** every family turns its generator's facts into a numbered list of decisive facts. Examples: the chain of gears from the driver to the target and each link's type; the voided receipt lines and category totals; the true time on the mirrored clock; the shortest path and its edges.
   - **What the critic does:** it sees those facts, the image and the agent's visible work (stated observations, a reasoning summary, tool calls). It returns:
     - every image claim the agent made, labeled correct, incorrect or unverifiable;
     - the status of each key fact: stated correctly, stated incorrectly, or not mentioned;
     - whether each step is valid, and whether the answer follows from the reasoning;
     - the first error, from a 12-type taxonomy (misread, omission, hallucination, attribute binding, spatial transform, tracking, counting, arithmetic, rule misapplication, instruction misread, premature commitment, gave up);
     - the agent's strategy, whether it self-verified, and its efficiency;
     - a written critique with strengths and weaknesses.
   - **Derived metrics (per agent and per family):**
     - **key-fact recall** and **key-fact misread rate** (perception);
     - **claim precision** (hallucination);
     - **valid-reasoning rate**;
     - **sound success**: correct *and* supported by valid reasoning;
     - **unsupported success**: correct answers that were guessed or reached through wrong steps;
     - **failure locus**: perception vs reasoning vs instruction vs no answer;
     - strategy mix, and accuracy with vs without self-verification.

     Head-to-head tables show *why* the loser failed on the items where two agents disagree.
   - **Agentic evidence grounding:** for agentic runs, crop boxes are compared deterministically with gold evidence boxes, giving the share of the evidence the agent actually inspected.
   - **Reliability of the critic:** `visionbench critique-audit` re-runs the critic on the same attempts and reports test-retest agreement for each judgment. Critique metrics should be read with that agreement in mind.
   - **Limitations:**
     - Claude models return a *summary* of their private reasoning, not the raw chain of thought, so the critique covers visible reasoning only.
     - When Claude critiques Claude there is a risk of self-preference. Grounding every judgment in generator facts reduces it. A judge from a different model family should be added when one is available.

## 4. Statistics

- **Unit of analysis.** The item: samples are averaged within an item first. Clusters are latent problems (re-renders) for procedural families and source items for external sets.
- **Uncertainty.**
  - Cluster-robust SE with a C/(C−1) correction.
  - 95% cluster bootstrap CIs (2,000 resamples).
  - The number of clusters is always reported.
- **Comparisons.**
  - Paired differences on shared items, with clustered SE, bootstrap CI, a sign-flip permutation p-value, Holm correction across pairs, the correlation between models' item scores, and the minimum detectable effect at 80% power.
  - Bootstrap rank intervals; overlapping intervals mean the ranking is unresolved.
- **Reliability.**
  - pass@k and pass^k use the unbiased combinatorial estimators.
  - Render consistency is the share of problems solved on every rendering.
- **Calibration.** ECE (10 bins), Brier score and overconfidence (mean confidence minus accuracy).
- **Item quality.**
  - A Rasch (1PL) fit gives item and family difficulty.
  - Suspect keys are items the strongest subject always fails while others succeed.
  - Blind-solvable items are those with blind accuracy ≥ 50%.
- **Power.** For about 3-point resolution between models, plan on ≥ 300 clusters per comparison. The report prints the MDE for every pair.

## 5. Recommended protocol for a publishable result

1. `visionbench build suite-test --split test --per-cell 20 --rerenders 1` (with a private secret set). That gives 15 families × 5 levels × 20 problems × 2 renders = 3,000 items and 1,500 clusters.
2. Import external sets at full size or a fixed stratified sample.
3. Run each model at its documented best setting, with `--samples 5 --conditions vision,blind,caption`, in both direct and agentic modes.
4. Run `visionbench verify <run> --process-judge` and `visionbench judge-audit <run>`.
5. Collect a human baseline on a 5–10% stratified subset (median and 5th percentile per family) and report human-normalized scores.
6. Publish the dev split, the generator versions and the report. Keep the test secret private, and rotate it every release.

## 6. Known limitations

- Synthetic renders differ from natural images. Check the correlation between procedural families and the external real-image sets before generalizing.
- `grid_rules` uniqueness is relative to the DSL. A human might infer a rule outside the DSL; the examples are chosen so that every DSL primitive is necessary.
- Judge-graded tracks (`chart_insight`, HSS) inherit judge error. Read their scores alongside the judge audit.
- Cost estimates use output-token priors until a pilot run measures them.
