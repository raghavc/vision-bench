# Where Frontier Multimodal Models Still Fail at Visual Reasoning (as of late Oct 2026)

Research date: 2026-10-30. Model generation in scope: GPT-6 Astra / GPT-6.1 Sol (OpenAI), Gemini 4 Argon / Gemini 3.x Flash (Google), Claude Opus 5.5 / Claude Fable 5.x (Anthropic), Muse Spark/Wonder (Meta), Qwen3.8, Kimi K3, DeepSeek V4.1, etc.

**Sourcing legend:** [OFFICIAL] = benchmark authors / lab / arcprize. [AGG] = third-party aggregator (BenchLM, llm-stats, DataLearner, benchmarklist, Developer's Digest, etc.), which can be wrong or mix settings. [PAPER] = arXiv preprint. [PRESS] = news write-up. [PRIOR] = figure from the original benchmark paper, cited from background knowledge and **not re-fetched in this session**. Verify before publishing.

**Important caveat:** I could not open any official 2026 frontier model card (GPT-6, Gemini 4 Argon, Claude Opus 5.5) and read its vision tables. The Gemini 4 Argon evaluation PDF downloaded but would not parse. Most frontier numbers below come from aggregators or press.

---

## 1. Static images: frontier vs human on hard visual-reasoning benchmarks; which are saturated; which skills fail

### Takeaway
Exam-style, knowledge-heavy and document/chart benchmarks (MMMU-Pro, OCR, chart/doc QA, visual math) are saturated or at human level, with frontier models at 87-98% and often above the human reference. "Perception-native" benchmarks still show big gaps: HSS (55% vs 93% human), BabyVision (best published about 50-69% vs 94% adult), MMSI-Bench (about 42-49% vs 97%), VisuLogic (35% vs 51%, with near-chance models), and ZeroBench (best reported about 41-53% depending on settings). The skills that still fail are fine-grained perception, counting, spatial and multi-view relations, abstract visual rule induction, and anything a model can't put into words.

### Cited Findings

**Humanity's Sixth Sense (HSS), Scale AI + Elorian, the newest and most relevant**
- Released 2026-10-07. 522 open-ended tasks (288 images + 234 videos). Human baseline is 93.1% (20 participants, same free-form conditions). The best model at launch was GPT-6-astra at 53.6%, and the median of 25 models was 30.9%. Metric: pass@1 averaged over 3 attempts. [PRESS] — [RuntimeWire](https://runtimewire.com/article/scale-ai-humanitys-sixth-sense-visual-reasoning-benchmark); Scale announcement [X post](https://x.com/ScaleAILabs/status/2107920376756117903)
- Leaderboard snapshot of 2026-10-28 [AGG]: Gemini-4-Argon (high) 55.4%, GPT-6-Astra (max) 53.6%, GPT-6.1-Sol (max) 46.6%, Claude-Opus-5.5 (xhigh) 44.6%, Muse Wonder 1.0 43.7%, Gemini-3.8-Flash 41.6%, Claude-Fable-5.1 40.8%, Qwen3.8-Max 33.0%, Kimi-K3 25.5%. Human 93.1%. No separate image and video scores are published. — [BenchLM HSS](https://benchlm.ai/benchmarks/scale-hss)
- Press reading: failures center on "identifying cues, spatial relationships, and social context," and extra reasoning time alone may not fix them. — [RuntimeWire](https://runtimewire.com/article/scale-ai-humanitys-sixth-sense-visual-reasoning-benchmark)
- Critique: Scale and Elorian control the tasks, the judging, and the human comparison behind the headline gap. — [Magica](https://magica.com/news/elorian-scale-ai-humanitys-sixth-sense-benchmark)

**BabyVision (UniPat AI; arXiv 2601.06521; v1 2026-01-10, v2 2026-07-07; ICML 2026)**
- 388 items, 4 families, 22 subtypes. Adult human average is 94.1%. At the paper's release, Gemini-3-Pro-Preview led with 49.7%, "lagging behind 6-year-old humans." Other scores: GPT-5.2 34.4%, Doubao-Seed-1.8 30.2%, Qwen3-VL-Plus 19.2%, Grok-4 16.2%, Claude-4.5-Opus 14.2%. — [arXiv abs](https://arxiv.org/abs/2601.06521); [The Decoder](https://the-decoder.com/even-the-best-ai-models-fail-at-visual-tasks-toddlers-handle-easily/)
- Authors' root cause: MLLMs "rely heavily on linguistic priors" to make up for weak visual understanding and lack "fundamental visual primitives." They propose BabyVision-Gen, a generation-based approach to visual reasoning. — [arXiv abs](https://arxiv.org/abs/2601.06521)
- Later numbers [AGG, self-reported]: Kimi K2.6 at 68.5% (llm-stats, June 2026). The "BabyVision w/ Python" (tool-assisted) variant lists Kimi K3 at 85.7% (BenchLM, July 2026). This suggests tool use and code closes much of the gap. — [llm-stats](https://llm-stats.com/benchmarks/babyvision); [BenchLM BabyVision Python](https://benchlm.ai/benchmarks/babyvisionpython)
- I found no GPT-6 Astra or Gemini 4 Argon BabyVision numbers.

**ZeroBench (arXiv 2502.09696)**
- Designed so that every model scored 0% at release (Feb 2025): 100 hard questions plus a subquestion set. [PRIOR] — [arXiv](https://arxiv.org/abs/2502.09696v1)
- 2026 reported scores vary widely by variant, tools, and pass@k [AGG]:
  - DataLearner (undated): GPT-5.6 Sol (max thinking + max tools) 53, Claude Opus 5 (max/tools) 52, DeepSeek-V4.1-Flash 49, Kimi K3 41.
  - Another aggregator (2026-09-10): Kimi K3 41%. Pass@5 board (2026-08-20): Kimi K3 23%.
  - BenchLM (2026-04-08): GPT-5.4 41%. llm-stats: Meta Muse Spark 33%. ZeroBench-Sub: Qwen3.5-122B 36.2%.
  - Sources: [DataLearner](https://www.datalearner.com/en/benchmarks/zerobench-main) (fetch failed; search snippet only); [anotherwrapper](https://anotherwrapper.com/tools/llm-pricing/evals/zerobench); [BenchLM](https://benchlm.ai/benchmarks/zeroBench); [llm-stats](https://llm-stats.com/benchmarks/zerobench)
- Inference: ZeroBench has gone from 0% to roughly 40-53% with tools in about 18 months, so it is no longer "zero." A Manifold market tracks when a lab will report 50% or more. — [Manifold](https://manifold.markets/Metastable/when-will-an-ai-lab-report-50-or-be)

**MMSI-Bench (multi-image spatial intelligence; ICLR 2026; arXiv 2505.23764)**
- 1,000 questions. Human performance is 97.2%. At publication the best proprietary model was GPT-5 at 41.9%. A later leaderboard row shows 49.2% (model name not legible; sampled about May 2026). — [arXiv](https://arxiv.org/html/2505.23764v2); [benchmarklist](https://benchmarklist.com/benchmarks/mmsi_bench/); [Moonlight review](https://www.themoonlight.io/review/mmsi-bench-a-benchmark-for-multi-image-spatial-intelligence)
- I found no 2026 frontier-generation (GPT-6/Gemini 4/Opus 5.5) MMSI-Bench scores.

**VisuLogic (arXiv 2504.15279)**
- Human baseline is 51.4%, and random is 25%. The project leaderboard (undated snapshot) shows Seed1.5-VL-Thinking at 35.0, dots.vlm1 at 32.2, and InternVL2.5-38B-RL at 31.1. The paper says most models score under 30%. — [VisuLogic project page](https://visulogic-benchmark.github.io/VisuLogic); [arXiv](https://arxiv.org/abs/2504.15279)
- I found no 2026 frontier closed-model VisuLogic results.

**TIR-Bench (agentic "thinking with images," arXiv 2511.01833)**
- 13 tasks, 22 MLLMs. The best non-agentic model at release was Gemini-2.5-Pro at 28.9%, and the best overall was o3 with a code interpreter at 46% (Nov 2025). o3-TU scored 16.4% on jigsaw puzzles. A third-party leaderboard (2026-05-28) lists Claude Sonnet 4.6 + PyVision at 46.7%. — [arXiv v2](https://arxiv.org/html/2511.01833v2); [AlphaXiv TIR-Bench](https://www.alphaxiv.org/benchmarks/university-of-southern-california/tir-bench); [benchmarklist](https://benchmarklist.com/benchmarks/tir_bench/) [AGG]

**Saturated or near-saturated areas**
- MMMU-Pro [AGG, Artificial Analysis]: Claude Opus 5.5 (Max) 88%, GPT-6 Astra (Max) 87%. — [Artificial Analysis MMMU-Pro](https://artificialanalysis.ai/evaluations/mmmu-pro)
- Older references: Gemini 3 Pro model card (Nov 2025) MMMU-Pro 81.0%, CharXiv-Reasoning 81.4%, ScreenSpot-Pro 72.7% (HN transcription of the model card). Claude Opus 4.6 system card (Feb 2026): CharXiv-Reasoning 68.5%, MMMU-Pro 73.9%. — [HN](https://news.ycombinator.com/item?id=45964748); [Anthropic Opus 4.6 system card](https://www-cdn.anthropic.com/0dd865075ad3132672ee0ab40b05a53f14cf5288.pdf); BenchLM Gemini 3.1 Pro: MMMU-Pro 83.9, CharXiv 80.2, ScreenSpot-Pro 84.4 [AGG] — [BenchLM compare](https://benchlm.ai/compare/gemini-3-1-pro-vs-gemini-4-argon)
- CharXiv human reasoning baseline is about 80.5% [PRIOR] — [arXiv 2406.18521](https://arxiv.org/abs/2406.18521). Frontier models are at or above it by late 2025 and 2026, per the figures above.
- "Hard Vision, Easy Vision" (MBZUAI, arXiv 2609.35718, 2026-09-28). This is the single best cross-capability map. It covers 6 systems (GPT-6 Astra, Fable 5, Kimi K3, Gemini 3.1 Pro, Qwen 3.8-Max, Muse Spark 1.3), 34 capabilities, and 55 benchmarks. [PAPER] — [arXiv HTML](https://arxiv.org/html/2609.35718v1)
  - **Easy (at or above the reference):**

    | Capability | GPT-6 Astra | Reference |
    |---|---|---|
    | OCR | 98.1 | 98.0 human |
    | Document/chart understanding | 89.5-92.8 (all six models) | 89.3 human |
    | 2D spatial reasoning | 96.0 | 95.8 human |
    | Visual math | 92.5 | 78.7 human |
    | 2D detection | 89.1 | 78.4 specialist |
    | 3D visual grounding | 73.8 | 46.5 reference |
    | Driving-scene reasoning | 80.8 | 73.1 reference |
    | Embodied understanding | 82.3 | 78.5 reference |
    | Video understanding | 74.3 | 71.5 reference |

  - **Hard (large gaps):**

    | Capability | Frontier result | Reference |
    |---|---|---|
    | Visual logical reasoning | Astra 81.1; other five models 57.6-67.5 | 87.0 |
    | Visual counting | Astra 82.7 | 94.0 (label ambiguous) |
    | Scientific reasoning (aggregate) | Astra 86.8 | 88.6 human |
    | 3D/multiview reasoning | Astra 89.6 | 94.1 human |
    | Depth (AbsRel, lower is better) | 0.63 | 0.44 |
    | 3D reconstruction error (lower is better) | 1.40 | 0.61 |
    | Image restoration (PSNR) | 17.2-17.7 | 30.7 |
    | Microscopy/pathology | 14.1-23.8 | 57.1 specialist, 82.0 human |
    | Remote-sensing grounding | 26.4 | 75.5 |
    | Medical 2D grounding | 61.5 | 81.8 |
    | Pose estimation | 75.2 | 83.6 |
    | Video segmentation | 84.5 | 91.0 |

  - The paper reports capability aggregates only, not per-benchmark scores for CharXiv, BLINK, VSI-Bench, ZeroBench, or BabyVision.
  - Cost: higher reasoning effort costs up to 13.22x baseline, sometimes for small gains, while tool use costs about 1.04-1.20x.
  - Quote: "domain reasoning does not necessarily imply domain perception."
- Chartography (Surge, third-party chart benchmark): Gemini 4 Argon 71.6%, GPT-6 Astra 71.0%, Claude Opus 5.5 66.3%, Claude Fable 5.1 46.2%. Released with the Argon launch on 2026-09-30. Hard, precise chart reading is **not** saturated, unlike CharXiv. — [emergent.sh summary of Google's Argon eval](https://emergent.sh/learn/gemini-4-argon-benchmarks) [AGG of OFFICIAL]

**Benchmark validity: "mirage" answering without the image**
- Mirage Probes (arXiv 2606.13870, June 2026): VLMs often answer image questions correctly with **no image**. Two regimes exist: textual-bias mirages and "spurious image" mirages, where the model builds false visual content in latent space. Text-side benchmark cleaning fixes only the first. — [arXiv abs](https://arxiv.org/abs/2606.13870)

**Not found or not verified for 2026 frontier models:** EMMA, VisualPuzzles, BLINK (human 95.7% [PRIOR], [arXiv 2404.12390](https://arxiv.org/abs/2404.12390)), O3-Bench, VisualToolBench, MathVision.

### Inferences
- **Saturated, so avoid building there:** MMMU/MMMU-Pro (87-88%), OCR, standard chart and document QA (CharXiv-style), visual math (MathVista/MathVision-style), 2D grounding and detection, generic VQA.
- **Real headroom, so focus here:**
  - Open-ended, perception-native tasks with human-easy, model-hard construction (HSS about 38-pt gap; BabyVision about 25-45 pt).
  - Multi-image spatial reasoning (MMSI about 48 pt).
  - Abstract visual rule induction (VisuLogic; other models trail GPT-6 Astra by 14-24 pt).
  - Precise metric geometry (depth, 3D reconstruction).
  - Counting.
  - Specialist-domain perception (pathology, remote sensing, medical grounding).
  - Precise chart extraction (Chartography about 70%).
- GPT-6 Astra stands out on visual logic (81 vs 58-68 for peers). Gaps are increasingly model-specific, which argues for per-vendor diagnostic evals rather than one leaderboard number.
- Tool and code access (BabyVision w/ Python at 85.7%; ZeroBench with "max tools" at about 53%) changes scores a lot. Evals should report with-tools and no-tools separately.
- Mirage results imply eval designers need image-ablation controls (blind baselines) and perturbation controls.

### Gaps
- No verified official 2026 lab model-card numbers for vision benchmarks (GPT-6 system card, Gemini 4 Argon eval PDF, Claude Opus 5.5 system card were not readable or not found).
- No HSS image-vs-video split or per-skill breakdown was found.
- No current frontier scores found for EMMA, VisualPuzzles, BLINK, O3-Bench, VisualToolBench, VisuLogic (closed models), or MMSI-Bench (2026 models).
- ZeroBench aggregator numbers conflict (41-53%). The official ZeroBench leaderboard was not checked.

---

## 2. Video: temporal reasoning, long video, event counting, object permanence, physics

### Takeaway
Standard video QA is at or above the reference: in "Hard Vision, Easy Vision," Astra scores 74.3 vs a 71.5 reference, and Gemini 4 Argon reports 91.7% on LVBench. Long-video scores are confounded by API frame limits. The real gaps are state-tracking tasks: streaming and periodic event counting, object identity through occlusion, event ordering, and dense temporal segmentation. Humans score about 93-100% on those, while the best models score about 37 overall on streaming counting.

### Cited Findings
- **LVBench (long video), vendor-run by Google, 2026-09-30:** Gemini 4 Argon 91.7%, GPT-6 Astra 87.5%, Claude Opus 5.5 83.7%, Claude Fable 5.1 79.7%. Caveat: Argon was fed 1 fps, while rivals got 800 (Astra), 600 (Opus 5.5), and 300 (Fable 5.1) frames because of API limits. The result partly measures how much video each API accepts. — [emergent.sh](https://emergent.sh/learn/gemini-4-argon-benchmarks); [DEV Community](https://dev.to/khasky/gemini-4-argon-beats-gpt-6-astra-and-claude-on-most-benchmarks-1nap) [AGG of vendor claims]
- **SVCBench (streaming video counting; arXiv 2603.12703; ECCV 2026; v3 2026-06-29):** 406 videos, 10,071 annotated events and state changes, 1,000 streaming QA pairs, 4,576 query points. It covers object counting (visible vs cumulative unique) and event counting (instantaneous vs cycles). Models show "significant deficiencies" and are especially poor at periodic event counting. — [arXiv abs](https://arxiv.org/abs/2603.12703)
  - Per a search summary of the paper: humans score 93-100 across subcategories, the best models reach about 37 overall, and E2-Periodic scores are under 4 vs 93.2 for humans. — [Pith](https://pith.science/paper/2603.12703) (summary; numbers not confirmed in the abstract)
- **TOC-Bench (temporal object consistency; arXiv 2605.09904; May 2026):** 2,323 human-verified QA pairs over 1,951 videos. A temporal-necessity filter removed 60.7% of candidate QAs because they were answerable without temporal info. Video-LLMs are weak on event counting, event ordering, identity-sensitive reasoning, and hallucination-aware verification. — [arXiv abs](https://arxiv.org/abs/2605.09904)
  - A search snippet adds that "longer reasoning traces don't close the gap." This was not confirmed in the abstract. — [arXiv HTML](https://arxiv.org/html/2605.09904)
- **Video segmentation (dense temporal prediction):** GPT-6 Astra 84.5 vs 91.0 reference. The causes are precise boundaries, small structures, nearby instances, and cross-frame consistency. — [Hard Vision, Easy Vision](https://arxiv.org/html/2609.35718v1)
- **Object permanence in video *generation* / world models:** WROP (arXiv 2609.28654) found models do OK on occlusion tracking but poorly on obstruction and collision. Another listing says none of ten video-generation models reliably remember objects after occlusion in dynamic scenes. — [arXiv 2609.28654](https://arxiv.org/html/2609.28654); [Pith citations](https://pith.science/citations/33a5087c-1633-4d52-b891-f34d10b9e7c8)
- **HSS includes 234 video tasks.** There is no separate video score, but the overall best is 55.4% vs 93.1% human. — [BenchLM HSS](https://benchlm.ai/benchmarks/scale-hss)

### Inferences
- Classic video QA (Video-MME / MVBench-style multiple choice) appears to be at or near the reference for frontier models. I found no current numbers, so treat this as likely but unverified.
- The headroom is in **stateful perception over time**: counting, identity persistence, ordering, and streaming updates. Here humans are near-perfect and models are far off.
- Long-video comparisons are confounded by frame budgets. A neutral eval company that controls frame input (same frames to every model) would add value.
- Many video QAs are answerable from single frames or text (TOC-Bench dropped 60.7%). Temporal-necessity filtering is a design must.

### Gaps
- No current (2026 frontier) numbers for Video-MME, MVBench, LongVideoBench, Perception Test, or the physics-video benchmarks (e.g., Physion/IntPhys-style).
- Best-model identity and exact scores on SVCBench and TOC-Bench were not confirmed from primary text.

---

## 3. 3D / spatial intelligence

### Takeaway
Single-image 2D spatial relations are solved (96 vs 95.8 human). Multi-view and multi-image spatial reasoning is still far off on hard sets: MMSI-Bench is about 42-49% vs 97% human. Metric 3D (depth, reconstruction) is about 1.4-2.3x worse than specialists. Long-horizon "spatial supersensing" (tracking world state across hours of video) fails consistently. The VSI-Bench line has validity problems: a no-temporal-structure baseline nearly solves one of its tasks.

### Cited Findings
- **VSI-Bench:** Cambrian-S-7B scores 67.5%, beating Gemini-2.5-Pro by more than 16 points. Spatial training data gave +30% absolute. Human level on the original VSI-Bench is 79.2 [PRIOR] — [Thinking in Space, arXiv 2412.14171](https://arxiv.org/abs/2412.14171). — [Cambrian-S arXiv 2511.04670](https://arxiv.org/pdf/2511.04670v1); [Moonlight review](https://www.themoonlight.io/review/cambrian-s-towards-spatial-supersensing-in-video)
- **VSI-SUPER (Cambrian-S):** Accuracy drops sharply on longer videos and fails completely beyond 60 minutes in streaming. — [Moonlight review](https://www.themoonlight.io/review/cambrian-s-towards-spatial-supersensing-in-video)
- **Critique, "Solving Spatial Supersensing Without Spatial Supersensing" (arXiv 2511.16655, Nov 2025):**
  - A bag-of-words SigLIP baseline ("NoSense") that discards temporal structure gets about 95% on VSR even on 4-hour videos.
  - A perturbation collapses Cambrian-S mean relative accuracy from 42% to 0%.
  - [arXiv](https://arxiv.org/abs/2511.16655v1)
- **VSI-Super-Wild (arXiv 2607.13681, 2026-07-15, ECCV 2026; authors include Marco Pavone and Yiming Li):** 6,980 human-verified QA pairs, 442 real videos (some over 4h), 8 scene categories, 13 MLLMs tested. Models "consistently fail at tasks that require coherent world-state tracking over time." Performance degrades with world-state complexity and horizon. Failure modes are spatial collapse, semantic shortcuts, insufficient update, and instance confusion. Models lack mechanisms to bind objects, agents, and environment into a unified spatial world model. — [arXiv abs](https://arxiv.org/abs/2607.13681)
- **MMSI-Bench:** 97.2% human vs 41.9% (GPT-5 at publication) and 49.2% (later leaderboard row). — [arXiv](https://arxiv.org/html/2505.23764v2); [benchmarklist](https://benchmarklist.com/benchmarks/mmsi_bench/)
- **Hard Vision, Easy Vision, frontier aggregates (2026-09-28):** 2D spatial 96.0 vs 95.8 human; 3D/multiview reasoning 89.6 vs 94.1 human; 3D grounding 73.8 vs 46.5 (beats reference); depth AbsRel 0.63 vs 0.44; 3D reconstruction error 1.40 vs 0.61. The stated cause is sensitivity to metric scale, local surface geometry, camera motion, and cross-view alignment. — [arXiv](https://arxiv.org/html/2609.35718v1)

### Inferences
- The discriminative 3D frontier is **metric** (exact distances, depth, camera pose and motion) and **long-horizon world-state tracking**. Qualitative spatial relations are mostly solved.
- Existing spatial benchmarks have shortcut problems (NoSense, semantic shortcuts). Shortcut-proof, procedurally controlled spatial evals are a real opening.
- Disagreement: Hard Vision, Easy Vision's multiview aggregate (89.6 vs 94.1) looks much closer to human than MMSI-Bench (about 49 vs 97). This is likely because aggregates average in easier benchmarks. A fair reading is that hard multi-image spatial items remain unsolved.

### Gaps
- No 2026 frontier scores for VSI-Bench, SpatialEval, mental-rotation-specific suites, or ERQA.
- I found no dedicated 2026 world-model evaluation with frontier MLLM numbers, beyond the WROP and video-generation permanence work.

---

## 4. Embodied / simulation / games / computer use

### Takeaway
- **Computer use (OSWorld-Verified)** is now *above* human (about 83-86% vs 72.4%) and is effectively saturated. ScreenSpot-Pro is at 92.7%.
- **Interactive novel-environment reasoning (ARC-AGI-3)** jumped from under 1% at launch (March 2026) to 62.7% (GPT-6 Astra Max, Oct 2026). Humans solve 100% of environments, so a gap remains, but it is closing fast.
- **Long-horizon games** remain hard. NetHack's best is 13.2% progress. Pokémon is now beatable but slow: about 50-325 h vs about 26 h for humans.
- **Robotics VLA:** LIBERO is saturated (98%+) and collapses under perturbation. Real-world generalization evals are thin.

### Cited Findings

**ARC-AGI-3 (interactive, game-like environments)**
- At launch on 2026-03-25 (official API, no custom tooling): Gemini 3.1 Pro 0.37%, GPT-5.4 0.26%, Claude Opus 4.6 0.25%, Grok 4.20 0%. Human testers solved 100% of environments on first try. The best preview agent got about 12.58% but scored 0.25% on the full benchmark. — [eWeek](https://www.eweek.com/news/ai-test-reality-check-neuron/); [jerrycards](https://www.jerrycards.com/news/arc-agi-3-frontier-models-below-1-percent) [PRESS]
- 2026-07-24: ARC Prize verified Claude Opus 5 (High) at 30.2%. — [explainx](https://explainx.ai/blog/arc-agi-3-opus-5-leaderboard-july-2026) [AGG]
- Snapshot of 2026-10-28, Standard harness [AGG mirror of arcprize semi-private]:
  - GPT-6 Astra: Max 62.7%, XHigh 59.3%, High 54.8%, Medium 38.6%, no reasoning 35.2%.
  - GPT-6.1 Sol Max: 52.7%.
  - Gemini 4 Argon: Low 49.8%, Medium 46.3%.
  - Claude Opus 5 High: 30.2%.
  - [BenchLM ARC-AGI-3](https://benchlm.ai/benchmarks/arcagi3)
- A separate tracker says GPT-6 Astra led at 62.71% as of 2026-10-23. llm-stats shows a conflicting 0.999, which is unreconciled and likely a different metric or error. — [llmlearner](https://llmlearner.com/rankings/arc-agi-3); [llm-stats](https://llm-stats.com/benchmarks/arc-agi-3)
- The official leaderboard page would not render scores in fetch. — [arcprize.org/leaderboard](https://arcprize.org/leaderboard)

**Computer use**
- **OSWorld-Verified:** Human baseline 72.36%. GPT-5.4 self-reported 75.0% in March 2026, the first reported crossing of human level. As of 2026-10-17, the board shows claude-fable-5[1m] 86.0% and claude-opus-5 83.4% (cites the xlang official leaderboard). — [Developer's Digest](https://www.developersdigest.tech/benchmarks/osworld) [AGG]; [presenc.ai](https://presenc.ai/research/osworld-computer-use-leaderboard-2026) [AGG]
  - An unverified claim of 90.19% for "Intelligence-Indeed Agent" exists. — [presenc.ai](https://presenc.ai/research/osworld-computer-use-leaderboard-2026)
- **OSWorld 2.0 (harder successor):** The best official entry fully completes about 44%. Google reports an offline subset at Gemini 4 Argon 69.2% vs GPT-6 Astra 72.6% (vendor-reported, 2026-09-30). — [presenc.ai](https://presenc.ai/research/osworld-computer-use-leaderboard-2026); [emergent.sh](https://emergent.sh/learn/gemini-4-argon-benchmarks)
- **Agent's Last Exam (computer use):** Gemini 4 Argon 39.5%, Claude Opus 5.5 38.2%, GPT-6 Astra 34.2%. This is still low. — [emergent.sh](https://emergent.sh/learn/gemini-4-argon-benchmarks) [AGG of vendor]
- **ScreenSpot-Pro (Oct 2026):** GPT-6 Astra 92.7%, Claude Opus 4.8 87.9%, GPT-5.4 85.4%. — [BenchLM ScreenSpot-Pro](https://benchlm.ai/benchmarks/screenspot-pro) [AGG]

**Games**
- **BALROG** (read 2026-10-09) [AGG summary of balrogai.com]: GPT-6-Astra-Max 68.3% progress (2026-09-18), Claude-Opus-5-Max 63.4%, GPT-5.6-Sol-Max 60.0%, Gemini-3-Pro 58.1%.
  - Per environment: BabyAI and BabaIsAI 100%; Crafter 76.8; TextWorld 75.7; MiniHack 65.0; **NetHack best 13.2%**.
  - The VLM (vision-input) track is stale: the best entry is Gemini 2.5 Pro at 35.7.
  - [creategame.ai](https://creategame.ai/blog/can-ai-play-video-games/); [balrogai.com](https://balrogai.com); [BALROG paper](https://arxiv.org/abs/2411.13543)
- **Pokémon:**
  - Claude Opus 4.7 beat Pokémon Red on 2026-05-16 after about 325 h (lightly harnessed). Earlier Claude models stalled (Opus 4.5 needed about 230k steps for 8 badges, then got stuck on Victory Road). Claude Fable 5 did FireRed (announced 2026-06-09; about 50 h, Bradshaw's outside estimate).
  - Gemini 2.5 Pro beat Blue in 2025 (813 h). GPT-5 beat Red in 6,470 steps (about 7 days).
  - Human reference is about 26 h.
  - Sources: [LessWrong, "A Year Late, Claude Finally Beats Pokémon"](https://www.lesswrong.com/posts/sehJYg5Yny9fvpbpt/a-year-late-claude-finally-beats-pokemon); [creategame.ai](https://creategame.ai/blog/can-ai-play-video-games/); [TechRadar](https://www.techradar.com/ai-platforms-assistants/chatgpt/gpt-5-just-completed-pokemon-red-in-a-new-world-record-time-claude-gemini-and-chatgpt-o3-arent-even-close)
- **VideoGameBench (arXiv 2505.18134; spring 2025 results):** Best full-benchmark score was 0.48% (Claude 3.7 Sonnet, Gemini 2.5 Pro). Doom II and Link's Awakening were 0% for all models. No 2026 results were found. — [arXiv v3](https://arxiv.org/html/2505.18134v3)
- **SIMA 2 (DeepMind, 2025-11-13):** Held-out MineDojo/ASKA task completion of 45-75% (SIMA 1: 15-30%). About 62% on the main suite vs about 70% human reference (secondary, unverified). — [DeepMind blog](https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/); [The Decoder](https://the-decoder.com/deepminds-latest-ai-agent-learns-by-exploring-unfamiliar-games-and-ai-built-worlds/); [MarkTechPost](https://www.marktechpost.com/2025/11/16/google-deepmind-introduces-sima-2-a-gemini-powered-generalist-agent-for-complex-3d-virtual-worlds/)

**Robotics VLA**
- **LIBERO:** A June 2026 survey of 582 VLA papers says models cluster above 98%, which reduces its discriminative power. The survey also finds metric reporting is one-dimensional (task success) and that current practice fails its four conditions for a valid progress claim. — [preprints.org survey](https://www.preprints.org/manuscript/202606.0425)
- **LIBERO-PRO:** Models scoring over 90% on standard LIBERO collapse to 0.0% under generalized perturbations, which points to memorization of action sequences and layouts. — [aitopics/arXiv LIBERO-PRO](https://aitopics.org/doc/arxivorg:49434AE9)
- **LIBERO-X (RSS 2026):** Adds progressive difficulty for spatial generalization, unseen and confounding objects, and instruction variation. — [arXiv 2602.06556](https://arxiv.org/abs/2602.06556)

### Inferences
- **Saturated:** OSWorld-Verified, ScreenSpot-Pro (above 90%), LIBERO, BALROG's easy environments (BabyAI, BabaIsAI).
- **Headroom:**
  - ARC-AGI-3: about 37 pts to human 100%, but moving fast (0.4% to 62.7% in 7 months).
  - NetHack (13%).
  - Agent's Last Exam (about 40%).
  - OSWorld 2.0 (about 44% fully complete).
  - Efficiency on long-horizon games: Pokémon takes about 2-12x human time.
  - VLA robustness under perturbation.
- Reasoning effort matters enormously on ARC-AGI-3 (Astra 35.2% with no reasoning to 62.7% at max). Evals should report compute and cost-normalized scores.
- The BALROG VLM track hasn't been updated with 2026 models. Vision-in-the-loop game evals for the frontier generation are under-measured.

### Gaps
- No current numbers found for VisualAgentBench, EmbodiedBench, SimplerEnv, RoboArena, or Minecraft benchmarks with 2026 frontier models.
- The ARC-AGI-3 human baseline is stated as 100% environment completion. A per-task human efficiency (action-count) baseline was not retrieved.
- OSWorld-Verified official xlang leaderboard not directly fetched.

---

## 5. Root causes: what analysts and labs say

### Takeaway
There is no consensus single cause. The literature points to four things, and the dominant bottleneck varies by model and benchmark pair:
1. **Language-prior substitution / verbalization bottleneck.** Models solve what they can describe in words and fall back on text priors otherwise (BabyVision, Mirage).
2. **Perceptual bandwidth and resolution limits** from visual-token budgets.
3. **No persistent visual state / world model** for tracking over time and views (VSI-Super-Wild, SVCBench, TOC-Bench).
4. **Lack of metric and dense-prediction precision** (depth, segmentation, pathology).

### Cited Findings
- **BabyVision:** MLLMs "rely heavily on linguistic priors" to compensate for weak visual understanding and lack fundamental visual primitives. The title frames this as "visual reasoning beyond language." — [arXiv 2601.06521](https://arxiv.org/abs/2601.06521)
- **Mirage Probes:**
  - Models answer correctly with no image.
  - "Spurious image" mirages build false visual content in latent space, so text-only benchmark cleaning is insufficient.
  - Mirage behavior is linearly decodable from activations.
  - Faithful grounding needs "interventions at the representational level."
  - [arXiv 2606.13870](https://arxiv.org/abs/2606.13870)
- **Look Light, Think Heavy (arXiv 2606.22565):** In multimodal chain of thought, verbal reflection rises and peaks midway while visual reflection diminishes. Models increasingly neglect the image as they reason in text. There is a 20% performance drop on visual probes vs textual ones. — [arXiv PDF](https://arxiv.org/pdf/2606.22565)
- **Perceptual Bandwidth Bottleneck (arXiv 2605.01345):** Limited visual-token budget is the core constraint. The paper proposes active, sequential evidence acquisition, i.e., controlled allocation of perception rather than only stronger reasoning. — [arXiv HTML](https://arxiv.org/html/2605.01345v2)
- **Downscaling Intelligence (arXiv 2511.17487):** Shrinking the LLM degrades both perception and reasoning. The paper identifies a "perception bottleneck" where visual details present in features are not extracted. — [arXiv HTML](https://arxiv.org/html/2511.17487)
- **ICLR 2026 cross-modal paper:** Unimodal fact recognition is not the bottleneck; combining modalities is. — [ICLR 2026 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/13192e17060e78eee6c5425740989181-Paper-Conference.pdf)
- **VSI-Super-Wild:** Models lack mechanisms to bind objects, agents, and environments into a unified spatial world model. Failure modes are spatial collapse, semantic shortcuts, insufficient update, and instance confusion. — [arXiv 2607.13681](https://arxiv.org/abs/2607.13681)
- **Hard Vision, Easy Vision:**
  - "domain reasoning does not necessarily imply domain perception."
  - Precision failures come from metric scale, surface geometry, camera motion, and cross-view alignment.
  - Restoration hallucinates content (four bowling pins became five).
  - Extra reasoning compute (up to 13x) yields small gains on some perception tasks.
  - [arXiv](https://arxiv.org/html/2609.35718v1)
- **HSS (press summary):** Extra reasoning time alone may not fix failures to identify cues, spatial relationships, and social context. — [RuntimeWire](https://runtimewire.com/article/scale-ai-humanitys-sixth-sense-visual-reasoning-benchmark)
- **ARC-AGI-3 (Chollet, per press):** Models only do well when humans build elaborate scaffolding around them. Critics dispute the scoring methodology. — [eWeek](https://www.eweek.com/news/ai-test-reality-check-neuron/)
- **LIBERO-PRO:** VLA success reflects rote memorization of action sequences and layouts. — [aitopics](https://aitopics.org/doc/arxivorg:49434AE9)

### Inferences
- **Eval-company implication 1:** The most defensible eval designs are (a) non-verbalizable perception tasks, (b) blind and perturbation controls to detect mirages and shortcuts, (c) stateful multi-frame and multi-view tracking, and (d) metric-precise answers with continuous scoring. Each maps to a documented root cause and to measured large gaps.
- **Eval-company implication 2:** Because gaps are model- and benchmark-specific (PARA-style attribution), a diagnostic product that separates perception errors from reasoning errors has more value than another aggregate leaderboard.
- Benchmark half-lives are short: ZeroBench went from 0 to about 50% in about 18 months, and ARC-AGI-3 from 0.4% to 63% in 7 months. Procedurally generated or refreshable private test sets are strategically important.

### Gaps
- No frontier-lab (OpenAI, Google DeepMind, Anthropic) public statement on vision root causes was retrieved this session.
- Tokenization and resolution claims rest on academic papers, not lab disclosures.
- The PARA perception-vs-reasoning attribution paper was referenced in search results but not fetched, so there is no URL or quantitative result for it.
