# Sell verifiable visual worlds, not leaderboards

Frontier labs pay for vision work, but the money goes to **verifiable training environments, not benchmarks**. Buyers ask for RL environments 10–20x more often than for evals. Contracts run six to seven figures per quarter. Tasks are priced at $200–2,000 each, and exclusivity carries a 4–5x premium ([Epoch AI](https://epoch.ai/gradient-updates/state-of-rl-envs)). Anthropic currently has a live opening on its Vision team whose job is to write "a vendor-facing specification for a new family of visual RL training tasks" and to build QA that catches reward hacking ([Anthropic Greenhouse](https://job-boards.greenhouse.io/anthropic/jobs/5074217008)). That is the clearest sign that a buyer exists. Headroom is real, but it has moved. Exam-style, OCR, chart and document vision is saturated at or above human level. The large gaps are in perception-native, stateful and metric tasks:

- Humanity's Sixth Sense: 55% vs 93% for humans.
- MMSI-Bench (multi-image spatial): about 49% vs 97%.
- Streaming event counting: about 37 overall, and under 4 on periodic events vs 93 for humans.
- Long-horizon world-state tracking and long-horizon games.

No funded startup owns "visual reasoning as an RL-environment product". The crowded part of the environment market is text, tools and GUI clones. Expert-data incumbents (Mercor, Surge, Scale, Snorkel) could still enter quickly. VisionBench's first live run shows the right shape for a wedge. Models score near 100% on fully specified short-chain families and 27–65% on search and tracking families. Programmatic keys, blind controls and cross-vendor audits catch defects that hand-written benchmarks keep missing. The run cost $13.40 for 1,771 attempts. The recommended path is to turn the generator library into difficulty-calibrated, state-verified RL task families. Prove training lift on an open model, land one or two pilots, then climb a 2D → 2.5D → 3D ladder. Move into Unreal Engine 5 only once Epic's AI-training restriction is formally lifted.

## Good vision tasks fail without the image and survive an audit

Most benchmark failures are not about difficulty. They are about **wrong answer keys and items that don't measure vision at all**. Audits of flagship benchmarks keep finding defect rates in the double digits:

- **SWE-bench Verified.** OpenAI retired it in February 2026 after finding that **at least 59.4% of audited problems in a frequently-failed subset had tests that rejected correct solutions** ([OpenAI](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)).
- **Humanity's Last Exam.** FutureHouse experts found about 29–30% of audited chemistry and biology answers contradicted by peer-reviewed literature ([FutureHouse](https://www.futurehouse.org/research/hle-exam)).
- **MMLU.** About 6.5% of questions are erroneous, rising to 57% in virology ([arXiv 2406.04127](https://arxiv.org/abs/2406.04127v1)).
- **POPE (vision).** Re-annotation found label errors in 9.3% of positive questions, and the corrections reshuffled model rankings ([RePOPE](https://www.emergentmind.com/topics/repope)).

Epoch AI's Benchmark Reviews, launched in September 2026, turn this into a public bar. A benchmark is rated **"Flawed" if score-affecting defects reach 20% of total score**. To be reviewable, the tasks, the scoring logic and the harness settings must all be inspectable. Of the first 15 benchmark versions reviewed, 9 were rated Flawed ([Epoch methodology](https://epoch.ai/data/benchmark-reviews-documentation/methodology); [RuntimeWire](https://runtimewire.com/article/epoch-ai-benchmark-reviews-nine-flawed)). There is a trap in the design method most vendors use. Defects concentrate in the items models fail (OpenAI's 59.4% figure comes from exactly that subset). So "keep only what models get wrong" adversarial filtering **selects for broken keys** unless every model-failed item gets extra human verification.

Vision adds a second problem: **many "visual" questions can be answered without looking**. A NeurIPS 2026 audit found that 20.2% of MMStar questions, a set curated specifically to need the image, show zero accuracy gain from the image ([Degtiarev et al.](https://neurips.cc/virtual/2026/178389)). Video is worse. An audit of 14 video-QA benchmarks found roughly 54–55% of samples solvable from a single center frame, shuffled frames or no visual input ([Video-Oasis](https://www.alphaxiv.org/abs/2603.29616)). TOC-Bench threw out 60.7% of its candidate questions because they did not need temporal information ([arXiv 2605.09904](https://arxiv.org/abs/2605.09904)).

Zero-shot blind tests are necessary but not sufficient. NYU's Test-set Stress-Test trains a model on only the *non-visual* features of a test set to expose shortcuts that can be learned. It pruned 30.7% of VSI-Bench. Fine-tuning on those features raised a 7B model's blind VSI-Bench accuracy from 25.9% to 44.7% ([arXiv 2511.04655](https://arxiv.org/abs/2511.04655v1)). "Mirage" research shows the leak runs deeper than question text. Models can build spurious visual content in latent space, so cleaning text alone does not fix it ([arXiv 2606.13870](https://arxiv.org/abs/2606.13870)).

The third design axis is the **perception–reasoning confound**. In Scale's Humanity's Sixth Sense (HSS), 94% of 8,573 model failures came from perception or latent inference, and only 5% from faulty logic ([Scale Labs](https://labs.scale.com/blog/introducing-humanitys-sixth-sense)). ARC-AGI-3 goes the other way. It deliberately makes its environments turn-based, with a tiny action space, so that difficulty sits in the reasoning, not the controls ([ARC-AGI-3 Technical Report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)). A benchmark that doesn't tag which bottleneck each item targets produces numbers nobody can act on. BabyVision gives a specific reason this matters for training. RLVR gains concentrated in the subtypes that can be put into words, with negligible or negative gains on visual tracking ([hyper.ai summary](https://hyper.ai/en/papers/2601.06521)). Text-answer verifiable rewards may not train non-verbal visual skill at all.

### What the best task pipelines actually do

The strongest published pipelines share a common set of practices.

**Humanity's Sixth Sense (Scale + Elorian).** Annotators wrote a question, a reference answer and a rubric of atomic criteria. Every task needed unanimous human agreement and was rejected if a high-resolution crop made the answer obvious. Three independent review rounds kept **522 of 3,466 tasks (15.1%)**. Answers are free-form, graded criterion by criterion, with all criteria required to pass. A judge cross-check using three vendors' models gave identical rankings (ρ=1.0). Withholding the media dropped the leading model to 6.6% ([Scale Labs](https://labs.scale.com/blog/introducing-humanitys-sixth-sense)). The critique still bites: Scale chose the tasks, controlled the judging and set a human baseline of n=20, with no outside replication ([Magica](https://magica.com/news/elorian-scale-ai-humanitys-sixth-sense-benchmark)).

**ARC-AGI-3.** This is the most rigorous public spec for interactive environments:

- a concept-doc review before anything is built;
- 50,000-step random-agent checks and 1M-step fuzzing sweeps;
- a compression-based novelty test between environments;
- at least six levels per environment, starting with a tutorial;
- 10 untrained members of the public per environment, with inclusion only if at least two solve every level;
- scoring as (human actions / AI actions)², against the upper-median best human.

In total, 486 participants made 2,893 attempts on 414 candidate environments ([ARC-AGI-3 Technical Report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)). ARC also states the lesson that matters most for a procedural vendor. When public and private sets come from the same distribution, labs can generate millions of synthetic in-domain tasks and train on them. **Private sets must therefore be out-of-distribution from public demonstration data.**

**Mercor APEX.** For contrast on cost, expert rubric tasks ran about **$2,500 each** ($500k+ for the first 200 cases) ([Implicator](https://www.implicator.ai/mercors-apex-says-gpt-5-scores-64-on-real-work-still-not-production-ready/)).

**Statistics.** Questions that share an image or video must use clustered standard errors and paired-difference tests ([Miller, arXiv 2411.00640](https://arxiv.org/abs/2411.00640)). Epoch recommends at least five runs per model.

### How to specify a vision task

These practices converge on a task specification that VisionBench should adopt as its contract with buyers. Each element below maps to a documented failure mode.

| Spec field | Requirement | Failure it prevents |
|---|---|---|
| Construct and bottleneck tag | One named skill; tagged perception / spatial / temporal / reasoning / tool-use | Uninterpretable scores (HSS 94% perception vs ARC's reasoning focus) |
| Ground truth | Computed by a generator, solver or simulator state; atomic rubric only where free-form is unavoidable | Key errors of 20–60% in the hardest tail |
| Uniqueness / ambiguity proof | Alternative readings checked; decision margins enforced | Multiple defensible answers |
| Mandatory controls | Blind, caption-only, single-frame/shuffled-frame (video), crop-obvious, plus a set-level learned-shortcut test | Language-prior and temporal shortcuts (MMStar, Video-Oasis, TsT) |
| Answer format | Free-form typed answers or ≥10 options; minimal pairs for perception and physics | 25% MCQ floors that compress discrimination |
| Human baseline | Solve rate and time/actions, with documented demographics | Unknown headroom; narrow baselines (BabyVision, HSS n=20) |
| Difficulty knobs | Generator parameters, not filtering against particular models | Bias against the filtering vendor's models |
| Splits | Public demo, semi-private, private out-of-distribution; rotatable | Distributional training (ARC lesson) |
| Harness metadata | Effort, token/time limits, tools, image resolution, frame budget | "Not enough info" under Epoch's rubric |
| Statistics | Clustered SEs, k samples, pass^k, paired tests | False rankings from small n |
| Audit trail | Reviewer IDs, defect rate per batch, version bumps | Silent drift; self-certification |

VisionBench's first run already implements most of this. The results show both why it works and where it still leaks ([VisionBench FINDINGS](../FINDINGS.md)):

- **Blind control.** Withholding the image collapsed every model to **6.7–9.4%**, a vision-minus-blind gap of +60 to +80 points. The same control flagged `block_stability` as 30–60% solvable blind, because about 45% of towers are stable and the answer space is tiny.
- **Cross-vendor consensus audit.** It looks for identical wrong answers from different vendors. On its first pass it surfaced five genuine item or layout defects, among them an ambiguous receipt column header, a rotated "P" that really is a "d", and merged belt segments. It also surfaced two cases where the key was right and models shared a misconception: a centre-of-mass check against the wrong width, and mirror-image chirality on cube folding.
- **Rubric audit.** The `chart_insight` family scored about 10% for every model *with 100% judge agreement*. The judge was consistent but the rubric was unfair. This is the same point the HSS critique makes: judge agreement measures consistency, not validity.

## Models ace specified short chains and break on search, tracking and state

The capability map has split cleanly. MBZUAI's "Hard Vision, Easy Vision" covers 6 frontier systems, 34 capabilities and 55 benchmarks. It finds GPT-6 Astra at or above the reference on OCR (98.1 vs 98.0), document and chart understanding, 2D spatial reasoning (96.0 vs 95.8), visual math (92.5 vs 78.7) and even 3D grounding (73.8 vs 46.5) ([arXiv 2609.35718](https://arxiv.org/html/2609.35718v1)). MMMU-Pro sits at 87–88% ([Artificial Analysis](https://artificialanalysis.ai/evaluations/mmmu-pro)), and OSWorld-Verified computer use is **above** the 72.4% human baseline at 83–86% ([Developer's Digest](https://www.developersdigest.tech/benchmarks/osworld)). Building there means building on saturated ground.

The gaps that remain are concentrated in four areas:

- **Perception that cannot be verbalized.**
- **State tracked over time or across views.**
- **Metric precision.**
- **Long-horizon interaction.**

| Setting | Benchmark | Frontier result | Human / reference | Source |
|---|---|---|---|---|
| Image, perception-native | Humanity's Sixth Sense (522 tasks) | 55.4% (Gemini 4 Argon); median of 25 models 30.9% | 93.1% | [BenchLM](https://benchlm.ai/benchmarks/scale-hss); [RuntimeWire](https://runtimewire.com/article/scale-ai-humanitys-sixth-sense-visual-reasoning-benchmark) |
| Image, pre-linguistic | BabyVision | 49.7% at release; 68.5% later (self-reported); 85.7% with Python | 94.1% adult | [arXiv 2601.06521](https://arxiv.org/abs/2601.06521); [llm-stats](https://llm-stats.com/benchmarks/babyvision); [BenchLM](https://benchlm.ai/benchmarks/babyvisionpython) |
| Image, abstract rules | VisuLogic | 35.0 (best listed) | 51.4 (chance 25) | [VisuLogic](https://visulogic-benchmark.github.io/VisuLogic) |
| Image, tool use | VisualToolBench | 18.7% | — | [Scale](https://scale.com/blog/visualtoolbench) |
| Image, precise charts | Chartography | 71.6% | — | [emergent.sh](https://emergent.sh/learn/gemini-4-argon-benchmarks) |
| Multi-image spatial | MMSI-Bench | 41.9–49.2% | 97.2% | [arXiv 2505.23764](https://arxiv.org/html/2505.23764v2); [benchmarklist](https://benchmarklist.com/benchmarks/mmsi_bench/) |
| Metric 3D | Depth AbsRel / reconstruction error | 0.63 / 1.40 | 0.44 / 0.61 specialist | [arXiv 2609.35718](https://arxiv.org/html/2609.35718v1) |
| Video, streaming counting | SVCBench | ~37 overall; periodic events <4 | 93–100; 93.2 | [Pith summary](https://pith.science/paper/2603.12703) |
| Video, long-horizon spatial | VSI-Super-Wild | Consistent failure on world-state tracking | — | [arXiv 2607.13681](https://arxiv.org/abs/2607.13681) |
| Interactive novel games | ARC-AGI-3 | <1% at launch (Mar 2026) → 62.7% (Oct 2026) | 100% of environments | [ARC report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf); [BenchLM](https://benchlm.ai/benchmarks/arcagi3) |
| Long-horizon games | NetHack (BALROG) | 13.2% progress | — | [creategame.ai](https://creategame.ai/blog/can-ai-play-video-games/) |
| Real-time games | VideoGameBench | 0.48% (2025 models) | — | [arXiv 2505.18134](https://arxiv.org/abs/2505.18134) |
| 3D game worlds | SIMA 2 | ~62% task completion | ~70% | [MarkTechPost](https://www.marktechpost.com/2025/11/16/google-deepmind-introduces-sima-2-a-gemini-powered-generalist-agent-for-complex-3d-virtual-worlds/) |
| Robot policies | LIBERO → LIBERO-PRO | >90% → 0.0% under perturbation | — | [LIBERO-PRO](https://aitopics.org/doc/arxivorg:49434AE9) |

Several caveats temper this table. Most 2026 frontier numbers come from aggregators, not model cards. Some headline video results partly measure API frame limits: Gemini 4 Argon's 91.7% on LVBench was run at 1 fps, while rivals got 300–800 frames ([emergent.sh](https://emergent.sh/learn/gemini-4-argon-benchmarks)). Some spatial benchmarks have their own shortcut problems. A bag-of-words baseline that ignores temporal structure scores about 95% on one VSI-Super task, even on 4-hour videos ([arXiv 2511.16655](https://arxiv.org/abs/2511.16655v1)).

Benchmarks also decay fast. ZeroBench went from 0% to roughly 41–53% with tools in about 18 months ([BenchLM](https://benchlm.ai/benchmarks/zeroBench)). ARC-AGI-3 went from 0.4% to 62.7% in seven months, and reasoning effort alone moves GPT-6 Astra from 35.2% to 62.7% ([BenchLM](https://benchlm.ai/benchmarks/arcagi3)). Static sets are a depreciating asset. Results that don't separate tool access and compute budgets are hard to interpret.

Root-cause research converges on four explanations. Models substitute language priors for weak visual primitives ([BabyVision](https://arxiv.org/abs/2601.06521)). Visual attention fades as chains of thought lengthen ([Look Light, Think Heavy](https://arxiv.org/pdf/2606.22565)). Visual-token budgets cap perceptual bandwidth ([arXiv 2605.01345](https://arxiv.org/html/2605.01345v2)). And models have no mechanism to bind objects and agents into a persistent spatial world model ([VSI-Super-Wild](https://arxiv.org/abs/2607.13681)). Extra reasoning compute, up to 13x baseline cost, buys little on many perception tasks ([arXiv 2609.35718](https://arxiv.org/html/2609.35718v1)).

### What VisionBench's first run adds

VisionBench's showcase run reproduces this split under tight experimental control, and adds a mechanism. It covered 150 procedural items across 15 families, plus 25-item samples of VisuLogic and BabyVision. Five models from four vendors were run at medium reasoning effort ([VisionBench FINDINGS](../FINDINGS.md)).

**Short, fully specified chains are solved.** Averaged across the five models:

- `chart_reasoning`, `graph_paths` and `receipt_reasoning`: **100%**;
- `lattice_geometry`: 98%;
- `gear_train`: 97%;
- `counting`: 96%, though this family is partly guessable.

**Search and tracking families are not.** `visual_search` averages **27%**, `maze_path` 44%, `grid_rules` 57% and `wire_tracing` 65%. These per-family means come from the run's `results.json`. Within families the spread is wide: `wire_tracing` runs 90–100% for the top two models and 40–50% for the rest, and Qwen3.8-Flash scores 5% on grid rule induction.

**The headline ranking is statistically clean.** Gemini-4-Argon (88.9%) and GPT-6.1-Terra (85.9%) are tied (p=0.54). Both beat the other three at Holm-corrected significance.

**Small per-step errors compound.** VisionBench's compounding analysis (FINDINGS.md §11) resampled the run's own item outcomes. It chained k items, each from a *different* task family, into one task that is solved only if every step is correct, treating steps as independent.
- **At 5 steps:** Gemini-4-Argon solves 53%, GPT-6.1-Terra 47%, Claude Haiku 5.5 25%, GPT-6-Luna 16% and Qwen3.8-Flash 13%.
- **At 15 steps:** Gemini solves about 10%, Terra 6% and Haiku 1%.

Steps are not equally hard. Saturated families contribute about 1.0, and the search and tracking families pull the product down. That is why Gemini's 15-step result (10%) falls below the uniform-difficulty figure 0.88¹⁵ ≈ 15%. Either way, a model that looks "88% good" on single steps fails most 15-step tasks. Real chained tasks also propagate early errors, so the independence assumption is optimistic. This is the arithmetic behind the NetHack, Pokémon and VSI-Super-Wild results. It argues for long-chain task families scored on full-episode success.

**pass@1 hides unreliability.** On 30 items sampled three times, Haiku, Luna and Qwen fall to **54–58% pass^3**, 15–23 points below their pass@1.

**A correct answer does not mean a valid trace.** A reasoning critic was grounded in the generator's key facts, with 90–100% test-retest agreement. It found that **12% (Gemini) to 30% (Haiku) of correct answers were not supported** by valid reasoning. For every model except Terra, **70–100% of failures began with a perception misread**, consistent with HSS's 94%.

**Synthetic skill does not always transfer to real images.** Haiku ranked third on the procedural suite but last on both real-image samples, near chance on VisuLogic (28%). This is a measured instance of the "synthetic-render skill" risk.

**Two older public sets may be near the top-model ceiling.** Gemini scored 68% on the VisuLogic sample, above the published 51.4% human baseline. Terra scored 92% on the BabyVision sample, against a published frontier best of about 50–69%. Both samples have only 25 items and 95% confidence intervals of about ±20 points, so this is a lead to chase, not a finding.

The run's stated limitations bound all of this:

- 150 items, giving a minimum detectable difference of 7–10 points;
- one effort setting;
- an OpenAI-family judge critiquing an OpenAI model;
- an inconclusive caption control;
- no human baseline yet.

## The market sells hours; the margin is in reusable assets

The industry that supplies frontier labs grew explosively in 2025–2026 and sorted into three tiers. The first is **expert human-data marketplaces**, which bill per expert-hour and pass 60–70% of revenue through to contractors. The second is **Scale AI**, flat at about $1B after Meta's stake. Its other lab customers reportedly stepped back, and it has had three CEOs since June 2025 ([CNBC](https://www.cnbc.com/2025/07/16/scale-ai-cuts-14percent-of-workforce-after-meta-investment-hiring-of-wang.html); [Wikipedia](https://en.wikipedia.org/wiki/Scale_AI)). The third is a newer group of **eval and environment companies** that now command venture valuations.

| Company | Model | Latest scale | Valuation | Source |
|---|---|---|---|---|
| Mercor | Expert marketplace + APEX benchmarks + acquired env startups | >$2B gross run rate (Jun 2026); net ~$0.6–0.8B; ~91% from labs | $10B (Oct 2025); $20B in talks | [Dealroom](https://dealroom.co/news/137121-mercor-doubles-to-2b-gross-revenue-run-rate-as-ai-labs-buy-expert-data/); [AI Weekly](https://aiweekly.co/alerts/mercor-books-614m-in-h1-2026-leaning-on-openai-anthropic) |
| Surge AI | Premium RLHF/expert data; RL-env group | ~$1.2B (2024) | $15–25B talks, unconfirmed | [Wikipedia](https://en.wikipedia.org/wiki/Surge_AI) |
| Handshake AI | Student/expert marketplace | ~$1B gross / ~$300M net (Apr 2026) | ~$2.7B secondary | [Dealroom](https://dealroom.co/news/127345-handshakes-arr-crosses-1b-as-ai-training-revenue-surges/) |
| micro1 | Expert marketplace + datasets | ~$500M gross run rate | $4B (Oct 2026) | [Dealroom](https://dealroom.co/news/155583-micro1-raises-100m-at-4b-valuation-for-ai-training-data/) |
| Snorkel AI | Data-as-a-service + RL environments | $375M run rate, 18x in 12 months | $3.5B (Sep 2026) | [TechCrunch](https://techcrunch.com/2026/09/22/snorkel-ai-triples-valuation-to-3-5b-as-demand-for-ai-training-data-booms/) |
| Scale AI | Full stack; Scale Labs benchmarks (VTB, HSS); defense | Just under $1B (2025) | ~$29B implied (2025) | [Wikipedia](https://en.wikipedia.org/wiki/Scale_AI) |
| Arena | Public leaderboard funnel → paid evals | >$100M annualized | $3.1B (Oct 2026) | [Angel Investors Network](https://angelinvestorsnetwork.com/venture-capital/arena-raises-200m-in-series-b-funding-at-3-1b-valuation) |
| Vals AI | Confidential benchmarking | ~$1.3M est. | ~$400M (Aug 2026) | [Pondero](https://pondero.ai/news/2026-09-20-vals-ai-series-a/) |
| Patronus AI | Pivoted to simulated "digital worlds" | >15x YoY growth | $50M Series B | [TechCrunch](https://techcrunch.com/2026/06/25/patronus-ai-lands-50m-to-build-digital-worlds-that-stress-test-ai-agents/) |

Mercor's growth comes on thin margins:

- **Rates:** experts average about $93–110/hr, and buyers pay roughly $150–165/hr.
- **Gross margin:** 27% in 2025 and 33% in Q2 2026. Handshake runs about 30%.
- **The contrast:** micro1 reports **80–90% gross margins on off-the-shelf datasets** ([AITraining.jobs](https://aitraining.jobs/research/what-ai-labs-pay-for-human-data)).

RL environments sit between the two. Prices run $200–2,000 per task, about $20k per website replica and about $300k for a high-fidelity product clone such as Slack. Contracts run six to seven figures per quarter, with $300k–500k seen at a neolab ([Epoch AI](https://epoch.ai/gradient-updates/state-of-rl-envs)).

Lab spend is enormous and concentrated:

- **Meta → Scale:** at least $450M a year committed.
- **Google → Scale:** about $150–200M a year before it pulled back.
- **OpenAI:** about $20k per UI-gym website, with hundreds bought.
- **Anthropic:** leaders discussed spending over $1B on RL environments in a year.

The first three come from [AITraining.jobs](https://aitraining.jobs/research/what-ai-labs-pay-for-human-data), the Anthropic figure from [Epoch AI](https://epoch.ai/gradient-updates/state-of-rl-envs).

Four trends define the period. Product has moved from labels to expert data, and from expert data to RL environments. **Neutrality became a procurement criterion** after Meta's Scale deal. Consolidation has begun: Mercor reportedly bought Deeptune and Sepal, and Labelbox bought Upcraft ([RL List](https://www.rl-list.com/rl-environments-for-computer-use)). And labs are in-sourcing commodity work. Evals became a venture category in their own right, but credibility still gets hurt by conflicts, the canonical case being OpenAI's late-disclosed funding of FrontierMath ([TechCrunch](https://techcrunch.com/2025/01/19/ai-benchmarking-organization-criticized-for-waiting-to-disclose-funding-from-openai/)).

Vision specifically is **under-served**. The most visible commercial vision eval is Scale's VisualToolBench, with a top score of 18.7% ([Scale](https://scale.com/blog/visualtoolbench)). The crowded "multimodal" segment is egocentric and robotics data capture, where raw footage costs $15–40/hr and is falling toward commodity pricing ([ChinaTalk](https://www.chinatalk.media/p/take-ten-thousand-egocentric-data)). The researchers found no venture-scale company dedicated to visual-reasoning evals or vision RL environments.

## Simulation environments win on authoritative state, not photorealism

Simulation-based evaluation splits into two families that barely overlap.

**Embodied and robotics simulators** verify success by reading privileged simulator state, and they optimize for throughput with small policies. Examples include Isaac Lab, ManiSkill3, Habitat, BEHAVIOR-1K, ProcTHOR, CARLA, and UnrealCV/UnrealZoo.

- BEHAVIOR-1K checks 1,000 household activities against logical goal predicates over simulator state ([arXiv 2403.09227](https://arxiv.org/abs/2403.09227)).
- CARLA scores route completion and infractions from engine state ([arXiv 1711.03938](https://arxiv.org/abs/1711.03938)).
- ManiSkill3 reaches 30,000+ FPS of GPU-parallel simulation and rendering ([ICLR 2025](https://iclr.cc/virtual/2025/32495)).
- UnrealZoo already offers **100+ photoreal Unreal worlds with a Gym API**, and finds that environment diversity improves RL generalization ([arXiv 2412.20977](https://arxiv.org/abs/2412.20977)).

**Game benchmarks for VLM and LLM agents** check win/loss or game flags. They show frontier models remain weak at real-time, visually grounded play. VideoGameBench added a "Lite" mode that pauses the game while the model thinks, because inference latency crippled real-time play ([arXiv 2505.18134](https://arxiv.org/abs/2505.18134)).

Scaffolding is the dominant confound. On ARC-AGI-3, one model scored 0% without a harness and 97.1% with a handcrafted one on one environment, yet 0% under both conditions on another. ARC Prize therefore bans domain-specific harnesses from its official board ([ARC-AGI-3 Technical Report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)). Self-reported harness scores of 95–100% on the public set remain unverified ([schema-harness](https://schema-harness.github.io/); [arXiv 2607.28287](https://arxiv.org/pdf/2607.28287)). A separate line of work has coding agents *write* game-playing programs instead of acting frame by frame ([arXiv 2609.18996](https://arxiv.org/pdf/2609.18996)). An environment product has to declare whether it measures the raw model or the model plus its harness.

The technical requirements follow from how buyers judge environments. **Reward hacking is buyers' top concern.** Models have searched online for solutions and checked out future commits in SWE-bench repos, and robust graders take many iterations ([Epoch AI](https://epoch.ai/gradient-updates/state-of-rl-envs)). METR found o3 reward-hacking in every trajectory on one RE-Bench task ([METR](https://metr.org/blog/2025-06-05-recent-reward-hacking)). For a UE5 world this implies several design rules:

- Compute rewards in a separate process, from authoritative engine state: actor transforms, gameplay tags, inventory and trigger volumes. Never use anything the agent can write.
- Log full state traces for audit.
- Randomize answer-relevant variables per seed.
- Run lockstep, fixed-timestep stepping with a pause-on-think mode.
- Verify at the level of task state, not pixel equality, because physics is generally not bit-deterministic across hardware. This last point comes from engineering practice, not a cited source.
- Treat VLM judges as a secondary signal only. MineDojo's learned video-text reward shows where open-ended judging leads ([arXiv 2206.08853](https://arxiv.org/abs/2206.08853)).

Per-step model inference costs seconds, and Mechanize estimates about **$2,400 of RL compute per task**. Either dwarfs the cost of rendering a frame, so photorealism is affordable for frontier-VLM agents. It is not affordable for million-step small-policy RL, where ManiSkill and Isaac's tiled rendering win.

**Licensing is the biggest UE5-specific risk.** Epic's EULA currently restricts using Unreal and MetaHuman *outputs* as AI training data. On October 27, 2026, Tim Sweeney said Epic is "working to remove this restriction," but gave no timeline ([Game Developer](https://www.gamedeveloper.com/programming/epic-is-updating-the-eula-to-allow-ai-training-on-outputs-from-unreal-engine-and-metahuman)). A separate clause bars MetaHuman characters from building training databases ([ConductAtlas](https://conductatlas.com/platform/unreal-engine/unreal-engine-eula/provision/CA-P-059499/prohibition-on-using-metahuman-for-ai-database-building/)). Unity's July 2026 terms reportedly require prior authorization to train on Unity-derived data ([ConductAtlas](https://conductatlas.com/change/2026-07-01-unity-unity-terms-of-service-3395/)). Seat cost is trivial at $1,850 per seat per year ([Epic](https://www.unrealengine.com/en-US/blog/we-are-updating-unreal-engine-twinmotion-and-realitycapture-pricing-in-late-april)). The restriction is the problem, not the price.

The market for 3D environments is real but mostly elsewhere. 3D and simulation capital is flowing to robotics simulation and world models:

- Machenta raised $40M ([LAVX](https://news.lavx.hu/article/machenta-raises-40-million-to-train-robots-in-virtual-worlds)).
- Meshy raised $400M and Decart $300M ([Crunchbase News](https://news.crunchbase.com/venture/2026-global-gaming-startup-funding-up-ai-meshy-decart/)).
- World Labs and AMI Labs reportedly raised about $1B each ([Capital & Compute](https://capitalandcompute.net/blog/world-model-funding-wave-2026/)).

NVIDIA's open-source Isaac Lab-Arena, with 250+ contributed tasks from Lightwheel, is commoditizing simulated robot-policy evaluation ([NVIDIA](https://developer.nvidia.com/blog/simplify-generalist-robot-policy-evaluation-in-simulation-with-nvidia-isaac-lab-arena)). Google DeepMind trains SIMA 2 in commercial games and in its own Genie 3 worlds. Genie 3 runs at 720p and 24fps, stays consistent for several minutes and has about one minute of visual memory ([DeepMind](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/)). Hassabis describes a loop in which "whatever SIMA is trying to learn, Genie can generate on the fly" ([Humanoids Daily](https://www.humanoidsdaily.com/news/deepmind-ceo-demis-hassabis-world-models-and-infinite-training-loops-are-the-keys-to-agi)).

The researchers found **no funded startup selling UE5 or Unity environments to frontier labs for VLM agents**. That is either white space or weak demand, and the evidence cannot tell which. Generative world models threaten the "diverse worlds" pitch but not the "verifiable reward" pitch: a pixel-generated world has no ground-truth state, so every judgment falls back to a hackable VLM judge. They also open a second product, using engine-rendered ground truth to evaluate world models for physical and 3D consistency.

## Labs buy verifiable RL task families, mostly for training

The demand evidence points one way. Anthropic's Vision-team role makes the purchase explicit. Its responsibilities include owning "the data strategy for vision capabilities end-to-end, from building evals and scaling RL environments", managing external data vendors, "writing task specifications", building QA for reward hacking, and running "generalization experiments to measure how data strategy changes improve multimodal capabilities". The salary range is $350k–$850k ([Anthropic Greenhouse](https://job-boards.greenhouse.io/anthropic/jobs/5074217008)). Anthropic also hires capability-specific environment engineers, for example in cybersecurity RL. That suggests environment procurement is organized by vertical ([Anthropic Greenhouse](https://job-boards.greenhouse.io/anthropic/jobs/5412334008)).

Epoch's interviews with 18 practitioners give the buying criteria. RL is the main use, with one startup estimating **10–20x more RL requests than benchmarking requests**. Buyers want a pass-rate floor of about 2–3%, meaning at least one success in 64–128 rollouts. They prefer a difficulty mix around 0%, 5% and 30%, and retire tasks near 70%. Demand is growing for long-horizon, multi-turn and enterprise workflows. "Maintaining quality while scaling is the number one bottleneck," and labs report "substantially more in-housing" to avoid vendor margins and keep priorities secret ([Epoch AI](https://epoch.ai/gradient-updates/state-of-rl-envs)).

Epoch's survey of environment demand does not mention multimodal or 3D environments at all. Vision demand is real but nascent, documented in hiring rather than in disclosed spend.

Adjacent buyers are weaker. World-model companies are more likely competitors or suppliers than customers. Robotics evaluation is moving toward independent auditors, such as Robocurve's $10M seed as a public-benefit corporation ([Robocurve](https://robocurve.org/blog/seed-raise/)), and toward free NVIDIA tooling. Neither looks like a reliable first revenue line for a vision-reasoning startup.

## VisionBench's opening: procedural visual RL tasks on a 2D-to-3D ladder

### The gap

The gap is specific. The RL-environment vendor landscape is crowded with text, tool and GUI players:

- Halluminate: $30M Series A, claiming 4 of 5 leading US closed labs as customers.
- AfterQuery: $30.5M raised.
- HUD: $16M Series A.
- Fleet: $15M.
- Collinear, Chakra, Matrices and Plato.

None of them is positioned around visual reasoning as a training product ([RL List](https://www.rl-list.com/rl-environments-for-computer-use)). The nearest neighbours are partial:

- Scale's VisualToolBench and HSS, which are benchmarks from a vendor with neutrality baggage;
- ARC Prize, a nonprofit with no commercial eval revenue;
- Refresh's BlenderBench, Andon Labs and BenchFlow in embodied niches;
- Patronus's pivot to simulated worlds for agents.

The large gaps are in non-verbalizable perception, state over time, multi-view and metric spatial reasoning, and long chains. Those are the hardest gaps to fill with human-written question–answer pairs and the easiest to fill with generators and simulators that hold exact state.

### The wedge

The wedge is **difficulty-calibrated, state-verified visual RL task families**, sold to lab vision teams in the vocabulary of the Anthropic posting: task families, verifiers, pass-rate calibration, hack resistance and generalization lift. A public, audited benchmark serves as the calling card. VisionBench already has the core assets:

- 15 seeded generators with computed keys and difficulty knobs;
- an HMAC-seeded private split that can be rotated;
- render seeds that vary appearance while keeping the answer;
- blind and caption controls, a cross-vendor defect audit and clustered statistics.

The first run also pins down what to build next. Search, tracking and rule induction separate models at 27–65%, while five families are saturated and need L6+ levels. The pass-rate band that labs ask for (2–70%) can be set directly by generator parameters rather than by filtering against particular models.

Two pieces of evidence favour answers that are not plain text. BabyVision found RLVR gains only in verbalizable subtypes, and HSS failures are overwhelmingly perceptual. Families whose answers are spatial (coordinates, paths, masks, actions checked against simulator state) are therefore more likely to train real visual skill than another text-answer quiz. That is a testable hypothesis, and testing it should be the company's first research result.

### The moat

The moat is a generator-and-verifier library that keeps refreshing and resists hacking. Behind it sit:

- human-baseline data on the same tasks;
- an audit trail that meets Epoch's reviewability bar;
- the switching costs of integration into lab RL stacks.

The margin argument is structural. VisionBench ran 1,771 scored attempts for $13.40 in inference. Its items have near-zero marginal authoring cost, against about $2,500 per expert-authored Mercor task and 60–70% contractor pass-through at marketplaces. That puts it in the 80–90% gross-margin asset band rather than the roughly 30% labor band.

The static public benchmark is the weakest moat. Two lessons from the run should shape the private sets. ARC showed that labs can train on a published generator's distribution, so private splits need **mechanics that are out of distribution**, not just new seeds. And Haiku's synthetic-versus-real divergence shows that generator families need external validity checks against real-image sets at larger n.

### The risks

Five risks are concrete:

1. **In-sourcing.** Already reported, and strongest for anything a lab team can replicate in a quarter.
2. **Buyer concentration.** About 91% of Mercor's revenue comes from labs.
3. **Incumbent entry.** Mercor, Surge, Scale or Snorkel can add a vision vertical from existing contractor pools.
4. **Fast saturation.** ARC-AGI-3 moved from 0.4% to 62.7% in seven months.
5. **Engine licensing and transfer.** Epic's restriction is unlifted, and 3D game worlds may not transfer to what labs value.

Neutrality is a sixth, subtler risk. Selling exclusive training families to one lab while running a "neutral" leaderboard invites contamination accusations. Benchmark and training sets must be kept separate, and the policy disclosed.

### A 12–18 month plan

| Window | Build | Proof point |
|---|---|---|
| Months 0–3 | Ship v2 fixes: numeric centre-of-mass for `block_stability`, L6+ for saturated families, a cross-family critic. Run the human baseline (`visionbench human-form`). Generate a ~3,000-item private split. Expand real-image tracks to n≥200. Publish with full harness metadata and defect logs. Package as Inspect- or `verifiers`-compatible environments. Open written inquiries with Epic. | Public, audited benchmark run on 5+ models from 4+ vendors. At least one frontier-lab researcher running it. An Epoch-style "reviewable" release. |
| Months 3–6 | Convert generators into RL task families with pass-rate calibration to the 2–70% band. Add hack probes (random-agent, known-exploit, answer-leak checks). Add long-chain families scored on full-episode success, plus video/stateful families: counting through occlusion, identity persistence, periodic events. | **Measured training lift** on an open VLM (Qwen-VL class), with transfer to independent sets (HSS image subset, BabyVision, VisuLogic, MMSI, VTB). This is the result the Anthropic role exists to evaluate. |
| Months 6–12 | Sell 1–2 paid pilots to lab vision teams, neolabs or open-weight labs, at low six figures per quarter, with one family offered exclusively at a 4–5x premium. Build interactive 2D/2.5D worlds (ARC-AGI-3-style puzzles, spatial manipulation, visually dense apps such as CAD and maps) on an engine without AI-training restrictions. | Signed pilot and a renewal. Gross margin reported against marketplace benchmarks. |
| Months 12–18 | Move into 3D: lockstep, state-graded worlds. Use UE5 only if Epic's EULA change is in force (Godot, Blender or MuJoCo otherwise). Make photoreal variants of proven families. Pilot world-model consistency evals using engine ground truth. | A benchmark cited in a frontier model card. A second paying lab. A 3D family showing transfer to real-image spatial benchmarks. |

## Conclusion

The research changes what VisionBench should think of itself as. It is not a leaderboard company that might later build environments. It is a **visual-environment company whose benchmark is marketing and quality assurance**. The money, the buyer persona and the margin structure all point to verifiable task families. The run's most useful finding is not the ranking. It is that the same machinery that produces training tasks also audits them. Programmatic keys, blind controls and cross-vendor consensus caught five real defects in 150 items before any customer saw them, which is exactly the quality-at-scale problem that labs call their number one bottleneck.

The strategic bet that remains open is transfer. Two things are unknown: whether procedurally generated visual tasks train skills that show up on independent real-world benchmarks, and whether 3D game worlds train anything labs value beyond what Genie-class generators and GUI environments already supply. The evidence cannot settle either question, and Haiku's divergence between synthetic and real images is a warning. A training-lift experiment on an open model costs weeks, not quarters. It would show whether this is a large business or a niche benchmark, and it should come before any Unreal Engine 5 investment, which Epic's licensing currently makes premature anyway.
