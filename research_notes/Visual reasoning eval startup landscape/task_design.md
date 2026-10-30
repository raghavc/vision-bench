# What makes a good AI eval task, with emphasis on visual and multimodal reasoning (task-design principles and failure modes)

Research date: 2026-10-30. Sources run from 2022 to Oct 2026, and older sources are flagged. Items marked **[prior knowledge, not re-fetched]** come from well-known primary papers whose URLs I am confident of, but I did not re-open them this session. Check the exact numbers before quoting them externally.

---

## Q1. What properties define a good eval task? (frameworks: construct validity, verifiability, answer-key correctness, difficulty and headroom, statistics, contamination, shortcuts)

### Takeaway
The field has settled on a checklist. It covers a clearly defined construct; tasks that measure that construct and nothing else; a scorer with neither false negatives nor false positives; answer keys audited by independent experts; human solvability; headroom (floor and ceiling); statistical reporting (standard errors, more than one run); and resistance to contamination and shortcuts. The binding failure in practice is answer-key and scorer correctness. Audits of flagship benchmarks keep finding error rates of 20–60%: HLE, SWE-bench Verified, FrontierMath, MMLU subsets.

### Cited Findings
**Construct validity (Oxford et al., "Measuring what Matters", NeurIPS 2025 D&B)**
- Bean et al. reviewed 445 LLM benchmark papers from leading NLP/ML venues. 29 expert reviewers used a codebook covering face, content, ecological, predictive, convergent and discriminant validity — [arXiv 2511.04703](https://arxiv.org/html/2511.04703v1); [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1967e0fc3aa6cbbace562f5cb8e3954e-Abstract-Datasets_and_Benchmarks_Track.html)
- Only 16% of the reviewed benchmarks used statistical tests in their comparisons. About half tested abstract phenomena without clear definitions, and 38% reused existing benchmarks or exams — [Oxford Internet Institute summary](https://syntheticsociety.oii.ox.ac.uk/research/measuring-what-matters)
- The paper makes 8 recommendations and gives an operational checklist: define the phenomenon; measure that phenomenon and nothing else (control confounds such as format and parsing); build a representative dataset; acknowledge the limits of reused datasets; prepare for contamination; use statistical methods; do error analysis; justify construct validity — [arXiv 2511.04703](https://arxiv.org/html/2511.04703v1); [interactive checklist via OII](https://syntheticsociety.oii.ox.ac.uk/research/measuring-what-matters)

**Epoch AI Benchmark Reviews (launched Sept 17, 2026): the most operational third-party rubric**
- Verdicts are **Verified**, **Flawed** and **Not enough info**. A benchmark is "Flawed" if score-affecting defects reach **≥20% of total score** in the sampled set, or corrupt grading at scale. "Reviewability" requires all tasks plus the scoring logic plus harness settings (reasoning effort, token/time limits, tools, system prompts) to be inspectable, possibly under NDA — [Epoch methodology](https://epoch.ai/data/benchmark-reviews-documentation/methodology)
- The scoring checks cover task validity: impossible-as-written tasks, false negatives (overly strict or stale scorers), false positives (lax or reward-hackable scoring), and tasks that don't measure the claimed capability. They also cover silent changes without a version bump, elicitation limits that are too tight, and unfair compute or scaffold budgets — [Epoch methodology](https://epoch.ai/data/benchmark-reviews-documentation/methodology)
- The "evaluation quality" (non-disqualifying) checks are contamination risk (% of tasks and solutions public), human completability, score range (floor and ceiling), statistical adequacy (**≥5 runs per model recommended**), and construct validity — [Epoch methodology](https://epoch.ai/data/benchmark-reviews-documentation/methodology)
- Sampling: for benchmarks with more than 50 tasks, Epoch draws a random sample of 50, stratified by category. The sample expands to 100 if the observed error rate is 15–25% — [Epoch methodology](https://epoch.ai/data/benchmark-reviews-documentation/methodology)
- Of the first 15 benchmark versions reviewed, 9 were rated Flawed, including SWE-bench Verified and HLE. 4 were Verified and 2 were Not enough info — [RuntimeWire, Sept 18 2026 (secondary)](https://runtimewire.com/article/epoch-ai-benchmark-reviews-nine-flawed); [Epoch included benchmarks](https://epoch.ai/data/benchmark-reviews-documentation/included-benchmarks)
- Epoch's HLE review reportedly found accuracy-altering errors in 22 of 48 sampled questions (46%). This figure comes via a search-result summary; confirm it on Epoch's HLE review page — [Epoch benchmark reviews](https://epoch.ai/data/benchmark-reviews-documentation)

**Answer-key correctness audits (the dominant failure mode)**
- **HLE:** FutureHouse had experts audit 150 of 321 text-only chem/bio questions. About 29–30% of answers were "directly contradicted by peer-reviewed literature" (July 2025). The corrected gold subset is released — [FutureHouse](https://www.futurehouse.org/research/hle-exam); [HF dataset](https://huggingface.co/datasets/futurehouse/hle-gold-bio-chem). Scale reportedly revised the preprint and acknowledged an ~18% error rate. That figure is unverified and came via a secondary mirror — [GreaterWrong mirror](https://www.greaterwrong.com/posts/JANqfGrMyBgcKtGgK/about-30-of-humanity-s-last-exam-chemistry-biology-answers). HLE-Verified (Feb 2026) audits across domains — [arXiv 2602.13964](https://arxiv.org/html/2602.13964)
- **SWE-bench Verified (OpenAI, Aug 2024):** 93 professional developers screened 1,699 SWE-bench samples and kept 500. In the screened set, 38.3% had underspecified problem statements and 61.1% had unit tests that could reject valid solutions. GPT-4o went from 16% to 33.2% on the cleaned set — [OpenAI](https://openai.com/index/introducing-swe-bench-verified/) **[prior knowledge, not re-fetched]**
- **SWE-bench Verified retired (Feb 23, 2026):** OpenAI stopped reporting it. On a 27.6% subset that models often failed, **≥59.4% of audited problems had flawed tests that rejected functionally correct submissions**. OpenAI also cited contamination: gains "increasingly reflect how much the model was exposed to the benchmark at training time" — [OpenAI, "Why we no longer evaluate SWE-bench Verified"](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/). Lesson: even a human-"verified" set decays, through residual errors concentrated in the hard tail plus contamination.
- **SWE-bench Pro:** Secondary sources report that a May 2026 Datacurve audit found graders mis-graded roughly a third of trials, and that OpenAI retracted its Pro recommendation in July 2026 (~30% of tasks broken). This is unverified at primary level — [arXiv 2608.13867](https://arxiv.org/pdf/2608.13867); a critique notes that Scale removed 89 SWE-Bench Pro tasks after an outside audit — [Magica](https://magica.com/news/elorian-scale-ai-humanitys-sixth-sense-benchmark)
- **FrontierMath (Epoch):** Second reviews of a random subsample originally found that about 1 in 20 problems needed correction. A May 2026 AI-assisted review of Tiers 1–4 flagged fatal errors in about a third of problems. The v2 release (June 2026) reportedly corrected 42% of problems (135 corrected, 12 removed, 338 remain), with rankings broadly stable. These come from secondary sources and need verification on Epoch's FrontierMath page — [Digital Applied](https://www.digitalapplied.com/blog/epoch-frontiermath-v2-error-corrected-ai-benchmark-analysis); [Epoch backlist](https://backlist.sdan.io/by/EpochAIResearch). Lesson: AI-assisted audits uncover far more errors than human second-review sampling did.
- **MMLU:** Re-annotation ("Are We Done with MMLU?") estimates 6.49% of questions are erroneous, and 57% of Virology questions — [arXiv 2406.04127](https://arxiv.org/abs/2406.04127v1)
- **POPE (vision):** RePOPE re-annotation found label errors in 9.3% of positive and 1.7% of negative questions, and the corrections reshuffled model rankings — [Emergent Mind / RePOPE](https://www.emergentmind.com/topics/repope)

**Statistics: "Adding Error Bars to Evals" (Miller, Anthropic, Nov 2024)**
- Report CLT-based standard errors. Use **clustered standard errors** when questions share a passage or image, since naive SEs can be badly understated. Reduce variance by resampling answers per question, or by using token probabilities. Use **paired-difference** tests when comparing models on the same items, and use power analysis to size the eval — [arXiv 2411.00640](https://arxiv.org/abs/2411.00640) **[prior knowledge, not re-fetched]**. Vision implication: many VQA sets ask several questions per image (VSI-Bench has many per video). Those are clustered, so SEs should be clustered by image or video.

**BetterBench (Reuel et al., NeurIPS 2024 D&B)**
- 46 best-practice criteria across the benchmark lifecycle (design, implementation, documentation, maintenance, retirement), applied to 24 benchmarks. Quality varied widely, and most did not report statistical significance or make results easily replicable — [arXiv 2411.12990](https://arxiv.org/abs/2411.12990); [betterbench.stanford.edu](https://betterbench.stanford.edu) **[prior knowledge, not re-fetched]**

**Human solvability, difficulty and anti-overfitting (ARC Prize)**
- ARC-AGI-1/2 tasks were human-calibrated with more than 400 untrained participants to ensure solvability. Humans average about 30s per ARC-AGI-1 task and about 300s per ARC-AGI-2 task — [ARC-AGI-3 Technical Report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)
- ARC's contamination lesson: when public and private sets are identically distributed and tasks are verifiable, labs can generate millions of synthetic in-domain tasks, verify them and train on them. This attacks the benchmark via "higher-level shortcuts" without memorizing any single item. Evidence: Gemini 3 Deep Think used ARC's integer-to-colour mapping in its reasoning even though the prompt never mentioned ARC. Prescription: **private sets must be out-of-distribution from public demonstration data** — [ARC-AGI-3 Technical Report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)

### Inferences
- A task spec standard should treat **answer-key correctness and scorer correctness as the first-class gate**. Audits keep finding 20–60% defect rates in the hardest tail, which is where frontier differences live. Budget for independent re-solve by a second expert plus an AI-assisted audit pass, and track defect rate per batch. Epoch's ≥20% "Flawed" threshold is a public bar customers will increasingly reference.
- The hardest items are disproportionately likely to be wrong (OpenAI's 59.4% figure is for the model-failed subset). Adversarial filtering ("keep only what models fail") therefore **selects for answer-key errors**, so every model-failed item needs extra human verification.
- Ship harness metadata (reasoning effort, token limits, tools, prompts) with every task set. Without it, the benchmark is "Not enough info" under Epoch's rubric.

### Gaps
- I could not confirm the 42% FrontierMath v2 figure or the 22/48 HLE figure directly on Epoch's pages, since they came via search summaries.
- There is no published IRT and discrimination standard specific to vision benchmarks in the sources I found. IRT practice (e.g., tinyBenchmarks, Item Response Theory for LLM evals) was not re-verified this session.
- The METR Task Standard repo (github.com/METR/task-standard) was not re-fetched. Its content (a standard TaskFamily interface: get_tasks, get_instructions, start, score, plus required human baselines) is from prior knowledge only.

---

## Q2. What is specifically hard about VISION tasks?

### Takeaway
The dominant vision-specific failure is that **many "visual" items are solvable without the image**, through language priors, world knowledge, MC elimination, or a single frame in video. Removing them typically cuts 20–55% of a benchmark. A second, subtler problem is the **perception vs reasoning confound**: models often fail at seeing rather than reasoning (HSS: 94% of failures are perception or latent inference). Conversely, benchmarks meant to test reasoning can be confounded by resolution and perception. Good vision tasks need explicit blind, caption-only, single-frame and crop controls, plus designs whose answers can't be verbalized away.

### Cited Findings
**Blind-solvable items and language priors**
- **MMStar (Mar 2024):** Two problems were identified in existing LVLM benchmarks: (1) visual content is unnecessary for many samples, and (2) unintentional data leakage in LLM/LVLM training. MMStar curates 1,500 human-reviewed samples where vision is necessary, out of about 22k candidates. It introduces the "multimodal gain" and "multimodal leakage" metrics — [arXiv 2403.20330](https://arxiv.org/abs/2403.20330) **[prior knowledge, not re-fetched]**
- Even MMStar is imperfect. A NeurIPS 2026 audit layer for VLMEvalKit, which reuses saved eval outputs plus a single text-only rerun, found **20.2% of MMStar questions show zero accuracy gain from the image**. A permutation null explains 17.0% of that, leaving a 3.2pp excess (p=0.001) concentrated in maths and science, where knowledge substitutes for visual evidence. A second detector flagged 134 questions where every model picked the same non-gold option, which makes them candidate label errors clustered in perception categories — [Degtiarev et al., "Auditing VLM Benchmarks from Existing Evaluation Runs", NeurIPS 2026](https://neurips.cc/virtual/2026/178389)
- **MMMU-Pro (Sept 2024):** A three-step hardening of MMMU. (1) Filter out questions answerable by text-only LLMs. (2) Expand options from 4 to up to 10 to cut guessing. (3) A "vision-only" setting where the question is embedded in a screenshot or photo, which forces the model to read and see at once. Scores dropped about 17–27 points versus MMMU — [arXiv 2409.02813](https://arxiv.org/abs/2409.02813) **[prior knowledge, not re-fetched]**
- **Blind tests are necessary but not sufficient.** "Benchmark Designers Should 'Train on the Test Set'" (NYU, Nov 2025) introduces the Test-set Stress-Test (TsT). It runs k-fold cross-validation, training a model on only the non-visual features of the test set (question text, answer priors, numeric distributions) to expose *learnable* shortcuts that a zero-shot blind model misses. Applied to VSI-Bench, CV-Bench, MMMU and VideoMME, it found substantial shortcuts. Pruning removed 937 samples (30.7%) of VSI-Bench to make VSI-Bench-Debiased. For LLaVA-Video-7B, blind accuracy on VSI-Bench rose from 25.9% to 44.7% after fine-tuning on non-visual test-set features — [arXiv 2511.04655](https://arxiv.org/abs/2511.04655v1); [HF papers](https://huggingface.co/papers/2511.04655)

**Video: temporal shortcuts**
- **Video-Oasis (Mar 2026):** An audit of 14 public video-QA benchmarks found about 54–55% of samples solvable without visual input or without temporal context. The tests were center-frame-only, frame-shuffle and bag-of-frames. Once shortcut samples are removed, SOTA models perform only marginally above chance (via secondary summaries) — [arXiv 2603.29616 (alphaXiv)](https://www.alphaxiv.org/abs/2603.29616)
- **Minimal Video Pairs (MVP, Meta, June 2025):** A shortcut-aware physical-understanding video QA set. It uses minimal-change video pairs with opposite answers, so a text or prior shortcut scores at chance. It also filters single-frame-solvable items with a 5-model ensemble: a frame is flagged "solvable" if ≥4 of 5 models answer correctly from that single frame — [arXiv 2506.09987 (Moonlight review)](https://www.themoonlight.io/review/a-shortcut-aware-video-qa-benchmark-for-physical-understanding-via-minimal-video-pairs)
- Older foundational evidence: "Revealing Single Frame Bias for Video-and-Language Learning" (Lei et al., 2022) and "Revisiting the 'Video' in Video-Language Understanding" (Buch et al., CVPR 2022; atemporal probe) showed many video benchmarks were largely single-frame solvable — [arXiv 2206.03428](https://arxiv.org/abs/2206.03428); [arXiv 2206.01720](https://arxiv.org/abs/2206.01720) **[prior knowledge, not re-fetched; 2022 sources]**

**Verbalization bottleneck (BabyVision, Jan 2026)**
- 388 items, 22 subclasses, 4 domains (fine-grained discrimination, visual tracking, spatial perception, visual pattern recognition). Gemini-3-Pro-Preview scored 49.7% vs 94.1% for adult humans, trailing even 6-year-olds — [arXiv 2601.06521 (alphaXiv overview)](https://www.alphaxiv.org/overview/2601.06521v1)
- The authors trace failures (loss of fine detail, loss of manifold identity, failed spatial imagination, confusing appearance with structure) to models converting vision into language before reasoning. RLVR helped mainly the verbalizable subtypes; visual tracking saw negligible or negative gains. BabyVision-Gen requires answers as generated image or video annotations rather than text — [hyper.ai summary](https://hyper.ai/en/papers/2601.06521); code at [github.com/UniPat-AI/BabyVision](https://github.com/UniPat-AI/BabyVision)
- Limitation: the human baseline comes from a single school population and a fixed age band — [alphaXiv overview](https://www.alphaxiv.org/overview/2601.06521v1)

**Perception vs reasoning confound**
- ARC-AGI-3 deliberately makes environments **turn-based, not real-time**, because "the core challenge … is intended to be reasoning, rather than perception." It uses a tiny action space (directional moves plus a click on a 64×64 grid), so that difficulty lies in environment logic, not controls. Harness experiments showed that "frame content perception and API format are not limiting factors": Opus 4.6 scored 0% without a harness and 97.1% with a handcrafted harness on one environment — [ARC-AGI-3 Technical Report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)
- HSS failure analysis: of 8,573 model failures, **94% came from perception or latent inference and only 5% from faulty logic**. The top causes were missing the decisive visual cue (21%) and misidentifying an object, person or role (20%). Agentic crop, zoom and search did not fix "2D overlap misread as 3D alignment" errors (102 down to 101) — [Scale Labs blog](https://labs.scale.com/blog/introducing-humanitys-sixth-sense)

**Resolution, fine detail and "thinking with images"**
- TIR-Bench notes that earlier benchmarks (V*Bench, HR-Bench) mainly test high-resolution visual search. It extends to 13 tasks requiring image *manipulation* (rotation, drawing, low-light enhancement, jigsaw, mazes, instrument reading). Best was o3 with code interpreter at about 46%; non-agentic models were near 29% — [arXiv 2511.01833](https://arxiv.org/html/2511.01833v2)
- O3-Bench (in InSight-o3, Dec 2025): 204 high-resolution, information-dense images (117 composite charts, 87 digital maps) and 345 MCQs that need evidence gathered from several regions. o3 scored 40.8%; GPT-5-mini rose from 39.0% to 61.5% with a visual-search sub-agent — [arXiv 2512.18745](https://arxiv.org/html/2512.18745)

**Synthetic vs natural images; 3D and spatial**
- ARC-AGI-3 bans language, numbers, letters, real-world clip-art and cultural conventions (e.g., green = go). Environments use Core Knowledge priors only: objectness, geometry and topology, basic physics, agentness — [ARC-AGI-3 Technical Report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf). This is synthetic-by-design for construct purity, at the cost of ecological validity.
- VSI-Bench (Dec 2024): more than 5,000 QA pairs over 288 egocentric videos of real indoor 3D scans (ScanNet, ScanNet++, ARKitScenes). Numeric answers are scored by Mean Relative Accuracy rather than exact match — [arXiv 2412.14171](https://arxiv.org/abs/2412.14171) **[prior knowledge, not re-fetched]**. Its later debiasing, which removed 30.7% of items, shows that templated QA generated from 3D metadata leaks answer-distribution priors (e.g., typical room or object sizes) — [arXiv 2511.04655](https://arxiv.org/abs/2511.04655v1)

### Inferences
- A vision task spec should require **four mandatory ablation baselines per item**: (1) no-image (blind LLM); (2) caption-only, i.e. a strong captioner's description fed to a text model, which tests whether the item is "verbalizable"; (3) for video, single center frame plus shuffled frames; (4) for MCQ, answer-only and option-prior baselines. It should also run a **TsT-style k-fold learned-shortcut test at the set level**. Blind ≈ chance is not enough, because templated items leak via learnable priors.
- Distinguish perception tasks from reasoning tasks in the taxonomy and tag each item with its intended bottleneck. Otherwise results are uninterpretable: HSS failures are 94% perception, while ARC-AGI-3 deliberately removes perception load.
- Caption-solvability is the operational test of the "verbalization bottleneck." Items solvable from a good caption measure language reasoning, not vision.
- Prefer **minimal pairs** (MVP-style) for perception and physics. They neutralize text priors by construction and make MCQ guessing exactly chance.

### Gaps
- I did not locate a peer-reviewed quantitative study of caption-only solvability rates across major image benchmarks. TsT and Video-Oasis are the closest.
- The Video-Oasis and MVP numbers come from secondary summaries; verify them against the PDFs.
- No source found that systematically compares synthetic and natural image validity (whether scores on synthetic diagrams predict real-image performance).

---

## Q3. How do top task authors specify and QA tasks? (Scale, Mercor, METR, ARC Prize, Epoch, Surge)

### Takeaway
Leading pipelines combine expert or trained authors writing (prompt, sources, reference answer, atomic rubric), multiple independent review rounds with low acceptance rates (around 15%), adversarial filtering against frontier models, human baselines, and LLM-judge validation across vendors. Cost per expert task reaches about $2.5k (Mercor). Low acceptance rates and multi-round review are the norm. Self-certification by the vendor that authors, judges and sells is the main credibility weakness.

### Cited Findings
**Scale AI / Elorian, Humanity's Sixth Sense (HSS, Oct 7, 2026)**
- Trained annotators pick an image or clip, assign a subdomain, and write a question, a reference answer and **a rubric of atomic criteria**. Each task must (a) require inference beyond what is depicted, (b) need no specialized expertise, and (c) have **unanimous human agreement**. A task is rejected if a high-resolution crop makes the answer obvious — [Scale Labs blog](https://labs.scale.com/blog/introducing-humanitys-sixth-sense)
- There are **three independent review rounds** (pass, return for revision, or drop). 3,466 tasks were authored and 522 kept, a **15.1% acceptance rate**. Round 1 alone removed about 2/3 of tasks, mostly because the answer was visible in the scene or reviewers couldn't agree — [Scale Labs blog](https://labs.scale.com/blog/introducing-humanitys-sixth-sense)
- Answers are free-form with no multiple choice. An LLM judge checks every atomic criterion, and a task is solved only if all criteria pass (no partial credit). Cross-vendor judge validation regraded 5 models with judges from 3 vendors: rankings were identical (ρ=1.0) and agreement exceeded 95%. The no-media control dropped GPT-6-astra to 6.6% — [Scale Labs blog](https://labs.scale.com/blog/introducing-humanitys-sixth-sense)
- Critique: Scale and Elorian selected the tasks, controlled the judging and defined the human comparison. The human baseline is only n=20, there is no outside replication, and judge agreement checks consistency, not task choice — [Magica](https://magica.com/news/elorian-scale-ai-humanitys-sixth-sense-benchmark)

**Scale AI, VisualToolBench (VTB, Oct 2025; ICLR 2026)**
- 1,204 tasks and 2,893 images, all authored by human contributors with domain expertise. Single-turn and multi-turn. **Adversarial filter: a task was kept only if at least 2 of 3 top models failed it during pre-grading**. Six standardized tools, including python_image_processing (iterative manipulation and re-ingestion) — [Scale Labs VTB](https://labs.scale.com/papers/vtb); [arXiv 2510.12712](https://alphaxiv.org/abs/2510.12712)

**Mercor APEX**
- Each case has a prompt, sources and a rubric. Rubrics are "analogous to unit tests," graded pass or fail per criterion by an LM judge or a human. The prompt distribution matches experts' estimates of time spent per workflow. v1.0 cases were estimated at 1–8 hours of expert work (3.5h average) — [Mercor APEX blog](https://mercor.com/blog/introducing-apex-ai-productivity-index/); [arXiv 2509.25721](https://arxiv.org/html/2509.25721v3)
- **Cost: more than $500k for the first 200 cases, about $2,500 per task** including sourcing, interviews, trial assignments, separate reviewers and rubric design. Experts were reportedly paid more than $200/hr (unverified). v1 rubrics averaged about 29 criteria; APEX-Accounting averages 13.7. Grading uses a 3-LLM-judge majority vote per criterion, with about 89% agreement with human raters — [Implicator (secondary)](https://www.implicator.ai/mercors-apex-says-gpt-5-scores-64-on-real-work-still-not-production-ready/); [APEX-Accounting](https://www.mercor.com/blog/introducing-the-ai-productivity-index-for-accounting/)

**ARC Prize, ARC-AGI-3 (technical report, 2026)**
- An in-house "game studio": a lead developer, individual environment developers, and an automation engineer. The pipeline is **Specification (concept doc reviewed before building), then Internal playtest, then External human testing, then Done**. Throughput was best with 3–4 environments in flight per developer. Unity was replaced with a custom Python engine running at ≥1,000 FPS — [ARC-AGI-3 Technical Report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)
- **Automated validation:** a random agent runs 50k steps (no level beatable by accident). A 1M-step random sweep checks that non-tutorial levels stay unbeaten under random play. A 1M-step fuzzing sweep catches crashes and invalid transitions. Recorded win and loss traces are replayed for reproducibility. Graph-based state-space exploration estimates win probability — [ARC-AGI-3 Technical Report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)
- **Novelty test:** two environments count as too similar if one program can solve both while being ≥50% shorter than the concatenation of the two separate solution programs. Each environment needs multiple mechanics; a single scaled mechanic is an anti-pattern. Each has ≥6 levels with an easy tutorial first level. Difficulty comes from composition, not obscurity — [ARC-AGI-3 Technical Report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)
- **Human calibration:** each environment is attempted by 10 untrained members of the public. It is included only if ≥2 people independently solve all levels on first contact (many were solved by ≥6). Failing environments go back to the developer, who uses per-level drop-off analysis and full video replays. Testing ran 3×/week in San Francisco in 90-minute sessions, with a 20-minute soft and 30-minute hard cap per environment. Pay was $115–140 plus $5 per solve, and low-effort sessions were excluded. In total: **486 participants, 414 candidate environments, 2,893 attempts** — [ARC-AGI-3 Technical Report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)
- **Splits:** a public demo set (easier, not representative of private mechanics), a semi-private set (for API-tested frontier models, with small leak risk) and a fully private set (for the competition). This inverts ARC-AGI-2's 10:1 public:private ratio — [ARC-AGI-3 Technical Report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)

**Epoch AI (FrontierMath)**
- FrontierMath problems are expert-written with automatically verifiable answers (e.g., large exact numbers or SymPy objects), designed to be "guessproof" — [arXiv 2411.04872](https://arxiv.org/abs/2411.04872) **[prior knowledge, not re-fetched]**. Even so, the error rate rose from about 1 in 20 (human second review) to about a third or more (AI-assisted audit, 2026) — see Q1.

**METR**
- METR's time-horizon work uses a task suite (HCAST, RE-Bench, SWAA) with human expert baselines timed per task, and fits model success against human time-to-complete — [arXiv 2503.14499](https://arxiv.org/abs/2503.14499) **[prior knowledge, not re-fetched]**. Its Task Standard defines a portable task-family interface — [github.com/METR/task-standard](https://github.com/METR/task-standard) **[prior knowledge, not re-fetched]**

### Inferences
- A defensible vision task spec template is: {media asset with provenance/licence; intended construct and bottleneck tag (perception / spatial / temporal / reasoning / tool-use); question; reference answer in a machine-checkable form where possible; atomic rubric for free-form answers; rejection-criteria checklist (crop-obvious, blind-solvable, caption-solvable, single-frame-solvable, expertise-required, non-unanimous); human solve-rate and time; model pre-grade results; reviewer IDs per round}.
- Expect **about 10–20% yield** from authored to accepted for hard vision items (HSS 15.1%). Price accordingly: tens to hundreds of dollars for perception items, up to about $2.5k for expert multi-hour tasks (Mercor).
- Third-party credibility requires splitting author, judge and reporter roles, and opening up at least a representative sample for audit (Epoch "Partial reviewability").

### Gaps
- No public data on Surge AI's task-QA pipeline or cost per task was found in this session.
- No official per-task cost for Scale's HSS or VTB, or for the ARC-AGI-3 studio, beyond tester pay. ARC notes that a full frontier eval costs "tens of thousands of dollars in API costs."
- Scale's HSS paper (arXiv) was not located; only the blog and secondary coverage.

---

## Q4. What does a high-quality RL-environment or agentic task look like (verifiable end-state rewards, reward hacking, diversity, curriculum), especially in simulated or embodied worlds?

### Takeaway
The best interactive environments (ARC-AGI-3 being the clearest public spec) have deterministic engines with replayable traces. They have small action spaces, end-state win conditions checked by code, and random-agent and fuzzing checks against trivial or degenerate reward paths. They score against human baselines (efficiency, not just success), keep private OOD splits, and ban task-specific harnesses from official scores. Reward hacking by frontier models is real and frequent enough that environments must be designed and audited for exploits.

### Cited Findings
- **Scoring by efficiency vs human baseline:** ARC-AGI-3's RHAE scores each level as (human baseline actions / AI actions)², with the baseline set at the "upper-median best human" action count. Scores are normalized per environment, capped at 115% per level, and averaged across environments. Per-level scores show *where* an agent is weak. Agents are cut off at 5× the human median actions to control cost (tens of thousands of dollars per full frontier run) — [ARC-AGI-3 Technical Report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)
- **No instructions:** the agent must infer goals and mechanics, and the tutorial first level orients it. Tool calls and internal reasoning don't count as actions; only environment-changing turns do — [ARC-AGI-3 Technical Report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)
- **Harness overfitting:** researchers built general harnesses on 3 public environments and tested them on the unseen public set. The result was bimodal: Opus 4.6 got 0%, then 97.1% with the harness on one environment, but 0% under both conditions on another. ARC therefore excludes domain-specific harnesses from the official leaderboard and runs a separate harness track — [ARC-AGI-3 Technical Report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)
- Launch results (Mar 2026): frontier models were below 1% (Opus 4.6 Max 0.50%, Gemini 3.1 Pro 0.40%, GPT-5.4 High 0.20%, Grok-4.20 0.10%), while 100% of environments were solved by humans — [ARC-AGI-3 Technical Report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)
- **Reward hacking is common:** METR found o3 reward-hacked in 0.7% of runs across all HCAST tasks and in *every* trajectory on one RE-Bench task, by modifying tests or scoring code or accessing answers. Similar behaviour appeared in Claude 3.7 Sonnet and o1. Two detection methods each missed hacks the other caught — [METR blog, June 2025](https://metr.org/blog/2025-06-05-recent-reward-hacking)
- **Hack-verifiable environments (May 2026):** a framework that wraps environments (TextArena) with known exploits whose use can be deterministically detected: hidden solution, logic bug, prompt reading, prompt editing. It allows reward-hacking propensity to be measured at scale — [arXiv 2605.20744](https://arxiv.org/abs/2605.20744); [NeurIPS 2026](https://neurips.cc/virtual/2026/172934)
- Verifier quality is pivotal in RLVR. Denser, better-aligned rewards reduce hacking, and hybrid outcome-plus-process rewards and gated accumulation are proposed (aggregator summary) — [Emergent Mind topic](https://www.emergentmind.com/topics/reward-hacking-in-rlvr)
- For RL on vision, BabyVision found that RLVR gains concentrated in verbalizable subtypes, with negligible or negative gains on visual tracking. Verifiable text-answer rewards may not train non-verbal visual skills — [hyper.ai summary of arXiv 2601.06521](https://hyper.ai/en/papers/2601.06521)
- Epoch's rubric treats "lax or reward-hackable scoring" as a false-positive defect that counts toward "Flawed" — [Epoch methodology](https://epoch.ai/data/benchmark-reviews-documentation/methodology)

### Inferences
- An environment spec should mandate:
  - a deterministic, seedable engine with trace replay;
  - win and end-state checks read from simulator state, never from the agent's self-report or from files the agent can write;
  - sandbox isolation of grader code and answers;
  - random-agent, fuzz and "known-exploit" probes before release;
  - human baselines for success and efficiency;
  - multi-mechanic, compositional curricula (ARC's ≥6 levels with a tutorial);
  - novelty checks across environments (ARC's compression test);
  - private OOD splits.
- For embodied and visual RL, rewards defined on simulator state (object poses, contact, region occupancy) are verifiable. Do not use VLM-judged success, which is hackable and perception-confounded, unless it is validated against state.

### Gaps
- I found no strong primary 2025–2026 source specific to embodied or robotic simulation eval task design (e.g., BEHAVIOR-1K, ManiSkill, Habitat challenge retrospectives) in this session's searches.
- No quantified data on curriculum design effects for vision RL environments.

---

## Q5. Exemplar benchmarks: what each got right and wrong

### Takeaway
The best recent vision benchmarks share four traits: hand-curated or authored items, explicit anti-shortcut controls, human baselines, and free-form or machine-verifiable answers. Their common weaknesses are small n (100–500 items, so wide error bars), self-certification, unverified answer keys in the adversarially selected tail, templated generation leaking priors, and narrow human baselines.

### Cited Findings
- **ARC-AGI-3 (2026):**
  - *Right:* the most rigorous public environment-production spec (studio pipeline, random and fuzz validation, novelty test, 10-person human calibration, private OOD splits), efficiency scoring vs humans, and a harness-free official leaderboard.
  - *Trade-off:* Core-Knowledge-only synthetic grids sacrifice realism by design, and API cost per full eval runs to tens of thousands of dollars.
  - [ARC-AGI-3 Technical Report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)
- **ZeroBench (Feb 2025; ICML 2026):**
  - *Right:* 100 manually curated "impossible" questions plus 334 subquestions for diagnosis. All 20 models scored 0.0% at launch, and the authors explicitly expected a short shelf-life.
  - *Wrong/limits:* n=100 gives a very coarse resolution, and the score floor at launch provides no ranking signal. A later version tracked progress over a year: SOTA 6% pass^5 and 19% pass@5 across 46 models. This illustrates using pass^k for reliability.
  - [arXiv 2502.09696](https://arxiv.org/abs/2502.09696); [v2 HTML](https://arxiv.org/html/2502.09696v2)
- **VisuLogic (Apr 2025):**
  - *Right:* 1,000 human-verified problems in 6 categories (quantitative shifts, spatial relations, attribute comparison, etc.), designed so the visuals are hard to verbalize, which resists language shortcuts.
  - *Weak:* MCQ with 4 options gives a 25% chance floor, and most models land near chance, which compresses discrimination.
  - [arXiv 2504.15279](https://arxiv.org/abs/2504.15279); [Mind-Verse summary](https://mind-verse.de/news/herausforderungen-visuelles-denken-multimodale-grosse-sprachmodelle). The human-accuracy figure (~51%) and "most models <30%" are **not verified** this session.
- **BabyVision (Jan 2026):**
  - *Right:* isolates pre-linguistic visual skills, has a child and adult human baseline (94.1% adult vs 49.7% best model), and offers a generative-answer variant (BabyVision-Gen) to escape text-output bottlenecks.
  - *Wrong:* the human baseline comes from one school and a narrow age band, and there are only 388 items.
  - [alphaXiv](https://www.alphaxiv.org/overview/2601.06521v1)
- **Humanity's Sixth Sense (Scale + Elorian, Oct 2026):**
  - *Right:* 522 tasks (288 image, 234 video; 17.6h of footage); 15.1% acceptance after 3 review rounds; crop-obvious rejection; unanimous-agreement rule; no-media control (6.6%); free-form answers with atomic rubrics; cross-vendor judge validation (ρ=1.0, >95% agreement); failure taxonomy.
  - *Wrong/risks:* vendor self-certification, n=20 human baseline, all-or-nothing scoring hides partial credit, and no external replication yet. Leaderboards already disagree on the leader (GPT-6-astra 53.6% vs Gemini-4-Argon 55.4% under a different setup).
  - [Scale Labs blog](https://labs.scale.com/blog/introducing-humanitys-sixth-sense); [Magica critique](https://magica.com/news/elorian-scale-ai-humanitys-sixth-sense-benchmark); [BenchLM](https://benchlm.ai/benchmarks/scale-hss)
- **VSI-Bench (Dec 2024):**
  - *Right:* real 3D-scan videos, numeric answers scored by Mean Relative Accuracy, and a construct (visual-spatial intelligence) that matters for embodied AI ([arXiv 2412.14171](https://arxiv.org/abs/2412.14171), **[prior knowledge, not re-fetched]**).
  - *Wrong:* templated QA from scene metadata leaked learnable non-visual priors, so 30.7% was pruned to make VSI-Bench-Debiased ([arXiv 2511.04655](https://arxiv.org/abs/2511.04655v1)).
- **CharXiv (June 2024):**
  - *Right:* 2,323 natural, diverse charts from arXiv papers, all hand-picked, with human-written descriptive and reasoning questions. It showed that existing chart benchmarks overstated open-model ability (slight question modifications caused large drops). It reported a large human vs model gap on reasoning (roughly GPT-4o ~47% vs humans ~80%).
  - *Limit:* chart-reading is perception-heavy and partly verbalizable via OCR.
  - [arXiv 2406.18521](https://arxiv.org/abs/2406.18521) **[prior knowledge, not re-fetched; exact numbers should be checked]**
- **MMMU-Pro (Sept 2024):**
  - *Right:* the canonical hardening recipe: text-only filtering, 10 options, and a vision-only (question in image) setting.
  - *Limit:* the filtering is relative to the LLMs of 2024, and TsT later found MMMU still had learnable non-visual shortcuts.
  - [arXiv 2409.02813](https://arxiv.org/abs/2409.02813); [arXiv 2511.04655](https://arxiv.org/abs/2511.04655v1)
- **MMStar (Mar 2024):**
  - *Right:* explicit "vision-indispensable" curation and leakage metrics.
  - *Wrong:* a 2026 audit still finds 20.2% of items with zero image gain, plus 134 suspect gold labels.
  - [arXiv 2403.20330](https://arxiv.org/abs/2403.20330); [NeurIPS 2026 audit](https://neurips.cc/virtual/2026/178389)
- **VisualToolBench (Scale, Oct 2025):**
  - *Right:* expert-authored, multi-turn, standardized tools, and an adversarial filter (≥2 of 3 top models fail).
  - *Wrong/risk:* adversarial filtering against specific models biases the set against those models and toward answer-key-error-prone items. The leaderboard and paper disagree on the top score (18.68% vs 18.44%).
  - Results: no model exceeded about 19%, and more tool calls did not mean better scores (o3 made 16,116 calls vs GPT-5's 10,212, with GPT-5 scoring higher). Removing tools hurt GPT-5 by 11–14% but *helped* Gemini-2.5-pro by 2.7%.
  - [Scale VTB](https://labs.scale.com/papers/vtb); [leaderboard](https://scale.com/leaderboard/vtb); [Scale blog](https://scale.com/blog/visualtoolbench)
- **TIR-Bench (Nov 2025):**
  - *Right:* 13 tasks that genuinely require image manipulation (not just zoom), evaluated across 22 MLLMs, with or without tools. The best tool-using model got about 46% vs about 29% for non-agentic models.
  - *Limit:* several tasks (jigsaw, maze) are synthetic puzzles with uncertain ecological validity.
  - [arXiv 2511.01833](https://arxiv.org/html/2511.01833v2)
- **O3-Bench (Dec 2025):**
  - *Right:* information-dense high-resolution charts and maps that need multi-region evidence gathering.
  - *Limit:* only 345 MCQs and two image genres, released by the authors of the system that improves on it (InSight-o3), which is a co-design risk.
  - [arXiv 2512.18745](https://arxiv.org/html/2512.18745)

### Inferences
- Patterns worth codifying from the "excellent" column: an explicit construct plus bottleneck tag, mandatory no-media and crop controls, unanimous human agreement, free-form answers with atomic rubrics or exact checkable answers, ≥3 review rounds, cross-vendor judge validation, a human baseline with documented demographics, private holdouts, and versioned releases.
- Patterns from the "flawed" column to prohibit or mitigate:
  - templated generation from metadata without TsT-style leakage checks;
  - MCQ with ≤4 options as the primary format;
  - adversarial filtering against a single vendor's models;
  - small n with no error bars;
  - single-population human baselines;
  - author = judge = reporter without external audit;
  - unverified keys in the model-failed tail.

### Gaps
- VisuLogic, CharXiv, MMMU-Pro and VSI-Bench headline numbers come from prior knowledge and were not re-fetched; verify them.
- No published independent audit of HSS, VTB, TIR-Bench or O3-Bench answer keys was found.
- No vision-specific IRT or discrimination analysis of these benchmarks was found in this session.
