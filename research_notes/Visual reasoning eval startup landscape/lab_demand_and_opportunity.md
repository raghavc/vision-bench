# Lab Demand and Startup Opportunity: Visual-Reasoning Evals + Simulated-World Environments (as of 2026-10-30)

Method note: 16 tool calls (searches + fetches). Primary sources fetched in full: Epoch AI RL-environment FAQ (2026-01-12), Anthropic "Research Engineer, Visual Knowledge Work" posting (live on Greenhouse as of 2026-10-30), Robocurve seed announcement (2026-09-14), RL List vendor directory (updated 2026-10-05). Many other figures come from search-result summaries of aggregator or blog sites; these are flagged "[secondary]" and should be checked before being quoted as fact. The ARC Prize leaderboard page did not expose its scores in text form, so I could not verify ARC-AGI-3 scores at the primary source.

## 1. Demand signals from frontier labs (2025–2026): hiring, leader statements, procurement, budgets

### Takeaway
The strongest evidence is a live Anthropic Vision-team role whose job includes **writing vendor-facing specs for visual RL training tasks** and managing external data vendors. That means at least one frontier lab buys vision RL environments and evals from outside. Spending on RL environments overall is large (Anthropic discussed >$1B over a year, per a Sept 2025 report), but buyers ask for RL environments 10–20x more often than for benchmarks. So the money is in training environments, with evals as a secondary purchase.

### Cited Findings
**Hiring / job postings**
- (Live 2026-10-30) Anthropic, "Research Engineer, Visual Knowledge Work" (Vision team). Responsibilities: "Own the data strategy for vision capabilities end-to-end, from building evals and scaling RL environments"; "Manage technical relationships with external data vendors, including writing task specifications"; "Develop and improve QA frameworks that catch reward hacking and ensure environment quality at scale". Representative project: "Writing a vendor-facing specification for a new family of visual RL training tasks". Salary $350k–$850k. — [Anthropic Greenhouse](https://job-boards.greenhouse.io/anthropic/jobs/5074217008)
- (Posted ~2026-09-03 / 2026-10-10 per mirrors) OpenAI, "Researcher, Multimodal Safety", on the Chat and Multimodal Safety team, which builds "research, training methods, and evaluations" for multimodal behavior. Preferred background includes "multimodal post-training or evaluations". [secondary mirrors, not verified on openai.com] — [Glassdoor](https://www.glassdoor.com/job-listing/jv?jl=1010258777831); [Zapply](https://zapply.jobs/jobs/02c01303-aba1-41c9-b767-800b62262382/)
- Job postings show in-house RL-environment teams at xAI and Anthropic; one founder reported "substantially more in-housing" recently (Jan 2026). — [Epoch AI FAQ](https://epoch.ai/gradient-updates/state-of-rl-envs)
- Anthropic also hires domain-specific RL-environment engineers (e.g., "Research Engineer, Cybersecurity RL", Zürich), a sign that environment building is organized by capability vertical. — [Anthropic Greenhouse](https://job-boards.greenhouse.io/anthropic/jobs/5412334008)

**Budgets / spend**
- (Reported Sept 2025, by The Information, cited by Epoch 2026-01-12) Anthropic discussed spending over $1B on RL environments over the following year. — [Epoch AI FAQ](https://epoch.ai/gradient-updates/state-of-rl-envs)
- (Epoch, 2026-01-12) OpenAI's projected 2026 R&D compute spend is about $19B, roughly double 2025. Epoch notes that compute will still dwarf environment spending. — [Epoch AI FAQ](https://epoch.ai/gradient-updates/state-of-rl-envs)
- (Epoch, 2026-01-12) Pricing benchmarks:
  - Contracts are often six to seven figures per quarter; one neolab researcher saw $300k–$500k contracts.
  - UI-gym website replicas run about $20k each (SemiAnalysis); high-fidelity product clones (e.g., Slack) about $300k.
  - Tasks run about $200–$2,000 each, rarely about $20k.
  - Exclusive deals cost about 4–5x non-exclusive ones.
  - Mechanize estimates about $2,400 of RL compute per task.
  — [Epoch AI FAQ](https://epoch.ai/gradient-updates/state-of-rl-envs)
- (Reported mid-2026) Mercor passed a $2B annualized gross revenue run-rate in June 2026, about four months after $1B. It booked about $614M gross in H1 2026, with roughly 91% from foundation-model labs. It pays 60–70% of topline to contractors. [secondary] — [Dealroom](https://dealroom.co/news/137121-mercor-doubles-to-2b-gross-revenue-run-rate-as-ai-labs-buy-expert-data/); [AI Weekly](https://aiweekly.co/alerts/mercor-books-614m-in-h1-2026-leaning-on-openai-anthropic)
- Surge AI reportedly generated about $1.2B revenue in 2024 (Forbes, via Contrary) and set up an internal RL-environment org by Sept 2025. No figure newer than 2024 has been published. [secondary] — [Contrary Research](https://research.contrary.com/company/surge-ai); [HeroHunt](https://www.herohunt.ai/blog/mercor-vs-surge-vs-scale-ai-2026-buyer-guide/)
- (July 2026) An industry tally counts 50+ vendors selling data and RL environments to labs, with about $8.5B combined annual revenue. This comes from a job-board site, so it is low reliability. — [aitraining.jobs](https://aitraining.jobs/roles/agentic-eval)

**Procurement patterns: who buys what**
- (Epoch, 2026-01-12) Buyers are frontier labs, neolabs and trailing labs. Products are used for three things: RL, benchmarking, and SFT on trajectories. One startup employee estimated RL requests outnumber benchmarking requests 10–20x ("RL is the main use"). Trailing labs prefer SFT. Vendors often cannot see how customers use their products. — [Epoch AI FAQ](https://epoch.ai/gradient-updates/state-of-rl-envs)
- (Epoch, 2026-01-12) The domains Epoch covers are SWE, computer use (screenshot-based, Playwright, MCP), math, finance, and enterprise workflows (Excel, Salesforce, Bloomberg-terminal clones). The FAQ does **not** discuss multimodal-specific or game/3D environments. — [Epoch AI FAQ](https://epoch.ai/gradient-updates/state-of-rl-envs)
- (Epoch, 2026-01-12) Buyers' quality criteria:
  - Reward hacking is the top concern.
  - Difficulty calibration: at least about a 2–3% pass rate (one success in 64–128 rollouts), a spread of difficulties, and tasks retired near a 70% pass rate.
  - "Maintaining quality while scaling is the number one bottleneck."
  — [Epoch AI FAQ](https://epoch.ai/gradient-updates/state-of-rl-envs)

**Lab-leader statements (vision / world models / agents)**
- (Late 2025) Demis Hassabis said he spends most of his research time on world models. He said SIMA 2 and Genie 3 are already used internally to train robots and other agents, and called world models essential for robotics and AGI. — [The Decoder](https://the-decoder.com/deepmind-ceo-hassabis-world-models-are-the-future-but-the-ai-bubble-is-real/)
- (Dec 2025 podcast) Hassabis described a Genie + SIMA loop: "whatever SIMA is trying to learn, Genie can generate on the fly", i.e., an open-ended loop of generated training worlds. — [Humanoids Daily](https://www.humanoidsdaily.com/news/deepmind-ceo-demis-hassabis-world-models-and-infinite-training-loops-are-the-keys-to-agi)
- Genie 3 launched Aug 2025. Project Genie opened to US AI Ultra subscribers on 2026-01-29. — [Wikipedia: Genie](https://en.wikipedia.org/wiki/Genie_(AI_model))

### Inferences
- The Anthropic posting is the clearest proof of a buyer: a Vision team that owns evals plus RL environments, outsources task creation to vendors, and needs reward-hacking QA. A visual-reasoning startup should sell to that buyer persona with that vocabulary: task families, verifiers, pass-rate calibration, hack-resistance.
- Budgets flow to **training environments with verifiers**, not to standalone benchmarks (10–20x request ratio). A pure "eval company" is fishing in the smaller pond. Benchmarks work best as a calling card that converts into RL-environment contracts.
- Google DeepMind's internal Genie/SIMA loop suggests GDM will generate 3D worlds in-house rather than buy them. It may still buy evaluation content or human-baseline data. This is inference only; I found no GDM procurement evidence.

### Gaps
- I found no verified 2025–2026 job postings or procurement evidence for multimodal evals or environments at Meta, xAI (beyond the generic "in-house" mention), Amazon, Apple, NVIDIA (as a buyer), or Chinese labs (DeepSeek, Qwen/Alibaba, ByteDance Seed, Moonshot). Not searched in depth because of the tool budget.
- I found no lab-level budget figure specific to *multimodal or vision* environments; the $1B figure covers RL environments overall.
- No 2026 primary statements from Altman, Amodei, Zuckerberg, or Musk on vision or spatial reasoning were collected.

## 2. Adjacent markets: robotics foundation models, world-model companies, AV, defense. Do they buy external evals or environments?

### Takeaway
Capital is pouring into world models and robotics: World Labs about $1B (Feb 2026) and AMI Labs about $1B seed (Mar 2026), both [secondary]. Independent embodied-eval startups have started to appear (Robocurve $10M seed, Sept 2026; YC's Enact and Runtime). But NVIDIA is commoditizing simulated robot-policy evaluation through the open-source Isaac Lab-Arena, with Lightwheel contributing benchmark suites. I found no direct evidence that Physical Intelligence, Figure, Skild, or 1X pay for external evals.

### Cited Findings
- (2026-09-14) Robocurve raised a $10M seed led by Initialized Capital, with Notable Capital, Decasonic, Y Combinator, Halcyon Futures and others.
  - Positioning: an independent third-party auditor of how frontier AI models control real robots, structured as a Public Benefit Corporation. "Frontier labs do not direct our evaluations or control our methodology or public results."
  - Programs and traction: a $500k bounty plus free YAM arms for academic benchmark groups; 200+ institutions signed up; 6M+ research views; 97k+ downloads of its "Inspect Robots" harness in the past three months.
  - No customers are named and no business model is stated.
  — [Robocurve blog](https://robocurve.org/blog/seed-raise/)
- (2026) YC robotics companies:
  - Enact (YC S2026) "creates the physical environments to benchmark and hill climb robotics models on high-value tasks".
  - Runtime.si (YC F2026) is "building the evaluation layer for physical intelligence".
  - Funding is undisclosed for both.
  — [YC robotics directory](https://www.ycombinator.com/companies/industry/robotics)
- NVIDIA Isaac Lab-Arena is an open-source framework for large-scale simulated evaluation of generalist robot policies, letting users swap scene, robot, or task independently.
  - Lightwheel contributed 250+ tasks from RoboCasa/LIBERO and is building RoboFinals, an industrial benchmark; RoboTwin is extending its 2.0 benchmark onto Arena.
  - Arena is integrated with Hugging Face LeRobot, with GR00T N and PI0.5 policies available for evaluation (Jan 2026).
  - Cosmos Predict/Transfer 2.5 are positioned for synthetic data and policy evaluation.
  — [NVIDIA Developer blog](https://developer.nvidia.com/blog/simplify-generalist-robot-policy-evaluation-in-simulation-with-nvidia-isaac-lab-arena); [Isaac Lab-Arena docs](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.1/pages/motivation/motivation.html); [MarketScreener](https://www.marketscreener.com/news/nvidia-releases-new-physical-ai-models-as-global-partners-unveil-next-generation-robots-ce7e59dede80f420)
- (July 2025) Genesis AI emerged with a $105M seed (Eclipse, Khosla) to train on synthetic data from a proprietary physics simulator. It is a model company that owns its simulator, not an eval vendor. [secondary] — [StartupIntros](https://startupintros.com/orgs/genesis-ai)
- World-model funding [secondary, valuations conflict across sources]:
  - World Labs (Fei-Fei Li) reportedly raised $1B in Feb 2026 (Autodesk $200M; AMD, NVIDIA, Fidelity also named). It launched a world-model API in Jan 2026, and its Marble product is public.
  - AMI Labs (Yann LeCun) reportedly raised about $1.03B seed, announced 2026-03-10, at a valuation reported variously as $3.5B pre-money, €3B, and $4.5B.
  - A Sept 2026 report quotes AMI co-founder Michael Rabbat saying AMI is still in R&D with no public product timeline.
  — [Capital & Compute](https://capitalandcompute.net/blog/world-model-funding-wave-2026/); [Wikipedia: AMI Labs](https://en.wikipedia.org/wiki/Advanced_Machine_Intelligence_Labs); [Tech Beat](https://techbeat.co/story/inside-ami-labs-and-world-labs-secretive-race-to-commercialize-world-models)
- Academic reviews note that "the lack of standardized evaluation protocols hinders fair comparison" in robot manipulation foundation models. — [arXiv 2512.22983](https://arxiv.org/pdf/2512.22983); [arXiv 2404.18201](https://arxiv.org/pdf/2404.18201)
- RL List notes embodied or visual benchmarks among environment vendors:
  - Andon Labs (YC W24): Drone-Bench and Butter-Bench; described as an Anthropic research partner.
  - BenchFlow: Robo Use, GenesisBench.
  - Refresh (YC): BlenderBench (3D animation and simulation in Blender).
  - HUD: covers robotics tasks.
  — [RL List](https://www.rl-list.com/rl-environments-for-computer-use)

### Inferences
- World-model companies (World Labs, AMI, Decart, Odyssey) are more likely **suppliers or competitors** of simulated worlds than buyers. They might buy evals of physical and spatial consistency for their own models. That is a speculative but plausible niche: "evals for world models", testing physics plausibility, object permanence and 3D consistency.
- In robotics, NVIDIA's free Arena plus Lightwheel's suites make "sim eval harness" a commodity layer. Startups there are differentiating on independence and real-hardware evaluation (Robocurve) or high-value physical tasks (Enact). A software-only visual-reasoning startup would face Arena as a free substitute.
- Independence is being sold explicitly (Robocurve's PBC structure and "labs do not direct our evaluations"), which suggests investors fund neutrality as a thesis.

### Gaps
- No evidence found on whether Physical Intelligence, Figure, Skild, 1X, Tesla Optimus, Waymo or Wayve buy external evals or environments. No search was done on autonomous-driving sim vendors (Applied Intuition, Foretellix, Parallel Domain) or defense simulation (e.g., Anduril, Palantir, DoD CDAO), apart from RL List noting Scale's Pentagon CDAO ceiling of $500M ([RL List](https://www.rl-list.com/rl-environments-for-computer-use)).
- Decart and Odyssey funding and business models were not researched.

## 3. Competitive landscape for "visual-reasoning evals + sim environments"

### Takeaway
The RL-environment market is crowded, but almost all of it targets **text, tool, and GUI computer use** (enterprise app clones, browsers, desktops). Incumbent data vendors are consolidating it (Mercor bought Deeptune and Sepal; Scale and Snorkel have environment offerings). Visual-reasoning evals exist mainly as **benchmarks run by incumbents or nonprofits**: Scale's VisualToolBench and ARC Prize's interactive ARC-AGI-3. I found no well-funded startup focused specifically on visual-reasoning or 3D-world RL environments for frontier labs. That is either white space or a sign of weak demand.

### Cited Findings
**Vision benchmarks / eval players**
- (Oct 2025) Scale AI + UIUC VisualToolBench: 1,204 open-ended vision tasks (603 single-turn, 601 multi-turn) across STEM, medical, finance, sports and generalist domains.
  - Models get a 6-tool API (python_image_processing, etc.) with a 20-call cap and are graded by rubric.
  - A task was kept only if at least 2 of 3 top MLLMs failed it during pre-grading.
  - The best model (GPT-5-think) scored about 18.4–18.7%.
  - The dataset is on Hugging Face.
  — [Scale blog](https://scale.com/blog/visualtoolbench); [arXiv 2510.12712](https://www.alphaxiv.org/abs/2510.12712.md); [Scale leaderboard](https://scale.com/leaderboard/vtb)
- (2026-03-25) ARC Prize launched ARC-AGI-3: interactive game environments with no stated goal or rules.
  - At preview, frontier LLMs scored below 0.4% (Gemini 3.1 Pro about 0.37%, Claude Opus 4.6 about 0.2%); the best purpose-built preview agent reached 12.58%; humans solved all environments.
  — [ARC-AGI-3 paper arXiv 2603.24621](https://arxiv.org/pdf/2603.24621); [Medium summary](https://medium.com/@AdithyaGiridharan/arc-agi-3-dropped-and-frontier-ai-scored-less-than-1-90cd70e65a61)
  - Later claims (all unverified):
    - Claude Opus 5 at 30.16% vs GPT-5.6 Sol at 13.33% on the public set (July 2026), per a third-party directory. [secondary, NOT verified on arcprize.org] — [Capital & Compute](https://capitalandcompute.net/ai-benchmarks/arc-agi-3/)
    - A harness team claims about 99% on the public set; ARC Prize has not verified this. — [Schema harness](https://schema-harness.github.io/)
- ARC Prize operates as a foundation, with donations and Kaggle prize competitions; no commercial eval revenue is visible. — [ARC Prize leaderboard](https://arcprize.org/leaderboard)

**RL-environment vendors (directory updated 2026-10-05)** — all from [RL List](https://www.rl-list.com/rl-environments-for-computer-use) unless noted
- AfterQuery, $30.5M (Altos; Forbes reported a $3.2B valuation in Sept 2026): expert data, computer-use trajectories, real-task benchmarks.
- Halluminate, $38.5M total ($30M Series A, Oct 2026, Oak HC/FT): finance and consulting benchmarks and environments; claims work with 4 of 5 leading US closed labs (self-reported).
- HUD, $16M Series A (June 2026): environment SDK, hosted training and eval; browser, computer-use and robotics tasks.
- Fleet AI, $15M (Sequoia, Menlo): enterprise gyms.
- Collinear, $10M: stateful simulation lab.
- Chakra Labs, about $10M: enterprise software clones.
- Huzzle Labs, $6M.
- Matrices, $5M: browser "gamified replica of the internet".
- Plato: simulated websites and Linux desktop; funding undisclosed.
- Andon Labs: embodied and drone benchmarks.
- BenchFlow: embodied benchmarks.
- Refresh: BlenderBench.
- Incumbents:
  - Scale AI: $1.6B raised; Meta's roughly $14.3B deal (June 2025); desktop-VM and MCP environments.
  - Snorkel AI: $350M Series E at $3.5B (Sept 2026); computer-use, terminal and enterprise environments.
- Infrastructure: Daytona ($79M), E2B ($32M), Cua ($500K).
- Consolidation:
  - Deeptune raised a $43M Series A led by a16z (Mar 2026) and was acquired by Mercor in July 2026. — [RL List](https://www.rl-list.com/rl-environments-for-computer-use)
  - Mercor also acquired Sepal AI (Feb 2026). [secondary] — [AI Weekly](https://aiweekly.co/alerts/mercor-books-614m-in-h1-2026-leaning-on-openai-anthropic)
- Mechanize: $9.1M seed (Apr 2026) at a reported $500M post-money. One source says Google DeepMind acqui-hired it (Sept 2026); another says it continues independently. This is a **conflict and unverified**. — [Synaptic](https://synaptic.com/resources/tip-offs/rl-evironments-startups-and-founders); [HeroHunt](https://www.herohunt.ai/blog/hiring-rl-environment-engineers-2026-labs-guide/)
- An industry guide counts about 20 seed-to-Series-A environment startups in 2026 and expects consolidation to 3–5 leaders by 2030 (opinion). [secondary] — [HeroHunt](https://www.herohunt.ai/blog/hiring-rl-environment-engineers-2026-labs-guide/)
- Traditional data providers (Mercor, Surge, Handshake, Turing) now sell RL environments. Their advantage is fast staffing and QA infrastructure. — [Epoch AI FAQ](https://epoch.ai/gradient-updates/state-of-rl-envs)

**Embodied / sim**
- Robocurve ($10M), Enact, Runtime.si, Lightwheel and NVIDIA Isaac Lab-Arena (see Section 2).

### Inferences
- **White space:** the vendor lists contain no funded company positioned around *visual reasoning* (charts, diagrams, spatial or geometric reasoning, multi-image, video, visual tool use) as a training-environment product. The closest are Scale (VTB as a benchmark), Refresh (BlenderBench), and Andon and BenchFlow (embodied). Anthropic's "visual knowledge work" posting shows the demand exists.
- **Threat:** Mercor, Scale, Surge and Snorkel can add a vision vertical quickly using their contractor pools. The defensible part is therefore *programmatic and procedural generation with exact verifiers*, i.e., unlimited, hack-resistant, difficulty-calibrated visual tasks, not human-labeled image QA.
- ARC-AGI-3 shows that interactive visual game environments are where frontier models are weakest and where public attention is. It also shows that a nonprofit benchmark can capture mindshare without revenue.

### Gaps
- Mercor's "extended sets", Epoch's vision benchmarks, and academic visual benchmark saturation (MMMU, MathVista, BLINK, etc.) were not researched in this pass.
- No funding data found for Chinese vision or eval vendors.
- No data on whether VisualToolBench has been updated with 2026 models.

## 4. Business models, neutrality value, and risks

### Takeaway
The evidence supports a model of **RL environments and task families sold per quarter (six to seven figures), with exclusivity at a 4–5x premium**, with benchmarks and leaderboards used as marketing and as a credibility signal. The main risks are lab in-sourcing (already reported), buyer concentration (Mercor gets about 91% of revenue from foundation-model labs), saturation, consolidation by incumbents, and game-engine license limits (Unity now requires authorization for AI training on its data).

### Cited Findings
- Pricing and exclusivity:
  - Contracts run six to seven figures per quarter, with $300k–$500k observed at a neolab.
  - Exclusive deals cost 4–5x non-exclusive ones.
  - Tasks run $200–$2k each; website replicas about $20k; complex clones about $300k.
  — [Epoch AI FAQ](https://epoch.ai/gradient-updates/state-of-rl-envs)
- Buyers ask for RL use far more than benchmarking (10–20x). Benchmarks are a minority use. — [Epoch AI FAQ](https://epoch.ai/gradient-updates/state-of-rl-envs)
- In-sourcing risk: "substantially more in-housing" was reported, for three reasons: avoiding vendor margins, keeping priorities confidential, and building in-domain expertise. — [Epoch AI FAQ](https://epoch.ai/gradient-updates/state-of-rl-envs)
- Concentration risk: about 91% of Mercor's revenue comes from foundation-model labs, mostly a handful of them (H1 2026). [secondary] — [AI Weekly](https://aiweekly.co/alerts/mercor-books-614m-in-h1-2026-leaning-on-openai-anthropic)
- Scaling bottleneck: "Maintaining quality while scaling is the number one bottleneck"; one founder's revenue is capped by contractor QA. — [Epoch AI FAQ](https://epoch.ai/gradient-updates/state-of-rl-envs)
- Saturation cycles:
  - ARC-AGI-3 went from below 1% (Mar 2026) to a reported 30% verified (July 2026) and a claimed 99% with a custom harness. [secondary/unverified] — [Capital & Compute](https://capitalandcompute.net/ai-benchmarks/arc-agi-3/); [Schema harness](https://schema-harness.github.io/)
  - Labs retire tasks near a 70% pass rate. — [Epoch AI FAQ](https://epoch.ai/gradient-updates/state-of-rl-envs)
- Neutrality as a product:
  - Robocurve is structured as a PBC with methodology independent of labs. — [Robocurve](https://robocurve.org/blog/seed-raise/)
  - ARC Prize is run as a foundation. — [ARC Prize](https://arcprize.org/leaderboard)
- Game-engine licensing:
  - Unity's ToS (updated 2026-07-01) reportedly requires prior authorization from Unity to train AI/ML models on Unity data or data derived from it. Whether "derived" covers rendered frames from a Unity-built sim is unclear; one blog cites Section 17.2. [secondary] — [ConductAtlas change log](https://conductatlas.com/change/2026-07-01-unity-unity-terms-of-service-3395/); [Engines Database](https://enginesdatabase.com/blog/state-of-unity-licensing-in-2026/)
  - Epic's EULA has prohibited using Unreal/MetaHuman outputs as AI training input. Epic (Tim Sweeney) said it would remove that restriction; the effective date is unconfirmed. — [Game Developer](https://www.gamedeveloper.com/programming/epic-is-updating-the-eula-to-allow-ai-training-on-outputs-from-unreal-engine-and-metahuman); [Times of AI](https://www.timesofai.com/news/epic-games-plans-eula-changes-to-allow-ai-training/)
  - Fab marketplace assets can carry a "NoAI" tag. — [Unreal forums via search](https://forums.unrealengine.com/t/unreal-engine-marketplace-licenses-question/1935404)
- Distribution via third-party platforms: Amazon sued Perplexity over agentic shopping and blocks most AI agents. This makes simulated replicas, not live sites, the safer substrate for agent environments. — [Epoch AI FAQ](https://epoch.ai/gradient-updates/state-of-rl-envs)

### Inferences
- **Recommended model (inference):** a hybrid with three parts.
  1. A small public, versioned visual-reasoning benchmark and leaderboard for credibility and model-card citations.
  2. Private held-out eval suites licensed per lab per quarter.
  3. The main revenue: procedurally generated, verifiable visual RL task families and environments, sold non-exclusive by default with an exclusivity premium.
- **Engine choice:** build on open-source renderers and simulators (Blender, MuJoCo, Isaac/Omniverse, custom WebGL, or Godot) rather than Unity, to avoid licensing gatekeeping. Godot as a safe choice is an inference; its license terms were not researched.
- **Neutrality** is worth most in evals (model cards, procurement, safety), not in training environments, where labs want exclusivity. The two can conflict: selling exclusive training data to Lab A while running a "neutral" leaderboard invites contamination accusations. Keep separate held-out sets and disclose the policy.

### Gaps
- No public pricing for eval-as-a-service or per-rollout environment licensing.
- No data on how often lab model cards cite third-party vision benchmarks in 2026.

## 5. Credible wedge and 12–18 month plan; traction that convinces labs and investors

### Takeaway
The evidence suggests the wedge should be **"verifiable visual-reasoning RL task families" that a lab Vision team can drop into training**: procedurally generated, difficulty-calibrated to a 2–70% pass band, and hardened against reward hacking. A public hard benchmark should serve as the door-opener. The proof points that matter are a paid pilot (six figures per quarter is the market norm), measured training lift on the buyer's held-out evals, and benchmark citations in model cards.

### Cited Findings (evidence underpinning the plan)
- The buyer persona and deliverable are spelled out by Anthropic: "vendor-facing specification for a new family of visual RL training tasks", "QA frameworks that catch reward hacking", and "generalization experiments to measure how data strategy changes improve multimodal capabilities". — [Anthropic Greenhouse](https://job-boards.greenhouse.io/anthropic/jobs/5074217008)
- Acceptance criteria labs use:
  - at least about a 2–3% pass rate, or one success in 64–128 rollouts;
  - a spread of difficulties;
  - retire tasks near 70%;
  - reward-hacking resistance.
  — [Epoch AI FAQ](https://epoch.ai/gradient-updates/state-of-rl-envs)
- Hard visual benchmarks attract attention: VTB's top score was below 19%, and ARC-AGI-3 launched with frontier models below 1%. — [Scale blog](https://scale.com/blog/visualtoolbench); [arXiv 2603.24621](https://arxiv.org/pdf/2603.24621)
- Comparable funding milestones:
  - HUD $16M Series A (June 2026); Halluminate $30M Series A (Oct 2026), claiming 4 of 5 US closed labs as customers; Matrices $5M; Robocurve $10M seed. — [RL List](https://www.rl-list.com/rl-environments-for-computer-use); [Robocurve](https://robocurve.org/blog/seed-raise/)
  - Robocurve's traction metrics as a pre-revenue eval company: 97k harness downloads, 6M research views, 200+ institutions. — [Robocurve](https://robocurve.org/blog/seed-raise/)

### Inferences (opinion / proposed plan — not evidence)
- **Months 0–3:**
  - Publish a hard, contamination-resistant visual-reasoning benchmark with a procedural track (unlimited fresh instances, exact verifiers) and a real-image track. Run it across 5+ frontier models from 4+ vendors.
  - Release an open harness, as Inspect-style or HUD-compatible environments, so lab researchers can run it in minutes.
  - Target metric: at least one frontier-lab researcher citing or running it.
- **Months 3–6:**
  - Turn the benchmark generators into RL task families with difficulty knobs.
  - Show *training lift* on an open VLM (e.g., Qwen-VL class): train on the procedural tasks and measure transfer to independent benchmarks such as VTB and MMMU-style sets. This is the key proof, because labs buy what moves their evals, and generalization experiments are what the Anthropic role runs.
- **Months 6–12:**
  - Land 1–2 paid pilots with lab vision or multimodal teams (Anthropic Vision team as the documented buyer; also neolabs and trailing or open-weight labs, which prefer SFT data).
  - Target pilot size: low six figures per quarter, consistent with the $300k–$500k contracts Epoch reports.
  - Offer exclusivity on a task family at a 4–5x premium.
- **Months 12–18:**
  - Extend to interactive visual environments: 2D/3D puzzle worlds in the ARC-AGI-3 style, spatial manipulation, and screenshot-based computer use in visually dense apps (CAD, charts, maps).
  - Possibly add world-model evals (physics and 3D consistency) for World Labs/AMI-type companies.
  - Get a benchmark cited in at least one frontier model card.
- **Investor-convincing metrics:**
  - signed paid pilot(s) with a frontier lab, plus a renewal;
  - documented training lift;
  - model-card citations;
  - number of labs running the public benchmark;
  - gross margin advantage of procedural generation over human-labeled tasks.
  - Highlight the margin point: Mercor passes 60–70% to contractors, so procedural generation is a structural margin moat. [Mercor data secondary — AI Weekly](https://aiweekly.co/alerts/mercor-books-614m-in-h1-2026-leaning-on-openai-anthropic)
- **Moat candidates:**
  1. a generator library plus verifiers (hard to hack and continually refreshed, which counters saturation);
  2. human-baseline data on the same tasks;
  3. trust and independence for the eval product;
  4. integration into lab RL stacks (switching costs).
  - The weakest moat is the static benchmark itself, which saturates within months (ARC-AGI-3 reportedly went from under 1% to 30% in about 4 months).

### Gaps
- No direct evidence on what conversion rate from free benchmark to paid contract eval startups achieve.
- No evidence found that any lab has publicly paid for a third-party *vision-specific* eval or environment, other than the Anthropic posting implying vendor use.
- Model-card citation patterns for third-party vision benchmarks in 2026 were not checked.
