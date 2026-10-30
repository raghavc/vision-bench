# Industry vendors supplying evals, training data and RL environments to frontier AI labs (focus: multimodal / visual)

Research date: 2026-10-30. Labels used below: **[disclosed]** means stated by the company or in its own release/filing. **[reported]** means attributed by a news outlet to sources. **[est.]** means a third-party estimate (Sacra, aggregator, analyst). Many figures in this space come only from aggregators. Where sources conflict, that is flagged.

Reading convention: "gross revenue / gross run rate" for marketplace vendors (Mercor, Handshake, micro1) **includes contractor payouts (60–70%)**, so net revenue is roughly 30–40% of the headline. Snorkel reports run rate net of expert payments, so the two kinds of figure are not comparable.

---

## Q1. Who are the major players and what is their 2025–2026 status?

### Takeaway
Since mid-2025 the market has split into three groups. (1) **Expert human-data marketplaces** (Mercor, Surge, Handshake AI, micro1, Turing, Invisible, Labelbox/Alignerr, Toloka, Prolific) grew very fast; several passed $1B in gross run rate in 2026. (2) **Scale AI** is stalled: revenue is flat at about $1B, it is on its third CEO, and it is pivoting toward government and defense after Meta's stake. (3) **Eval / benchmark companies** (Arena/LMArena, Artificial Analysis, Vals AI, Patronus, plus nonprofits Epoch AI and METR, and platforms like Kaggle Game Arena and Hugging Face) are now venture-scale too, led by Arena at about $3.1B.

### Cited Findings

**Scale AI**
- Meta bought a 49% non-voting stake for about $14.3B in June 2025, implying a valuation of about $29B. No new priced round had been reported as of 2026 [reported/est.] — [Wikipedia: Scale AI](https://en.wikipedia.org/wiki/Scale_AI); [Teahose](https://www.teahose.com/guides/scale-ai-valuation) (aggregator)
- Alexandr Wang left for Meta. Jason Droege became interim CEO. Reuters reported that Google, Scale's largest customer, planned to cut ties, and that Microsoft, OpenAI and xAI planned to step back (June 2025) [reported] — [CNBC, 2025-07-16](https://www.cnbc.com/2025/07/16/scale-ai-cuts-14percent-of-workforce-after-meta-investment-hiring-of-wang.html)
- July 2025: Scale cut 14% of staff (about 200 FTEs) and ended work with about 500 contractors, mostly in the data-labeling business. Droege said Scale had "ramped up our GenAI capacity too quickly" and consolidated 16 GenAI teams into 5 [disclosed memo, reported] — [TechCrunch, 2025-07-16](https://techcrunch.com/2025/07/16/scale-ai-lays-off-14-of-staff-largely-in-data-labeling-business/)
- Francis deSouza (ex-Google Cloud COO) was named CEO on 2026-07-30, effective 2026-08-10, with Wang as chairman. He is the third CEO since June 2025. This comes from search snippets of Wikipedia and secondary pages; I did not confirm it in a press release — [Wikipedia: Scale AI](https://en.wikipedia.org/wiki/Scale_AI); [Yahoo/Semafor item](https://finance.yahoo.com/sectors/technology/articles/scale-ceo-says-ai-related-162550402.html)
- May 2026: the Pentagon CDAO raised Scale's agreement ceiling from $100M to $500M. This is a ceiling, not booked revenue [disclosed] — [Scale blog](https://scale.com/blog/scale-ai-pentagon-cdao-500-million-agreement); [The Next Web](https://thenextweb.com/news/scale-ai-500m-pentagon-defense-contract-cdao)
- Research and evals arm: "Scale Labs" (labs.scale.com, formerly SEAL) hosts papers and leaderboards, including VisualToolBench and a "visual_language_understanding" leaderboard — [Scale Labs VTB paper page](https://labs.scale.com/papers/vtb); [Scale Labs VLU leaderboard](https://labs.scale.com/leaderboard/visual_language_understanding)
- Humanity's Last Exam: a benchmark built by Scale AI with the Center for AI Safety — [Wikipedia: Scale AI](https://en.wikipedia.org/wiki/Scale_AI)
- Scale's contributor platform Outlier pays "up to $1,350 per task" for tasks that take 6–7 hours (stated by Outlier, Oct 2026, as compiled by AITraining.jobs) — [AITraining.jobs pricing compilation](https://aitraining.jobs/research/what-ai-labs-pay-for-human-data)
- Physical AI: a 2026 vendor guide lists Scale programs such as a Universal Robots collaboration and demonstration capture. It describes first-person (egocentric) capture as one line among many, with a "Meta caveat" — [truelabel guide](https://truelabel.ai/blog/best-egocentric-video-data-providers) (vendor-authored)

**Surge AI**
- Bootstrapped; Edwin Chen owns about 75%. Reuters (July 2025) reported a first external raise of up to $1B at a valuation above $15B. Bloomberg later reported talks at $25B or more. **I found no confirmation that a round closed as of Oct 2026.** A June 2026 Dan Shipper post still says Surge passed $1B in revenue "without raising any outside capital" — [Wikipedia: Surge AI](https://en.wikipedia.org/wiki/Surge_AI); [Dan Shipper on X](https://x.com/danshipper/status/2069805581263847467); [Yahoo Finance](https://finance.yahoo.com/news/surge-ai-quietly-hit-1b-150057861.html)
- Revenue: about $1.2B in 2024 [reported, TechCrunch Sep 2025, via compilation; Sacra est.]. About $1.4B in 2025 is aggregator-only and unverified — [Sacra Surge](https://sacra.com/c/surge-ai/); [AITraining.jobs](https://aitraining.jobs/research/what-ai-labs-pay-for-human-data)
- Customers include OpenAI, Google, Anthropic and Meta [reported] — [AITraining.jobs compilation citing TechCrunch Sep 2025](https://aitraining.jobs/research/what-ai-labs-pay-for-human-data)
- Contrary Research reports that Surge charges "50% to ten times more than competitors" (Sep 2026) [reported] — [AITraining.jobs compilation](https://aitraining.jobs/research/what-ai-labs-pay-for-human-data); [Contrary Research](https://research.contrary.com/company/surge-ai). The Contrary page as surfaced by search also contained funding claims that conflict with the bootstrapped history, so treat it with caution.

**Mercor**
- Over $2B gross annualized revenue in June 2026, double the start of 2026. $500M run rate in Sep 2025, and $1M to $500M in 17 months per CEO Brendan Foody. Pays 60–70% of topline to contractors, so net is about $600–800M. Free-cash-flow profitable, with about $6M profit in H1 2025. Pays more than $2M a day to 30,000+ weekly active contractors [reported] — [Dealroom, 2026-07-06](https://dealroom.co/news/137121-mercor-doubles-to-2b-gross-revenue-run-rate-as-ai-labs-buy-expert-data/)
- Funding: $100M Series B at $2B (early 2025); $350M Series C at $10B (Oct 2025, led by Felicis with Benchmark, General Catalyst and Robinhood Ventures) [reported] — [Dealroom](https://dealroom.co/news/137121-mercor-doubles-to-2b-gross-revenue-run-rate-as-ai-labs-buy-expert-data/)
- July 2026: Bloomberg reported talks at about $20B, with Nvidia reportedly exploring a stake. **Conflict:** Tracxn and Public.com list $20B or a $500M Series D-1 (Aug 2026), while ValueAddVC (Oct 1, 2026) found no report that it closed — [TechCrunch, 2026-07-09](https://techcrunch.com/2026/07/09/mercor-is-in-talks-for-a-20b-valuation/); [Bloomberg, 2026-07-09](https://www.bloomberg.com/news/articles/2026-07-09/ai-training-startup-mercor-discusses-20-billion-valuation); [Yahoo/Nvidia](https://finance.yahoo.com/technology/ai/articles/nvidia-reportedly-explores-stake-ai-204528438.html); [ValueAddVC](https://valueaddvc.com/blog/how-does-mercor-make-money-2b-arr-20b-valuation-and-the-expert-data-marketplace-explained)
- Reported acquisitions of RL-environment startups: Deeptune (July 2026) and another environment startup (Feb 2026). The only source is a vendor directory, so this is unverified — [rl-list.com](https://rl-list.com/rl-environment-companies)
- APEX (AI Productivity Index): a benchmark family released Sep 2025. The original set has 400 tasks across investment banking, consulting, big law and primary care, graded by expert rubrics. Extensions are APEX-Agents (long-horizon, cross-application) and APEX-SWE. Mercor calls it "the industry standard" (Oct 1, 2026, self-claim) — [Mercor APEX](https://www.mercor.com/apex/); [APEX paper](https://www.alphaxiv.org/abs/2509.25721); [APEX-Agents arXiv](https://arxiv.org/pdf/2601.14242); [Mercor on X](https://x.com/mercor/status/2105782085894156549); [HF dataset](https://huggingface.co/datasets/mercor/APEX-v1-extended)

**Handshake AI** (AI-training unit of the college-jobs network Handshake, launched Jan 2025)
- Gross annualized AI-training revenue: $5–10M soon after launch, $550M in Jan 2026, and nearly $1B in Apr 2026. Net is about $300M. Total company gross ARR is about $1.15B including roughly $150M from recruiting [reported] — [Dealroom](https://dealroom.co/news/127345-handshakes-arr-crosses-1b-as-ai-training-revenue-surges/); [Sacra](https://sacra.com/c/handshake/) (est. ~$1.1B, Apr 2026); [Sacra research note](https://sacra.com/research/handshake-indeed-for-data-labelers/)
- Valuation: the last priced round was $3.5B in 2022. A Caplight secondary mark is about $2.7B (June 30, 2026) [est.]. Reportedly acquired Cleanlab in 2026 (aggregator only) — [Sacra](https://sacra.com/c/handshake/); [TrueUp](https://www.trueup.io/co/handshake); [Yespress](https://yespress.io/handshake)
- A KuCoin headline says the business is "approaching $10B" in revenue. This contradicts its own body text, which says about $1B, and is treated as an error — [KuCoin](https://www.kucoin.com/news/flash/handshake-ai-training-business-nears-10b-annual-revenue-in-2026)

**micro1**
- About $4B valuation in a Series B led by Dream Ventures (reported Sep 2026: $100M+). An Oct 12, 2026 company release confirmed the $4B valuation plus a roughly $25M employee tender. That is up from a $500M valuation in Sep 2025 (Series A, 01 Advisors) — [Dealroom](https://dealroom.co/news/155583-micro1-raises-100m-at-4b-valuation-for-ai-training-data/); [Forbes Australia](https://www.forbes.com.au/news/uncategorized/this-25-year-old-raised-us100-million-for-his-ai-startup-at-a-us4-billion-valuation/); [AI Weekly](https://aiweekly.co/alerts/micro1-raises-100m-at-4b-valuation-8x-jump-from-500m-mark-in-september-2025)
- Revenue: $500M gross run rate per TechCrunch (Sep 2026). Net is about $150–200M (Aug 2026). Other sources say about $200M ARR in early 2026, so the figures conflict. Benzinga reported $50M in Sep 2025 — [TechCrunch Snorkel article, 2026-09-22](https://techcrunch.com/2026/09/22/snorkel-ai-triples-valuation-to-3-5b-as-demand-for-ai-training-data-booms/); [Benzinga](https://www.benzinga.com/news/topics/25/09/47761555/scale-ai-rival-micro1-hits-50m-revenue-at-age-3-as-24-year-old-ceo-ali-ansari-lands-35m-from-twitters); [AITraining.jobs](https://aitraining.jobs/research/what-ai-labs-pay-for-human-data)
- Customers reportedly include Microsoft, Amazon, xAI and "leading frontier labs" (Spanish-language secondary report) — [ecosistemastartup](https://ecosistemastartup.com/micro1-con-25-anos-multiplica-x8-su-valoracion-a-us4-000m/)

**Snorkel AI**
- $350M Series E at $3.5B (TechCrunch, 2026-09-22), led by Insight Partners and S32, up from $1.3B about 17 months earlier. $375M annualized run rate, **18x in 12 months**. Expert payments are booked in COGS, not stripped from revenue. It pivoted "last year" from labeling software to "data-as-a-service" (completed datasets and RL environments), using a hybrid of synthetic data and subject-matter experts [disclosed/reported] — [TechCrunch, 2026-09-22](https://techcrunch.com/2026/09/22/snorkel-ai-triples-valuation-to-3-5b-as-demand-for-ai-training-data-booms/)

**Turing**
- $111M Series E at $2.2B (Mar 2025, led by Khazanah). No 2026 round found. It positions itself around "RL farms" and RL environments for agents (Oct 2026 CEO interview) — [SiliconANGLE](https://siliconangle.com/?p=688790); [Pulse2](https://pulse2.com/turing-111-million-series-e-raised-at-2-2-billion-valuation-for-agi-infrastructure); [Dealroom talk](https://dealroom.co/news/talk-nI1owceD-xg-inside-turings-rl-farms-for-the-agentic-age/)
- A blog cites "$300M reported revenue (Reuters)". I could not verify this against Reuters — [newmarketpitch (aggregator)](https://newmarketpitch.com/blogs/news/ai-training-data-top-startups-valuation)

**Invisible Technologies**
- $100M at a valuation above $2B (Sep 2025, led by Vanara Capital, a TPG spinout), with $144M raised in total. $134M revenue in 2024 [disclosed in its own release]. Sacra estimates about 11% EBITDA margin — [Bloomberg, 2025-09-16](https://www.bloomberg.com/news/articles/2025-09-16/scale-ai-rival-invisible-technologies-valued-at-over-2-billion); [SiliconANGLE](https://siliconangle.com/?p=711488); [AITraining.jobs](https://aitraining.jobs/research/what-ai-labs-pay-for-human-data)

**Labelbox (Alignerr expert network)**
- Reported Series E of $100M+ (equity plus credit) on 2026-10-14 at a $1.25B pre-money valuation, led by Infinitum Partners. This rests on a single aggregator and is unverified. It acquired Upcraft (Feb 2026) for Alignerr expert recruiting — [StockAnalysis](https://stockanalysis.com/private/labelbox/); [Yespress](https://yespress.io/labelbox)

**Toloka** (Nebius subsidiary)
- $72M led by Bezos Expeditions (May 7, 2025). Shopify CTO Mikhail Parakhin became executive chairman — [Nebius newsroom](https://nebius.com/newsroom/nebius-welcomes-bezos-expeditions-as-lead-investor-in-ai-data-business-toloka); [Business Wire](https://www.businesswire.com/news/home/20250507227499/en/Nebius-Welcomes-Bezos-Expeditions-as-Lead-Investor-in-AI-Data-Business-Toloka)

**Prolific**
- Platform fee of 42.8% for corporate customers and 33.3% for academic (stated) — [AITraining.jobs](https://aitraining.jobs/research/what-ai-labs-pay-for-human-data). No 2025–26 funding found.

**Arena (formerly LMArena; rebranded 2026-01-28)**
- Seed: $100M at $600M (May 2025). Series A: $150M at $1.7B post-money (Jan 2026, led by Felicis and UC Investments), with about $30M annualized consumption run rate for its paid AI Evaluations product (launched Sep 2025). Series B: $200M at $3.1B (announced Oct 8, 2026, co-led by Lightspeed and Khosla), with annualized revenue above $100M. Whether $3.1B is pre- or post-money is unclear — [PR Newswire (Series A)](https://www.prnewswire.com/news-releases/lmarena-raises-150-million-to-build-the-worlds-most-trusted-ai-evaluation-platform-302653012.html); [SiliconANGLE, 2026-01-06](https://siliconangle.com/2026/01/06/ai-evaluation-startup-lmarena-raises-150m-1-7b-valuation/); [Bloomberg Law (seed)](https://news.bloomberglaw.com/private-equity/lmarena-goes-from-academic-project-to-600-million-startup); [Angel Investors Network (Series B)](https://angelinvestorsnetwork.com/venture-capital/arena-raises-200m-in-series-b-funding-at-3-1b-valuation); [Wikipedia: Arena](https://en.wikipedia.org/wiki/Arena_(AI_platform)); [LMArena blog](https://lmarena.ai/blog)
- Business model: the free public leaderboard is the funnel. It sells paid evaluation services to labs and enterprises — [Wikipedia: LMArena](https://en.wikipedia.org/wiki/LMArena)

**Artificial Analysis**
- Seed from Nat Friedman and Daniel Gross (reported Jan 2026). Amounts conflict across aggregators. Revenue is not disclosed; StartupHub estimates about $8.1M [est., low confidence]. The model is enterprise subscriptions plus private benchmarking. It claims 500+ models, 100+ inference providers and 1,000+ endpoints benchmarked (Aug 2026) — [Latent Space wiki](https://latent.space/wiki/artificial-analysis-28355378/); [StartupHub](https://www.startuphub.ai/startups/artificial-analysis)

**Vals AI**
- $40M Series A led by a16z (Aug 2026, about $400M valuation reported). Positioned as "confidential benchmarking for frontier models". Revenue estimated near $1.3M (unverified) — [Pondero](https://pondero.ai/news/2026-09-20-vals-ai-series-a/); [Sacra Vals](https://sacra.com/c/vals-ai/)

**Patronus AI**
- $50M Series B led by Greenfield Partners (2026-06-25), $70M total. Claims revenue grew more than 15x year over year. Pivoted to "digital world models": simulated environments for training and stress-testing agents — [TechCrunch, 2026-06-25](https://techcrunch.com/2026/06/25/patronus-ai-lands-50m-to-build-digital-worlds-that-stress-test-ai-agents/)

**Epoch AI** (nonprofit, primarily Open Philanthropy funded)
- OpenAI commissioned and funded FrontierMath (300 problems, owned by OpenAI except a 50-problem holdout). This was disclosed late (Dec 2024 / Jan 2025), causing a credibility controversy. Contributing mathematicians were not systematically told — [Epoch clarification](https://epoch.ai/blog/openai-and-frontiermath); [TechCrunch, 2025-01-19](https://techcrunch.com/2025/01/19/ai-benchmarking-organization-criticized-for-waiting-to-disclose-funding-from-openai/)
- Epoch has also published RL-environment market pricing data (Jan 2026; see Q2) — [AITraining.jobs citing Epoch](https://aitraining.jobs/research/what-ai-labs-pay-for-human-data)

**METR** (501(c)(3), spun out of ARC Evals in Dec 2023)
- Part of a roughly $38M Audacious Project award (Oct 2024) to "Canary" with RAND. METR's share is reportedly about $17M (unverified). It reportedly does not accept lab payment for evaluations (secondary source) — [Wikiwand: METR](https://www.wikiwand.com/en/articles/METR); [Longterm wiki](https://www.longtermwiki.com/sourcing/funding-round/snRLwPkL2U)

**Kaggle Game Arena** (Google DeepMind + Kaggle)
- Launched Aug 2025 with chess. Werewolf and poker were added in early 2026. Open-source environments and harnesses. This is a lab-owned, free eval platform, not a vendor — [Google blog (launch)](https://blog.google/technology/ai/kaggle-game-arena/); [Google blog (updates)](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/kaggle-game-arena-updates/)

**Hugging Face**
- Launched "Community Evals" (Feb 2026, beta). Benchmark datasets host their own leaderboards and collect results from model repos automatically. Revenue is about $130M in 2024 [est., Latka] and roughly $150M in 2026 [reported, unverified] — [InfoQ, Feb 2026](https://www.infoq.com/news/2026/02/hugging-face-evals); [Latka](https://getlatka.com/companies/hugging-face)

**RL-environment natives**
- Mechanize was founded Apr 2025 by three former Epoch AI researchers to build coding-agent environments and evals. It estimates a minimum of about $500 per task, rising to "a few thousand" (Aug 2025) — [HeroHunt](https://www.herohunt.ai/blog/hiring-rl-environment-engineers-2026-labs-guide/); [AITraining.jobs](https://aitraining.jobs/research/what-ai-labs-pay-for-human-data)
- Vendor directories list many small RL-environment shops (e.g., Markov, hillclimb) whose lab relationships are self-reported — [rl-list.com](https://www.rl-list.com/); [Troveo landscape](https://www.troveo.ai/resources/rl-environment-companies)

### Inferences
- The "Big 4" expert-data vendors by gross run rate (Mercor > Surge ≈ Handshake > micro1) are now each at $0.5–2B+. Scale has fallen from leader to roughly flat at about $1B. Its neutrality problem with labs other than Meta is the defining event of the period.
- Eval companies now have real valuations (Arena $3.1B, Vals $400M) but are far smaller by revenue than data vendors (Arena about $100M run rate vs Mercor about $2B gross).

### Gaps
- "HSS" (Scale): I could not identify which Scale product or program this refers to. It needs clarification.
- Surge AI: no confirmed closed round or audited 2025 revenue. Mercor's $20B round is unconfirmed as closed.
- Handshake AI current valuation; Prolific and Turing 2026 financials; Artificial Analysis seed amount.
- deSouza's appointment as Scale CEO was seen only in secondary snippets.

---

## Q2. Revenue, valuations, customers, pricing models, contract sizes and margins

### Takeaway
Marketplace vendors bill by the expert-hour with roughly a 30–40% take (gross margin 27–33% at Mercor, about 30% at Handshake). Scale and Surge historically command higher margins and prices. RL environments are priced per task ($200–2,000) or per replica environment (about $300K), and frontier labs spend six to seven figures per quarter per vendor. Lab concentration is extreme. Meta committed at least $450M a year to Scale, Google spent about $150–200M a year with Scale before leaving, and Anthropic discussed spending over $1B on RL environments in a year.

### Cited Findings
**Headline numbers table (dates, type)**

| Company | Latest valuation | Latest revenue figure | Source |
|---|---|---|---|
| Scale AI | ~$29B implied (Jun 2025) [reported] | just under $1B in 2025, $870M in 2024; expects >$1B in 2026 [disclosed to Forbes] | [Wikipedia](https://en.wikipedia.org/wiki/Scale_AI) |
| Mercor | $10B (Oct 2025); $20B in talks (Jul 2026) | >$2B gross run rate (Jun 2026); net ~$0.6–0.8B | [Dealroom](https://dealroom.co/news/137121-mercor-doubles-to-2b-gross-revenue-run-rate-as-ai-labs-buy-expert-data/) |
| Surge AI | $15–25B+ in talks (2025), not closed | ~$1.2B in 2024 | [Wikipedia](https://en.wikipedia.org/wiki/Surge_AI) |
| Handshake AI | $3.5B (2022 priced); ~$2.7B secondary (Jun 2026) [est.] | ~$1B gross AI / ~$300M net (Apr 2026) | [Dealroom](https://dealroom.co/news/127345-handshakes-arr-crosses-1b-as-ai-training-revenue-surges/) |
| micro1 | $4B (Sep–Oct 2026) | ~$500M gross run rate (Sep 2026) | [TechCrunch](https://techcrunch.com/2026/09/22/snorkel-ai-triples-valuation-to-3-5b-as-demand-for-ai-training-data-booms/) |
| Snorkel AI | $3.5B (Sep 2026) | $375M run rate (net of expert COGS) | [TechCrunch](https://techcrunch.com/2026/09/22/snorkel-ai-triples-valuation-to-3-5b-as-demand-for-ai-training-data-booms/) |
| Arena | $3.1B (Oct 2026) | >$100M annualized | [Angel Investors Network](https://angelinvestorsnetwork.com/venture-capital/arena-raises-200m-in-series-b-funding-at-3-1b-valuation) |
| Turing | $2.2B (Mar 2025) | ~$300M (unverified) | [Pulse2](https://pulse2.com/turing-111-million-series-e-raised-at-2-2-billion-valuation-for-agi-infrastructure) |
| Invisible | >$2B (Sep 2025) | $134M in 2024 [disclosed] | [Bloomberg](https://www.bloomberg.com/news/articles/2025-09-16/scale-ai-rival-invisible-technologies-valued-at-over-2-billion) |
| Labelbox | $1.25B pre (Oct 2026, unverified) | n/a | [StockAnalysis](https://stockanalysis.com/private/labelbox/) |
| Vals AI | ~$400M (Aug 2026) | ~$1.3M [est.] | [Pondero](https://pondero.ai/news/2026-09-20-vals-ai-series-a/) |
| Patronus | n/d; $70M raised (Jun 2026) | >15x YoY growth | [TechCrunch](https://techcrunch.com/2026/06/25/patronus-ai-lands-50m-to-build-digital-worlds-that-stress-test-ai-agents/) |

**Lab spend and customer concentration** (all from the [AITraining.jobs compilation, 2026-10-30](https://aitraining.jobs/research/what-ai-labs-pay-for-human-data), which cites the original outlets)
- Meta to Scale: at least $450M a year for 5 years, tied to the 49% stake (Forbes, Jun 2025) [reported]
- Google to Scale: about $200M planned for 2025 before cutting ties; about $150M in 2024 (Reuters) [reported]
- Google to Appen: $82.8M in FY23, 30.3% of Appen revenue, ended Mar 2024 [disclosed, ASX]
- Anthropic: leaders discussed spending over $1B on RL environments over the following year (TechCrunch, Sep 2025) [reported]
- OpenAI: about $20K per website for UI-gym clones, with hundreds bought (SemiAnalysis, Jan 2026) [reported]. "Project Mercury" paid about $150/hr to ex-investment bankers for finance modelling (Bloomberg/Fortune, Oct 2025) [reported]
- About $500M a year of training data sold to Chinese labs (Forbes, Aug 2026) [reported]
- Before the Meta deal, Scale's top 5 customers (Google, OpenAI, Microsoft, xAI, Tesla) probably made up over 60% of revenue [analyst est.] — [Teahose](https://www.teahose.com/guides/scale-ai-competitors)

**Pricing models**
- *Per expert-hour (marketplaces):* Mercor's average contracted rate is about $93–110/hr (Oct 2026). Oncologists earn $350–450/hr and lawyers $100–150/hr. The buyer pays roughly $150–165/hr for a $109/hr expert [est., HeroHunt]. micro1 pays $60–70/hr on average and $170–180/hr for finance, medical and legal work, keeping about 30–40%. Handshake pays physicians $110–140/hr — [AITraining.jobs](https://aitraining.jobs/research/what-ai-labs-pay-for-human-data)
- *Per task:* Outlier/Scale pays up to $1,350 per 6–7 hour task. Text preference comparisons cost $0.10–0.50 each, and domain-expert annotation $1.50–8+ per task [est.]. Simple image annotation historically cost about 2¢ per image [est.] — [AITraining.jobs](https://aitraining.jobs/research/what-ai-labs-pay-for-human-data)
- *RL environments (Epoch AI, Jan 2026):* $200–2,000 per RL task, with rare tasks at $20K. A high-fidelity product replica (e.g., Slack) costs about $300K. Neolab contracts run $300–500K, and frontier labs spend six to seven figures per quarter. Exclusivity costs about 4–5x the non-exclusive price [reported] — [AITraining.jobs citing Epoch AI](https://aitraining.jobs/research/what-ai-labs-pay-for-human-data)
- *Off-the-shelf datasets:* micro1 reports 80–90% gross margins on these. micro1 also lists data-partnership tiers of $100K–500K, $500K–2M, and $2M+ for unique data — [AITraining.jobs](https://aitraining.jobs/research/what-ai-labs-pay-for-human-data)
- *Eval platforms:* Arena sells paid evaluation services; Artificial Analysis sells enterprise subscriptions and private benchmarking — [Wikipedia: LMArena](https://en.wikipedia.org/wiki/LMArena); [StartupHub](https://www.startuphub.ai/startups/artificial-analysis)
- *Government:* USAspending awards include about $814K (Tharros Defense) for frontier-model cyber evaluation and $332K (SecureBio) for biology benchmarks — [AITraining.jobs](https://aitraining.jobs/research/what-ai-labs-pay-for-human-data)

**Gross margins**
- Mercor: 27% in 2025, 33% in Q2 2026, with 46% expected next year [reported]. Handshake: about 30% [est., Sacra]. Scale: at least 50% [est., Sacra]. Appen: 40.3% in 2025 [disclosed]. Innodata DDS: 39% in FY2025 [disclosed] — [AITraining.jobs](https://aitraining.jobs/research/what-ai-labs-pay-for-human-data)
- Sacra notes these companies trade at about 30x annualized **net** revenue — [Sacra Handshake](https://sacra.com/c/handshake/)

### Inferences
- Margin structure favors **productized assets** (environments, off-the-shelf datasets, eval IP at 80–90%) over pass-through expert labor (about 30%). A small startup should sell reusable eval and environment IP rather than hours.
- Exclusivity at 4–5x is a key pricing lever for frontier-lab buyers.

### Gaps
- No public per-item price list for **vision** RLHF or vision eval items specifically. Typical eval-contract sizes for eval-only vendors (Vals, Artificial Analysis, Arena) are undisclosed.
- Headcount figures were not gathered (except Scale's 2025 layoff numbers and Mercor's contractor counts).

---

## Q3. What exactly do they deliver?

### Takeaway
The product mix has moved from bulk labels to (a) expert-written SFT and preference data with rubrics, (b) RL environments and verifiable tasks for agents (coding, computer use, enterprise app replicas), and (c) evals and benchmarks, often public "marketing" benchmarks plus private or held-out sets sold to labs. Multimodal and robotics data are a growing but separate segment dominated by collection and capture companies.

### Cited Findings
- Snorkel: completed datasets plus RL environments, with synthetic data generated alongside subject-matter experts — [TechCrunch](https://techcrunch.com/2026/09/22/snorkel-ai-triples-valuation-to-3-5b-as-demand-for-ai-training-data-booms/)
- Mercor: experts in fields such as physics and finance create specialized training data. It also publishes the APEX benchmarks with expert-authored rubrics — [Dealroom](https://dealroom.co/news/137121-mercor-doubles-to-2b-gross-revenue-run-rate-as-ai-labs-buy-expert-data/); [Mercor APEX methodology](https://www.mercor.com/apex/methodology/)
- Patronus: "digital world" simulated environments for agent training and evaluation — [TechCrunch](https://techcrunch.com/2026/06/25/patronus-ai-lands-50m-to-build-digital-worlds-that-stress-test-ai-agents/)
- Vals AI: confidential (private) benchmarking for frontier models — [Pondero](https://pondero.ai/news/2026-09-20-vals-ai-series-a/)
- Scale: public leaderboards and benchmarks via Scale Labs (e.g., VisualToolBench, Humanity's Last Exam), red-teaming and defense agents (CDAO) — [Scale Labs](https://labs.scale.com/papers/vtb); [Scale blog](https://scale.com/blog/scale-ai-pentagon-cdao-500-million-agreement)
- Epoch AI: commissioned private benchmark problems for OpenAI (FrontierMath), with a holdout set — [Epoch](https://epoch.ai/blog/openai-and-frontiermath)
- Arena: crowdsourced pairwise human-preference leaderboards plus paid private evaluations — [Wikipedia: Arena](https://en.wikipedia.org/wiki/Arena_(AI_platform))
- Turing: "RL farms" and environments for agentic models — [Dealroom talk](https://dealroom.co/news/talk-nI1owceD-xg-inside-turings-rl-farms-for-the-agentic-age/)
- Robotics/physical: Mecka AI builds its own hardware to capture egocentric data ($60M Series B led by Sequoia, reported ~Oct 2026). Human Archive (YC W26) sells synchronized egocentric datasets. Build AI released about 1M hours of factory egocentric video under a commercial license. Appen annotated egocentric video for a "frontier robotics lab" (50K+ units) — [Automate.org](https://www.automate.org/robotics/industry-insights/robotics-raises-icon-mecka-ai-multiply-labs-mmi-ultra/aph); [ChinaTalk](https://www.chinatalk.media/p/take-ten-thousand-egocentric-data); [Unidata](https://unidata.pro/blog/best-egocentric-data-providers-for-robotics/); [truelabel](https://truelabel.ai/blog/best-egocentric-video-data-providers)
- Egocentric and teleop data prices [est.]: egocentric video costs $15–40/hr, with annotation adding $8–25/hr. Teleoperation costs $28–60 per raw hour. Indian suppliers resell at $15–50/hr. Human Archive charges $1–10 per raw hour. Creator video licensing runs $1–4/min (Bloomberg, Jan 2025) — [AITraining.jobs](https://aitraining.jobs/research/what-ai-labs-pay-for-human-data)
- Robotics data buyers: Dyna Robotics grew its egocentric training set from about 1K to more than 1M hours. NVIDIA's EgoScale (Feb 2026) shifted collection from exoskeleton teleop toward camera footage. A commentary source says bulk footage now trades near zero — [CryptoBriefing](https://cryptobriefing.com/robotics-labs-motion-video-data-ai/); [ChinaTalk](https://www.chinatalk.media/p/take-ten-thousand-egocentric-data)
- Annotation-infrastructure layer (Encord, Labelbox) vs managed collection (Centific, Toloka) vs full stack (Scale) — [truelabel](https://truelabel.ai/blog/best-egocentric-video-data-providers)

### Inferences
- Vendors increasingly publish a **public benchmark as marketing** (APEX, VTB, HLE, Game Arena) while selling private data and evals behind it. This is the template a vision-eval startup would follow.
- Robotics and egocentric data is a commodity-raw-footage market moving toward specialized capture. It is mostly sold to robotics labs, not LLM labs.

### Gaps
- No primary detail on which vendors supply red-teaming to which lab. No confirmed frontier LLM lab (OpenAI, GDM, Anthropic) egocentric data contract was found.

---

## Q4. Trends 2025–2026

### Takeaway
(1) Labeling has shifted to expert data, and expert data is shifting to RL environments. (2) Revenue has exploded among marketplaces with thin net margins. (3) Neutrality now drives procurement: labs fled Scale after Meta's stake, and buyers now ask about strategic stakes. (4) Consolidation has begun (Mercor buying environment startups, Labelbox buying Upcraft, Handshake reportedly buying Cleanlab). (5) Labs are partly in-sourcing. (6) Evals are now a venture category in their own right, but credibility and conflict-of-interest issues persist.

### Cited Findings
- Scale's layoffs were explicitly in data labeling. It had "ramped up GenAI capacity too quickly" (Jul 2025) — [TechCrunch](https://techcrunch.com/2025/07/16/scale-ai-lays-off-14-of-staff-largely-in-data-labeling-business/)
- Snorkel moved from labeling software to data-as-a-service and RL environments. Its 18x growth is attributed to labs' "insatiable appetite for high-end training data" (Sep 2026) — [TechCrunch](https://techcrunch.com/2026/09/22/snorkel-ai-triples-valuation-to-3-5b-as-demand-for-ai-training-data-booms/)
- Neutrality: Google, Microsoft, OpenAI and xAI reportedly stepped back from Scale after Meta's investment — [CNBC](https://www.cnbc.com/2025/07/16/scale-ai-cuts-14percent-of-workforce-after-meta-investment-hiring-of-wang.html). Mercor having no frontier-lab strategic stake is described as "a live procurement question" — [HeroHunt](https://www.herohunt.ai/blog/hiring-ai-training-talent-labs-guide-2026/) (vendor blog)
- In-sourcing: the larger risk to Mercor is that labs build expert networks in-house — [Dealroom](https://dealroom.co/news/137121-mercor-doubles-to-2b-gross-revenue-run-rate-as-ai-labs-buy-expert-data/). OpenAI runs an in-house Human Data team with its own tooling. Anthropic hires to *orchestrate* vendors. SemiAnalysis is cited as saying labs are internalizing more (vendor blog, secondary) — [HeroHunt](https://www.herohunt.ai/blog/hiring-ai-training-talent-labs-guide-2026/)
- Consolidation: Mercor acquired Deeptune (Jul 2026) and another environment startup (Feb 2026) [unverified] — [rl-list](https://rl-list.com/rl-environment-companies). Labelbox acquired Upcraft (Feb 2026) — [Yespress](https://yespress.io/labelbox)
- Evals as a business: Arena grew from a $30M to a $100M+ run rate in Jan–Oct 2026 — [SiliconANGLE](https://siliconangle.com/2026/01/06/ai-evaluation-startup-lmarena-raises-150m-1-7b-valuation/); [Angel Investors Network](https://angelinvestorsnetwork.com/venture-capital/arena-raises-200m-in-series-b-funding-at-3-1b-valuation)
- Eval credibility: OpenAI's undisclosed funding of FrontierMath — [TechCrunch](https://techcrunch.com/2025/01/19/ai-benchmarking-organization-criticized-for-waiting-to-disclose-funding-from-openai/). HF Community Evals is pitched against "black-box leaderboards" — [InfoQ](https://www.infoq.com/news/2026/02/hugging-face-evals)
- Scale pivoting toward defense: the CDAO ceiling rose to $500M (May 2026) — [Scale blog](https://scale.com/blog/scale-ai-pentagon-cdao-500-million-agreement)
- Labs building free eval platforms themselves: Kaggle Game Arena (GDM) — [Google blog](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/kaggle-game-arena-updates/)

### Inferences
- Neutrality is now a selling point. A small independent vision-eval vendor with no lab equity can pitch itself as "neutral", but it must avoid FrontierMath-style undisclosed funding.
- In-sourcing pressure is strongest for commodity expert hours and weakest for reusable, hard-to-build assets: environments, private held-out evals, and specialized modalities.

### Gaps
- No primary SemiAnalysis or The Information text was accessed on lab in-sourcing magnitude.

---

## Q5. Multimodal / vision: who supplies vision evals and vision RL data, and how saturated is it?

### Takeaway
Vision is **under-served relative to text, code and agents** among the big vendors' public positioning. The most visible commercial vision eval is Scale's **VisualToolBench** (Oct 2025). The best model scored only about 18.7%, which shows the headroom. Arena runs vision and image arenas, but its paid business is general-purpose. Most other "multimodal" supply is either (a) generic annotation platforms (Labelbox, Encord, Appen) or (b) robotics and egocentric capture companies. I found **no dedicated, venture-scale "vision reasoning eval / vision RL environment" vendor**. That suggests the niche is not saturated, though big vendors could enter quickly.

### Cited Findings
- VisualToolBench (Scale + UIUC, Oct 2025): 1,204 open-ended tasks (603 single-turn, 601 multi-turn) across 5 domains, using a 6-tool API for "thinking with images". Scored with rubrics and an o4-mini judge at about 88% human alignment. The best model (GPT-5-think) scored 18.68% (18.44% on the leaderboard) — [Scale Labs paper](https://labs.scale.com/papers/vtb); [Scale leaderboard](https://scale.com/leaderboard/vtb); [Scale blog](https://scale.com/blog/visualtoolbench); [arXiv 2510.12712](https://alphaxiv.org/abs/2510.12712)
- Scale Labs also runs a "visual_language_understanding" leaderboard — [Scale Labs](https://labs.scale.com/leaderboard/visual_language_understanding)
- Academic and research context: a 2026 ACM survey notes the field is developing in-house benchmarks for the "thinking with images" paradigm that focus on the **faithfulness of the reasoning process**, not just final answers. Vision-R1 (ICLR 2026) cites a shortage of high-quality multimodal reasoning data and used only 10K multimodal math items — [ACM survey](https://dl.acm.org/doi/full/10.1145/3797161.3797177); [Vision-R1, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/680b2a8135b9c71278a09cafb605869e-Paper-Conference.pdf)
- Public vision benchmarks run by non-vendors: ImageCLEF 2026 MultimodalReasoning (exam-style vision QA). AIMultiple's June 2026 benchmark of 40 models on charts and visual logic — [ImageCLEF](https://mbzuai-nlp.github.io/ImageCLEF-MultimodalReasoning/2026/); [AIMultiple](https://aimultiple.com/visual-reasoning)
- Generic vendor reviews list Surge (RLHF), Scale (alignment and eval) and Mercor (expert data and eval) without vision specialization — [HeroHunt top-10](https://www.herohunt.ai/blog/top-10-human-data-providers-full-in-depth-review/) (vendor blog)
- Robotics and egocentric video is the crowded multimodal segment. Capture companies (Mecka AI, Human Archive, Build AI, Indian suppliers), annotation platforms (Encord, Labelbox, Appen, Scale), and robotics labs collecting their own data (Dyna, Genesis AI) all compete there. Prices for raw footage are falling toward zero — [ChinaTalk](https://www.chinatalk.media/p/take-ten-thousand-egocentric-data); [Automate.org](https://www.automate.org/robotics/industry-insights/robotics-raises-icon-mecka-ai-multiply-labs-mmi-ultra/aph); [TechCrunch Genesis AI](https://techcrunch.com/2026/05/06/khosla-backed-robotics-startup-genesis-ai-has-gone-full-stack-demo-shows/); [CryptoBriefing](https://cryptobriefing.com/robotics-labs-motion-video-data-ai/)
- Vision-adjacent prices [est.]: egocentric video at $15–40/hr plus $8–25/hr for annotation; Sama image moderation at about $0.56 per image (2022) — [AITraining.jobs](https://aitraining.jobs/research/what-ai-labs-pay-for-human-data)

### Inferences
- **Saturation assessment:** egocentric and robotics *data collection* is crowded and commoditizing. Vision *reasoning evals and vision RL environments* for frontier VLMs have one prominent incumbent asset (Scale VTB, from a vendor with neutrality baggage) plus academic benchmarks that saturate quickly. That looks open, especially for verifiable or procedural vision tasks that resist contamination and grade reasoning faithfulness.
- Big expert marketplaces (Mercor, Surge, micro1) could add vision quickly because they already sell rubric-graded expert tasks. The defensible wedge is likely contamination-resistant, procedurally generated or held-out vision items with verifiable answers, packaged as both an eval and an RL environment (RL-environment pricing is $200–2,000 per task, with exclusivity at 4–5x).

### Gaps
- No public data on which vendor supplies vision RLHF or vision RL environments to OpenAI, GDM or Anthropic. No vision-specific contract sizes or per-item vision eval prices.
- Did not verify whether Arena, Artificial Analysis or Vals sell vision-specific paid evals (Arena has public vision arenas, but I did not confirm a paid vision product in this pass).
- Did not survey smaller vision-eval startups (e.g., any YC companies focused on VLM evals). A targeted follow-up search is recommended.
