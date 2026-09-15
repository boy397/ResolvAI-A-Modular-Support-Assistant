# Decision Log

Built with AI coding assistance (Claude/Claude Code), disclosed per the assignment's rules.
The golden set was also labeled by Claude, not independently by the candidate — see
`data/golden/SAMPLING_AND_LABELLING_METHODOLOGY.md` for what that means and why it matters
before defending these numbers live.

1. **Brand: AppleSupport**, chosen from the brand-profiling table (`stage_01_data_ingestion`)
   over AmazonHelp/SpotifyCares/Uber_Support — AppleSupport had the highest real reply volume
   (106,860) with a valid threading rate of 99.87% and a non-trivial boilerplate rate (19.6%),
   meaning there's enough non-boilerplate substance to build real intents/retrieval from, and
   real thread-reconstruction integrity to trust the pairs.
2. **LLM provider: local (Qwen2.5-0.5B-Instruct via `transformers`, CPU), not Groq/Gemini** —
   the original build hit persistent free-tier 429s from Groq partway through classification
   (hundreds of sequential calls with no budget for that). Switching to a small on-device model
   removes rate limits entirely and costs nothing, at a real, measured cost in output quality
   (see REPORT.md failure modes #1 and #4 — this is the main tradeoff of this decision and it's
   visible directly in the results, not hypothetical).
3. **Escalation is rule-first, not LLM-first** — cheaper, faster, fully auditable (every decision
   traces to a specific threshold or keyword). Cost: the rule set (confidence threshold +
   similarity threshold + risk keywords) was tuned by eye before seeing golden-set results, and
   turned out to miss safety-relevant language ("dangerously," "lost...pics") entirely — see
   REPORT.md failure mode #2. Kept rule-first as the right call, but the specific rules need
   revision.
4. **Intent taxonomy was LLM-clustered, then manually rewritten** — the raw output of
   `IntentDiscovery` (K-means clusters named by the local 0.5B model) was genuinely low quality:
   duplicate/near-duplicate keys (`fix`, `fix_this`, `fixiOS11`), verbose non-snake_case names,
   and overly narrow single-example clusters (e.g. a cluster for one specific AssistiveTouch
   complaint). Replaced with 8 hand-written intents grounded in a fresh sample of real messages,
   informed by visibly dominant themes (this corpus is dominated by iOS 11 rollout fallout —
   battery drain and bugs — plus account access, hardware, connectivity, how-to, complaint, and
   positive-feedback buckets).
5. **`IntentDiscovery.discover_intents()` was made idempotent (skips regeneration if
   `config/intents.yaml` already exists)** — the original code re-ran clustering + LLM naming on
   every `main.py` invocation, which would silently destroy the manual review from decision #4
   the next time anyone re-ran the pipeline. This was a real reproducibility bug, not a
   hypothetical one — caught by actually re-running `main.py` twice during this build.
6. **Dataset: full `twcs.csv` (2.8M rows), not the small demo `sample.csv`** — loads in ~15s
   using pandas 3.0's default PyArrow-backed string columns (~770MB peak memory even for the
   full file with `usecols` trimmed to the 5 needed columns), well within a modest machine's
   budget, so there was no real reason to settle for the tiny demo sample. `sample.csv` stays
   in the repo as a documented fallback for anyone who can't spare the download/memory.
7. **`data/raw/twcs.csv` is gitignored, not committed** — 500MB+ exceeds what should go in a git
   repo, and Kaggle's terms don't clearly permit redistribution; `models/*.pkl`,
   `artifacts/retrieval_index.pkl`, and `config/intents.yaml` *are* explicitly un-ignored and
   committed instead, since the README's Render-deployment instructions require them and the
   original `.gitignore` excluded them by accident (a real bug — `models/*.pkl` was blanket
   -ignored despite the README telling you to commit exactly those files).
8. **No BM25/hybrid retrieval** — dense-only (sentence-transformers `all-MiniLM-L6-v2`) to fit
   the time budget; flagged as a "next week" item, not attempted.
9. **ROUGE/BLEU dropped from reply-quality metrics** — boilerplate historical replies ("please
   DM us...") score artificially well on lexical overlap regardless of whether a given reply is
   actually good; LLM-judge + (attempted) human agreement is the primary quality signal instead.
   Note: the judge itself turned out to have real reliability problems (REPORT.md #4) — this
   decision's tradeoff is now visible in the results, not just theoretical.
10. **Left the retrieval/golden-set leakage (100% overlap) and the circular "LLM few-shot: 100%
    accuracy" baseline in place rather than re-engineering around them** — a deliberate choice to
    ship real, honest, even-if-embarrassing numbers and write the "what's misleading about my
    headline number" section around them, rather than quietly fixing the pipeline and losing the
    evidence of the problem. Both are documented as the top item in "what I'd do next."
11. **Golden-set escalation criteria** (used for hand-labeling `ground_truth_escalate`): escalate
    on any safety/legal/fraud signal, any need for account/order/device-specific verification,
    any severe/unusual symptom beyond the well-documented bug patterns, or insufficient
    standalone context (many raw pairs are mid-thread replies like "Thanks" or "Same here"
    captured without the preceding thread); auto-handle only well-documented patterns with a
    standard historically-grounded fix. This is a defensible starting policy, not a definitively
    "correct" one — flagged in the methodology doc as something to be ready to defend or revise.
12. **Windows console/file logging forced to UTF-8**
    (`src/resolvai/__init__.py`) — the default `cp1252` codepage crashes the instant the logger
    prints a tweet containing an emoji, which is common in this dataset. Found by actually
    hitting the crash mid-run, not by inspection.
