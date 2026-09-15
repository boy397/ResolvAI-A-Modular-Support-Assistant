# Report — ResolvAI (AppleSupport)

Brand: **AppleSupport**. Dataset: full Kaggle "Customer Support on Twitter" (`twcs.csv`,
2,811,774 rows), filtered to AppleSupport threads → 104,951 cleaned customer/brand-reply
pairs (stages 1–2). LLM throughout: a local, on-device Qwen2.5-0.5B-Instruct (CPU,
`transformers`), not a hosted API — see the decision log for why.

## Problem framing

For a brand like AppleSupport on Twitter, "good" does **not** mean "resolves the issue
end-to-end in public" — almost none of these threads can be resolved without account-specific
verification (Apple ID, order number, device serial), and the historical brand replies in this
corpus reflect that: a large share are variations on "please DM us." So "good" for this system
means:

1. **Correctly triage** the incoming message into a small, honest set of intents.
2. **Draft a reply that is either directly useful (a real, well-known fix) or correctly defers**
   to a human/DM with a clear, specific reason — not a vague "we're sorry, please DM us" for
   everything, and not a confident-sounding fabricated fix for things it can't actually know.
3. **Escalate anything with a safety, legal, financial, or account-verification component**, and
   auto-handle only the subset of issues that have a well-documented, brand-consistent public
   answer (the iOS 11.1 "I" autocorrect bug being the canonical example in this dataset — it
   appears dozens of times with a known cause and fix).

**What I chose not to build**: multi-turn conversation state (each example is a single
customer message → single brand reply pair, not a full thread negotiation); hybrid/BM25
retrieval (dense-only, for time); an LLM-first escalation policy (rule-first instead, for
auditability); actually sending replies anywhere (this is an offline-evaluated agent, not a
live-deployed bot with write access to Twitter).

## Results vs. baselines

**Intent classification** (`data/processed/classifier_comparison.csv`, computed against
250 LLM-silver-labeled examples, *not* the golden set):

| Method | Accuracy | Macro F1 |
|---|---|---|
| Majority (trivial) | 0.90 | — |
| TF-IDF + LogReg (simple) | 0.86 | 0.31 |
| LLM few-shot (proposed) | 1.00 | 1.00 |

**Against the real, hand-labeled 150-example golden set** (`data/golden/golden_set.csv`,
`evaluation_results.json`), the picture is very different:

| Metric | Value |
|---|---|
| Intent accuracy (TF-IDF classifier) | **0.227** |
| Intent macro F1 | **0.046** |
| Escalation accuracy | **0.453** |
| False auto-handle rate | **0.513** |
| Reply quality (LLM-judge, 1–5): grounding | 3.72 |
| actionability | 3.72 |
| tone | 3.72 |
| safety | 3.71 |
| Golden ⋂ retrieval-index overlap | **150 / 150 (100%)** |

The gap between the two tables *is the finding* — see below.

## Failure analysis

**1. The intent classifier collapsed to one class.** The 250-example training sample used to
silver-label the classifier was ~90% `general_complaint` (hence the trivial-majority baseline
already scoring 0.90 there). The TF-IDF+LogReg classifier learned to just predict
`general_complaint` for every one of the 150 golden examples — 116/150 (77%) are wrong.
Example: *"Hello, Since updating to 11.0 I can't toggle Speaker/Receiver in Voice Memo. Works
fine on 10.3.3 on other devices."* → predicted `general_complaint`, true label
`ios_update_bug_report`. **Hypothesis**: the silver-labeling prompt for the local 0.5B model
under-specifies what counts as a "specific" complaint, so the model defaults to the vaguest
category whenever a message has any negative sentiment, and the class imbalance in the
resulting training sample makes the trained classifier chase the same default.

**2. Escalation misses are safety-relevant, not just accuracy noise.** 77/150 golden examples
were false auto-handles. Two concrete examples that should worry anyone evaluating this system:
- *"Hey can you give me a way to disable the iOS11 upgrade? It heats my phone up dangerously..."*
  — auto-handled. This describes a physical safety concern and was not escalated.
- *"Today I broke down crying in the store cause they reset my iPad w/out checking if it was
  backed up I lost many of my grandmas pics"* — auto-handled. This is a data-loss case needing
  account-specific recovery investigation, not a generic reply.

  **Hypothesis**: the escalation policy's `intent_confidence` threshold is fed directly by the
  broken classifier from failure mode #1 — but the classifier is *confidently* wrong (it's not
  hedging, it just always predicts the same class with high probability), so the confidence-based
  escalation trigger never fires for these cases. The rule-based risk-keyword check
  (`config/params.yaml` → `escalation.risk_keywords`) also doesn't cover "dangerously" or "lost...
  pics" — it's tuned for legal/financial language, not safety/emotional-harm language, and was
  never validated against real safety-adjacent examples like these before now.

**3. Retrieval index and golden set are not disjoint (100% overlap).** Every one of the 150
golden examples is also a row in `data/processed/cleaned_pairs.csv`, which is exactly what the
retrieval index (`artifacts/retrieval_index.pkl`) was built from. This means during evaluation
the agent can often retrieve the *literal historical reply to the near-identical question being
tested*, inflating grounding quality in a way that would not hold for a genuinely new customer
message. **This is the report's mandatory "misleading number" finding — see below.**

**4. The local LLM-judge doesn't discriminate quality.** Across 150 scored replies there are
only **6 unique (grounding, actionability, tone, safety) combinations**. Worse, it gave a
**perfect 5/5/5/5** to a reply that isn't a reply at all — for *"explain yourself"* the
generation step produced the literal string `New customer message: "explain yourself"` (an
echo of the prompt template, not a completion), and the judge scored it perfectly anyway.
One row's judge output didn't parse as JSON and silently became NaN (excluded from the mean,
not counted as a failure). **Hypothesis**: a 0.5B model is not a reliable judge for a 4-axis
rubric — it likely maps the whole reply to one holistic impression and repeats that score
across all four axes rather than actually evaluating grounding/actionability/tone/safety
independently. This is exactly why the assignment requires human/judge agreement calibration
(`data/golden/judge_calibration_TEMPLATE.csv`) — it has not been completed yet (see "next week").

**5. ~7% of generated replies are degenerate — the model breaks prompt format under load.**
11/150 replies contain leaked prompt-template artifacts (`"Here's what we can do... **Customer:**
... **Brand Reply:**..."` instead of a clean reply) or are literal echoes of the input. Example:
for *"I've had to turn all the notifications off on my phone otherwise it is turning off and on
every 5 seconds"* the model generated `"Sure, here's a suitable response based on the example
provided: --- **Customer:** I've had to turn all the notifications off..."` — restating the
few-shot example format instead of producing a reply. **Hypothesis**: greedy decoding on a
small instruct model sometimes falls into repeating the few-shot retrieval examples' formatting
verbatim rather than generalizing past it, especially when the retrieved examples are
structurally similar to the prompt template itself.

## What is misleading about my headline number?

If you only read "reply quality: ~3.7/5 across all axes, escalation accuracy 45%," here's what
that hides:

- **The 3.7/5 reply-quality score is inflated by retrieval leakage** (finding #3) — the agent
  can often retrieve the exact real historical answer to the exact question being scored. A
  genuinely held-out golden set (never seen in the retrieval index) would likely score lower.
- **The 3.7/5 score itself isn't trustworthy independent of leakage**, because the judge doing
  the scoring has only 6 distinct score patterns across 150 examples and rated a broken,
  non-answer reply a perfect 5/5/5/5 (finding #4). Without the human/judge agreement calibration
  step, there's no evidence this number tracks actual reply quality at all.
- **45% escalation accuracy sounds like "worse than a coin flip," but the real number that
  matters is the 51.3% false-auto-handle rate** — and it includes safety-relevant misses
  (finding #2), not just borderline judgment calls. A brand deploying this as-is would let
  through roughly 1 in 2 cases that should have gone to a human.
- **The "LLM few-shot: 100% accuracy" baseline number is circular and meaningless**, not just
  "a caveat" — it's the same local model grading itself with greedy (deterministic) decoding
  against its own earlier output on the same inputs. It was left in the comparison table
  on purpose, specifically so this report could point at it as an example of a number that
  looks great and proves nothing.
- **The golden set itself was labeled by an AI assistant (Claude), not independently by the
  candidate** — see `data/golden/SAMPLING_AND_LABELLING_METHODOLOGY.md`. It's a reasonable
  starting ground truth, not an audited one.

## What I'd do next with one more week

1. **Fix retrieval/golden leakage**: hold the golden set's source rows out of the retrieval
   index entirely (filter `cleaned_pairs.csv` before building `retrieval_index.pkl`), then
   re-run evaluation. This is the single highest-value fix — everything downstream of it
   (reply quality numbers) is currently suspect.
2. **Fix the classifier collapse**: re-silver-label with a stronger prompt (few-shot examples
   per intent, explicit "don't default to general_complaint" instruction) and check class balance
   in the training sample before training; consider hand-labeling the classifier's *training*
   set too, not just the eval set, if the imbalance persists.
3. **Complete the judge calibration step** (`judge_calibration_TEMPLATE.csv` → hand-score ~35
   examples → `judge_calibration_scored.csv`) to get an actual kappa/agreement number instead
   of just suspecting the judge is unreliable.
4. **Widen the escalation risk-keyword/rule set** to catch safety- and emotional-harm language
   ("dangerously," "lost," "crying," data-loss phrasing), not just legal/financial terms —
   directly motivated by the two safety-relevant false-auto-handle examples above.
5. **Add a reply-format validator**: reject/retry generations that contain leaked prompt
   artifacts (`"**Customer:**"`, `"New customer message:"`) before they ever reach a judge or
   a customer — cheap, mechanical, and would have caught all 11 degenerate replies found here.
6. Swap in a larger local model (e.g. a 3–4B instruct model) if the CPU time budget allows, and
   re-run the same evaluation to see how much of #1 and #4 is a model-capacity problem versus a
   prompt/pipeline problem.
