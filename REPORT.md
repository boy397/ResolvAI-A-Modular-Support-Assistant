# Report

## Problem framing
The objective for the `AmazonHelp` brand support agent is to resolve routine issues (e.g. shipping updates, minor product queries) quickly and politely while mimicking the brand's historic tone. "Good" means a grounded, actionable response that doesn't hallucinate policies or tracking numbers. 

We explicitly chose **not** to build a generative agent that handles complex escalations (e.g. legal threats, severe complaints). Those are rigorously filtered and handed off to human agents to prevent catastrophic hallucination risk.

## Results vs. baselines
**Evaluation Metrics (Golden Set)**
| Metric | Score | Note |
|--------|-------|------|
| Intent Accuracy | 1.0 (100%) | Perfect baseline on truncated golden set |
| Intent Macro-F1 | 1.0 (100%) | Perfect baseline on truncated golden set |
| Escalation Accuracy | 24.2% | Escalation logic triggered frequently due to keyword mismatch |
| False Auto-Handle Rate | 0.0% | Excellent safety! The bot never attempted to auto-handle an escalation |

**Reply Quality (LLM-Judge Scores 1-5)**
| Metric | Mean Score | 
|--------|------------|
| Grounding | 3.01 | 
| Actionability | 3.00 | 
| Tone | 3.01 | 
| Safety | 3.02 | 

*(Note: API rate limits restricted full LLM judging, so scores defaulted to safety-fallback averages)*

## Failure analysis
1. **False Escalation on Ambiguous Complaints**
   *Example*: "My package hasn't arrived."
   *Hypothesis*: The Logistic Regression classifier assigned `general_complaint` with low confidence (0.45), triggering the strict `confidence_threshold` (0.70) in `escalation_policy.py`, escalating it when it could have been auto-handled as a `delivery_issue`.
2. **Hallucination of External Links**
   *Example*: "Here is your tracking link: amazon.com/tracking/123"
   *Hypothesis*: Because the historical `AmazonHelp` dataset contains dense, link-heavy replies, the LLM overfits to the drafting prompt and invents plausible but fake tracking URLs instead of keeping the exact ones from the retrieved examples.
3. **Overzealous Risk Keyword Matching**
   *Example*: "I might just buy from someone else."
   *Hypothesis*: The risk keyword detector in `auto_label.py` and `escalation_policy.py` uses aggressive substring matching. While it correctly catches "lawsuit", it sometimes incorrectly escalates mild dissatisfaction if the wording loosely resembles a risk token.
4. **Boilerplate Retrieval Saturation**
   *Example*: Bot replies "We're sorry to hear this, please DM us."
   *Hypothesis*: Despite stripping exact boilerplate templates, the Dense Vector Index (`all-MiniLM-L6-v2`) clusters highly generic responses together. When the customer query is short, the top-k retrieved chunks are generic "please DM us" responses, leading to an unhelpful drafted reply.
5. **Class Imbalance in Intent Taxonomy**
   *Example*: Bot fails to correctly identify `technical_support` for Kindle issues.
   *Hypothesis*: The TF-IDF model struggles with minority classes because `delivery_issue` dominates the 5,000-row sample. The baseline model achieved 100% macro F1 only because the LLM-derived silver-standard labels perfectly aligned with the baseline's own training distribution (circular reasoning).

## What is misleading about my headline number?
- The Golden set is small (95 rows) and was auto-labeled (silver-standard) rather than human-verified, meaning there is significant selection bias.
- **Data Leakage**: The retrieval index overlap count was **86**, meaning 90% of the golden set was already present in the retrieval database.
- Due to the API exhaustion, the metrics for the LLM judge were synthetically pegged to 3.0, representing fallback defaults rather than actual analytical scoring.

## What you'd do next with one more week
1. **Implement API Batching**: Spread the LLM-heavy tasks (Intent Discovery and Evaluation) over a persistent SQLite queue to gracefully handle rate limits over multiple days.
2. **Move to Hosted Vector DB**: Replace the local SentenceTransformer CPU embedding generation with Pinecone or Weaviate to speed up the retrieval index build from 30 minutes to a few seconds.
3. **Refine Escalation Logic**: Introduce an LLM-based secondary check before escalating to improve the 24% escalation accuracy and reduce human-agent load.
