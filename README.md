# ResolvAI — Grounded AI Customer Support Agent

Classifies incoming customer messages by intent, drafts a reply grounded in how the
brand has historically resolved similar issues, and decides whether to auto-handle
or escalate to a human — for the Hiver SDE Intern take-home assignment.

## Quickstart (reproduce headline results in under 15 minutes)

```bash
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Download the Kaggle "Customer Support on Twitter" dataset (thoughtvector/customer-support-on-twitter)
# and place twcs.csv at data/raw/twcs.csv. Only have the small demo sample.csv Kaggle ships alongside
# it? Point config/config.yaml -> data_ingestion.raw_data_path at data/raw/sample.csv instead -- the
# pipeline runs unchanged, just over far fewer brand threads.
python main.py
```

No API key needed. `params.yaml -> llm.provider` defaults to `"local"`, which runs a small
instruct model (Qwen2.5-0.5B-Instruct, ~1GB) on-device via `transformers` -- first run downloads
it once from Hugging Face and caches it; every call after that is free, has no rate limit, and
needs no network. This is what makes the free-tier-429 problem from Groq/Gemini go away entirely.
Switch `llm.provider` to `"groq"` or `"gemini"` (+ the matching key in `.env`) if you'd rather use
a hosted model and accept their free-tier limits.

`main.py` runs, in order: brand profiling + subsampling, thread reconstruction +
cleaning, intent discovery (LLM-named clusters, review `config/intents.yaml`
afterward), classifier baselines (majority / TF-IDF+LogReg / LLM few-shot,
comparison saved to `data/processed/classifier_comparison.csv`), and builds the
retrieval index.

## Golden-set evaluation (separate, manual step — cannot be automated)

```bash
python -m resolvai.pipeline.stage_06_golden_set_template
# hand-label data/golden/golden_set_TEMPLATE.csv, save as golden_set.csv
```

Then use `EvaluationJudge` (see `src/resolvai/components/evaluation_and_judge.py`)
to compute intent accuracy/F1, escalation accuracy + false-auto-handle rate,
LLM-judge scores, and human/judge agreement (Cohen's kappa). A short evaluation
script is the natural next addition here — wire it up against your filled-in
golden set once labeling is done.

## Running the demo app locally

```bash
python app.py
# open http://localhost:8080
```

## Running with Docker

```bash
docker compose up --build
```

## Deploying to Render

1. Push this repo to GitHub.
2. New "Web Service" on Render, pick this repo, environment: Docker.
3. Set env vars `GROQ_API_KEY` / `GEMINI_API_KEY` in Render's dashboard.
4. **Important:** `main.py` (training) is not run automatically on deploy — the
   repo must already include `config/intents.yaml`, `models/*.pkl`, and
   `artifacts/retrieval_index.pkl` (run `main.py` locally first and commit the
   small artifact files, or add a Render "pre-deploy command" that runs it).

## Project layout

```
src/resolvai/
  components/     # actual logic: ingestion, cleaning, intents, classifiers, retrieval, escalation, eval
  pipeline/        stage_XX_*.py wrap components for `main.py`; predict_pipeline.py is what app.py calls
  config/          ConfigurationManager: reads config.yaml + params.yaml into typed configs
  entity/          dataclasses describing each stage's config
  utils/           yaml/json/pickle helpers, swappable LLM client
config/config.yaml # paths; params.yaml holds hyperparameters/thresholds
app.py             # Flask serving app (this is what Render runs)
main.py            # training/setup pipeline (run once, or when data/config changes)
```

## Known limitations (see REPORT.md for the full analysis)

- Intent taxonomy started as LLM-named clusters and was hand-reviewed and rewritten
  (see `config/intents.yaml` and the decision log) — a small local model's cluster
  names were too noisy to trust as-is. `IntentDiscovery.discover_intents()` now
  refuses to regenerate `config/intents.yaml` if it already exists, specifically so
  `python main.py` can be re-run without clobbering that manual review.
- Classifier comparison in stage 04 uses LLM-generated labels as reference for two
  of the three methods being compared, which is circular. The hand-labeled golden
  set is the only trustworthy number.
- Escalation confidence threshold and keyword list are a starting point, tuned by
  eye rather than optimized against the golden set — revisit after evaluation.
- The local model (0.5B params) is small enough to run free on a CPU-only laptop,
  which is the whole point, but it's noticeably weaker than Groq/Gemini's hosted
  models at strict JSON output and nuanced reply drafting — expect more judge-score
  variance and occasional unparseable JSON (handled, but logged as a warning) than
  you'd get from a larger hosted model.
