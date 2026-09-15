# ResolvAI — Grounded AI Customer Support Agent

Classifies incoming customer messages by intent, drafts a reply grounded in how the
brand has historically resolved similar issues, and decides whether to auto-handle
or escalate to a human — for the Hiver SDE Intern take-home assignment.

## Quickstart (reproduce headline results in under 15 minutes)

```bash
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt

cp .env.example .env   # if not already present, fill in your key
# set GROQ_API_KEY or GEMINI_API_KEY in .env (both have free tiers)

# put the Kaggle "customer-support-on-twitter" CSV at data/raw/sample.csv, then:
python main.py
```

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

- Intent taxonomy is LLM-named from clusters — review `config/intents.yaml` by hand.
- Classifier comparison in stage 04 uses LLM-generated labels as reference for two
  of the three methods being compared, which is circular. The hand-labeled golden
  set is the only trustworthy number.
- Escalation confidence threshold and keyword list are a starting point, tuned by
  eye rather than optimized against the golden set — revisit after evaluation.
