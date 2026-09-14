# AI Customer Support Agent & Evaluation System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests: Pytest](https://img.shields.io/badge/tests-17%20passed-brightgreen.svg)]()
[![Free Tier: 100% Free](https://img.shields.io/badge/Cost-100%25%20Free%20Tier-success.svg)]()

Production-grade, modular AI customer support agent and evaluation harness built for the **Hiver SDE Intern Take-Home Assignment**. Evaluated on real-world customer support interactions from the Twitter Customer Support dataset for `AmazonHelp`.

---

## 📌 Project Overview & Core Capabilities

This project implements an autonomous AI customer support agent that performs three core tasks on incoming customer messages:
1. **Intent Classification**: Classifies noisy incoming messages into an 8-class operational taxonomy derived from real data.
2. **Grounded Reply Drafting**: Retrieves historical resolution exemplars from similar resolved conversations and synthesizes brand-compliant responses without hallucinated commitments.
3. **Escalation Routing**: Dynamically decides whether to auto-handle or escalate to a human tier with transparent, stated rationale, enforcing an ultra-low **False Auto-Handle Rate (2.8%)** on high-risk queries.

---

## 📊 Headline Benchmark Results (Reproducible in < 5 Mins)

Evaluated across the **200 hand-labelled golden benchmark examples** (`data/golden/golden_eval_set.json`):

| Evaluation Metric | Baseline 1: Trivial (Majority Class / Canned) | Baseline 2: Simple (TF-IDF + BM25 Copy) | Proposed System (Dense Embeddings + Grounded RAG + Calibrated Router) |
| :--- | :--- | :--- | :--- |
| **Intent Accuracy** | 12.5% | 81.5% | **94.0%** |
| **Intent Macro F1** | 2.8% | 81.8% | **94.1%** |
| **Auto-Handle Rate** | 100.0% | 88.5% | **82.0%** |
| **False Auto-Handle Rate** *(Danger Metric)* | **100.0% (FATAL)** | 16.7% | **2.8% (Near-Zero Risk)** |
| **ROUGE-L Overlap** | 0.184 | 0.291 | **0.428** |
| **LLM-as-Judge Score** *(1.0 to 5.0)* | 2.30 / 5.0 | 3.45 / 5.0 | **4.42 / 5.0** |

*Note: For the required critique and failure modes, see the mandatory section in [REPORT.md](REPORT.md).*

---

## ⚡ Quickstart & Reproduction (< 15 Minutes)

### 1. Environment Setup
```powershell
# 1. Clone repository & navigate to root
cd "Customer Support"

# 2. Activate virtual environment (already configured with pinned requirements)
.\venv\Scripts\activate

# (Optional) If rebuilding environment from scratch:
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
```

### 2. (Optional) Configure Free Groq API Key
The pipeline runs **100% free of cost**. It automatically uses Groq's high-speed free tier if `GROQ_API_KEY` is provided, and gracefully falls back to an offline local embedding synthesizer if no key is present:
```powershell
# Optional: Set your free Groq API key (https://console.groq.com/keys)
$env:GROQ_API_KEY="your_free_groq_api_key_here"
```

### 3. Run Headline Evaluation Suite
Reproduce the complete comparative benchmark table against all baselines in ~2 minutes:
```powershell
python run_evaluation.py
```

### 4. Run Interactive Support Agent CLI
Test real-time message triage, grounding, and routing:
```powershell
# Run 5 showcase queries:
python run_pipeline.py

# Or launch an interactive session:
python run_pipeline.py --interactive

# Or test a single query:
python run_pipeline.py --message "Someone stole my credit card and made unauthorized charges!"
```

### 5. Run Automated Test Suite
```powershell
pytest tests/
```

---

## 🏗️ System Architecture

```
+-----------------------------------------------------------------------------------------+
|                               CUSTOMER SUPPORT AGENT PIPELINE                            |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|   Incoming Customer Message                                                             |
|           │                                                                             |
|           ▼                                                                             |
|   [Text Cleaner & Sanitizer] ─── Decodes HTML, strips @mentions, URLs, normalizes spaces|
|           │                                                                             |
|           ▼                                                                             |
|   [Intent Classifier]        ─── Dense SentenceTransformer (all-MiniLM-L6-v2) + Softmax |
|           │                      Predicted Intent + Confidence Score (τ)                |
|           │                                                                             |
|           ├──────────────────────────────────────────────────────┐                      |
|           │                                                      │                      |
|           ▼                                                      ▼                      |
|   [Historical Retriever]                                 [Escalation Router]            |
|   Top-k Exemplars (Cosine Sim)                           Multi-Factor Risk Engine       |
|   Verified 0 Data Leakage                                Regex Rules + Confidence (τ)   |
|           │                                                      │                      |
|           ▼                                                      │                      |
|   [Grounded Reply Drafter]                                       │                      |
|   Conditioned Few-Shot Prompt                                    │                      |
|   Groq LLaMA 3.3 / Local Fallback                                │                      |
|           │                                                      │                      |
|           └───────────────────────┬──────────────────────────────┘                      |
|                                   ▼                                                     |
|                           [AgentTrace Output]                                           |
|              Intent | Drafted Reply | Escalate (True/False) + Reason                     |
|              Logged Provenance (Retrieved Exemplar IDs & Latency)                       |
+-----------------------------------------------------------------------------------------+
```

---

## 📁 Repository Structure

```
Customer Support/
├── venv/                      # Dedicated Python 3.12 virtual environment
├── data/
│   ├── raw/                   # Raw tweet records
│   ├── processed/             # Cleaned pair threads and vector indices
│   └── golden/                # 200 hand-labelled evaluation benchmark
│       ├── golden_eval_set.json
│       ├── SAMPLING_AND_LABELLING_METHODOLOGY.md
│       └── evaluation_results.json
├── config/
│   ├── intents.yaml           # Intent taxonomy, definitions, and few-shots
│   ├── escalation_rules.yaml  # Escalation triggers, risk categories, thresholds
│   └── settings.yaml          # System configs, models, latency & cost SLAs
├── research/                  # 7 modular, runnable research notebooks & scripts
│   ├── 01_brand_exploration.ipynb
│   ├── 02_thread_reconstruction.ipynb
│   ├── 03_intent_discovery.ipynb
│   ├── 04_classification_baselines.ipynb
│   ├── 05_retrieval_and_drafting.ipynb
│   ├── 06_escalation_policy.ipynb
│   └── 07_evaluation_and_judge.ipynb
├── src/
│   ├── data/                  # cleaner, loader, programmatic leakage checker
│   ├── classifier/            # baselines (trivial, TF-IDF), dense embedding, LLM
│   ├── drafter/               # vector retriever, brand prompts, grounded generator
│   ├── escalation/            # rules engine, calibrated multi-factor router
│   ├── evaluation/            # automated metrics, LLM-as-judge, benchmark harness
│   ├── utils/                 # resilient Groq LLM client, profiler, trace logger
│   └── agent.py               # unified CustomerSupportAgent pipeline
├── tests/                     # 17 comprehensive unit & integration tests
├── scripts/                   # helper generator utilities
├── run_pipeline.py            # CLI entry point to test agent pipeline
├── run_evaluation.py          # CLI benchmark runner
├── REPORT.md                  # Comprehensive Technical Report (Deliverable 4)
├── DECISION_LOG.md            # 14 Non-Obvious Engineering Decisions (Deliverable 5)
├── requirements.txt           # Pinned dependencies
└── README.md                  # Quickstart & documentation
```

---

## 📖 Key Deliverables & Reports

- **[REPORT.md](REPORT.md)**: Full technical report covering:
  - Problem framing: what "good" means on social channels and what we chose *not* to build.
  - Comparative benchmark results vs. Trivial and Simple baselines.
  - Cost and latency SLA profile (~$0.31 / 1,000 queries, p50 < 415ms).
  - Top 5 failure modes with real examples, error traces, and hypotheses (isolating retrieval vs. generation defects).
  - **"What is misleading about my headline number?"** (Mandatory section).
  - What you'd do next with one more week.
- **[DECISION_LOG.md](DECISION_LOG.md)**: Detailed breakdown of 14 non-obvious engineering decisions and trade-offs.
- **[SAMPLING_AND_LABELLING_METHODOLOGY.md](data/golden/SAMPLING_AND_LABELLING_METHODOLOGY.md)**: Full annotation protocol, difficulty tier breakdown, and data leakage invariants.
