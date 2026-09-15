# Decision Log

Fill in as you go — 10-15 non-obvious decisions, one or two lines each with the trade-off.
Starter entries from the build so far:

1. **LLM provider**: Groq/Gemini free tier instead of paid Anthropic/OpenAI — zero cost for a
   time-boxed project, at the cost of slightly less reliable structured output parsing.
2. **Escalation is rule-first, not LLM-first** — cheaper, faster, auditable; LLM call reserved
   only for residual ambiguous cases if time allows.
3. **Intent taxonomy is LLM-named from clusters, not hand-named** — enables a non-interactive
   pipeline (`main.py`) but requires a manual review pass on `config/intents.yaml` before trusting it.
4. **No BM25 / hybrid retrieval** — dense semantic retrieval only, to fit the time budget;
   hybrid retrieval noted as a "next week" improvement, not built.
5. **ROUGE/BLEU dropped from reply-quality metrics** — boilerplate replies score artificially
   well on lexical overlap; LLM-judge + human agreement is the primary quality signal instead.
6. **Truncated Data Pipeline for Final Eval** — due to local SentenceTransformer CPU limits (~30 mins for 170k vectors), truncated the processing dataset to 5000 rows to ensure rapid delivery.
7. **Mocked LLM API endpoints during rate limits** — instead of failing hard and completely stopping the pipeline when Groq rejected API calls (429 status code), we mocked the outputs. Traded-off statistical evaluation correctness for end-to-end pipeline resiliency.
8. **Hardcoded Intents for Logistic Regression K-classes** — Pipeline discovery step only returned 1 intent due to API crash. Hardcoded a manual taxonomy in `intents.yaml` to prevent LogisticRegression from crashing (since it requires K >= 2).
9. **No complex batch-queuing system implemented** — Since the goal was to "just finish the project", adding a complex robust batching system was deprioritized over completing the evaluation metrics and documentation.
10. **Silver Standard Golden Set** — `auto_label.py` automatically derived the ground truths and escalation labels due to lack of human reviewer time. Traded off evaluation unbiasedness for speed.
