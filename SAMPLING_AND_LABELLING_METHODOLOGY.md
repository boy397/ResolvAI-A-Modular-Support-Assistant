# Sampling & Labelling Methodology

## Golden Set Generation (150 rows)
To generate the golden evaluation set for the `AmazonHelp` brand, I performed a **stratified sampling approach** rather than purely random sampling. 

1. **Rough Classification**: Before manually labelling, I ran the un-tuned TF-IDF + Logistic Regression baseline over the entire dataset to assign a "rough intent" to each customer message.
2. **Stratification**: I grouped the messages by these rough intents and sampled an equal number of rows from each group (totaling ~150 rows). This ensures that rare intents (like `return_request` or `technical_support`) are adequately represented in the evaluation set, preventing the majority classes (like `delivery_issue`) from overwhelming the metrics.
3. **Difficulty Injection**: I randomly marked 25% of the sampled rows as `ambiguous` and 15% as `hard` to explicitly test the escalation policy's boundary conditions.

## Labelling Strategy (Silver-Standard Fallback)
Due to time constraints and API rate-limiting exhaustion on the free tier, the golden set was transitioned to a **silver-standard auto-labelled set** using `auto_label.py`. 

- **Ground Truth Intent**: Derived directly from the `suggested_intent` cluster rather than manual human verification.
- **Ground Truth Escalation**: Assigned programmatically via regex heuristics targeting risk keywords (e.g., "lawyer", "sue", "scam") and specific high-risk intents (e.g., "complaint", "billing"). 
- **Judge Rubric**: The reply-quality checklist elements (Grounding, Actionability, Tone, Safety) were populated using mocked fallback values to simulate the grading behavior of an LLM-as-a-judge system when the Groq API crashed.
