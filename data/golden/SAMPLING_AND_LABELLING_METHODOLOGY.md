# Sampling and Labelling Methodology

## Sampling Strategy

The dataset used is the TWCS (Customer Support on Twitter) dataset. To create a high-quality golden set for evaluating our customer support agent, we employed a stratified sampling approach. 

1. **Initial Filtering:** The data was filtered to isolate conversations involving the chosen brand (`AmazonHelp`) and reconstructed into conversational threads (customer query and brand reply pairs).
2. **Rough Intent Prediction:** We ran a TF-IDF + Logistic Regression classifier (trained in earlier stages) over all cleaned pairs to generate a "rough intent" prediction for each customer message.
3. **Stratification:** We stratified the dataset based on these rough intents to ensure diverse representation across all issue types (e.g., `general_complaint`, `billing_dispute`, `refund_request`).
4. **Difficulty Tagging:** A subset of the sampled messages was randomly tagged with varying difficulty levels (`clear`, `ambiguous`, `hard`) to test the agent's robustness across different query complexities.
5. **Sample Size:** 150 rows were sampled based on these criteria to form the initial template for the golden set.

## Labelling Approach (Silver-Standard)

> **Note:** The labelling for this project was done using an LLM-assisted "silver-standard" approach to simulate the required manual effort within a constrained timeline.

The following fields in the golden set were populated using heuristics and the Gemini 2.5 Flash model:

- **`ground_truth_intent`**: Inherited from the initial TF-IDF classifier's `suggested_intent`.
- **`ground_truth_escalate`**: Determined via a heuristic rule. It flags a conversation for escalation if risk keywords (e.g., "lawsuit", "scam") are present, or if the intent inherently requires human intervention (e.g., `billing_dispute`, `legal_action`).
- **`escalation_reason`**: A short rationale derived from the heuristics described above.
- **`good_reply_checklist`**: The Gemini 2.5 Flash model generated 2-3 essential elements that a successful customer support reply must address for each query.

## Known Limitations

- **Circular Labelling Risk:** Because the `ground_truth_intent` is derived from an earlier classifier, evaluating that same classifier against this golden set might artificially inflate its performance metrics.
- **Heuristic Escalation Constraints:** Using heuristics for `ground_truth_escalate` creates a simplified ground truth that might not fully capture nuanced situations where a human agent would choose to escalate.
- **Lack of Pure Human Review:** The "silver-standard" relies heavily on automated and LLM-assisted labelling. A truly robust evaluation requires manual verification by domain experts to refine intents and escalate policies accurately.
