"""Runs the full evaluation harness against the hand-labeled golden set.
Requires: data/golden/golden_set.csv (you filled in the TEMPLATE by hand),
models/*.pkl and artifacts/retrieval_index.pkl (from `python main.py`).

Usage: python -m resolvai.pipeline.stage_07_evaluate
"""
import json
import pandas as pd

from resolvai import logger
from resolvai.config.configuration import ConfigurationManager
from resolvai.components.evaluation_and_judge import EvaluationJudge
from resolvai.components.retrieval_and_drafting import RetrievalDrafting
from resolvai.components.escalation_policy import EscalationPolicy

STAGE_NAME = "Evaluation"


class EvaluatePipeline:
    def main(self):
        cm = ConfigurationManager()
        golden_config = cm.get_golden_set_config()
        retrieval_config = cm.get_retrieval_config()
        escalation_config = cm.get_escalation_config()

        golden_df = pd.read_csv(golden_config.golden_set_path)
        assert golden_df["ground_truth_intent"].notna().all() and (golden_df["ground_truth_intent"] != "").all(), (
            "golden_set.csv has unlabeled rows -- finish hand-labeling before running evaluation"
        )

        judge = EvaluationJudge(golden_config)
        retriever = RetrievalDrafting(retrieval_config)
        retriever.load_index()
        escalation = EscalationPolicy(escalation_config)

        results = {}

        # --- Intent metrics ---
        results["intent"] = judge.evaluate_intent(golden_df)
        logger.info(f"Intent metrics: {results['intent']}")

        # --- Escalation metrics (incl. false auto-handle rate) ---
        def agent_decision_fn(text):
            vec = judge.vectorizer.transform([text])
            confidence = float(judge.classifier.predict_proba(vec).max())
            retrieved = retriever.retrieve(text)
            return escalation.decide(text, confidence, retrieved)["action"]

        results["escalation"] = judge.evaluate_escalation(golden_df, agent_decision_fn)
        logger.info(f"Escalation metrics: {results['escalation']}")

        # --- Generate replies + LLM-judge scores for the golden set ---
        judge_scores = []
        for _, row in golden_df.iterrows():
            retrieved = retriever.retrieve(row["customer_text"])
            reply = retriever.generate_reply(row["customer_text"], retrieved)
            score = judge.judge_reply(row["customer_text"], row.get("good_reply_checklist", ""), reply)
            score["customer_text"] = row["customer_text"]
            score["generated_reply"] = reply
            judge_scores.append(score)
        judge_df = pd.DataFrame(judge_scores)
        judge_df.to_csv(golden_config.golden_dir / "judge_scores.csv", index=False)

        numeric_axes = ["grounding", "actionability", "tone", "safety"]
        results["reply_quality_mean_scores"] = judge_df[numeric_axes].mean(numeric_only=True).to_dict()
        logger.info(f"Reply quality (LLM-judge means): {results['reply_quality_mean_scores']}")

        # --- Data leakage check (for the "misleading number" report section) ---
        pairs_df = pd.read_csv(golden_config.cleaned_pairs_path)
        results["golden_retrieval_overlap_count"] = judge.check_leakage(golden_df, pairs_df)
        logger.info(f"Golden/retrieval overlap (should be low): {results['golden_retrieval_overlap_count']}")

        # --- Human vs judge agreement (only if you've filled in judge_calibration_scored.csv) ---
        calib_path = golden_config.golden_dir / "judge_calibration_scored.csv"
        if calib_path.exists():
            scored = pd.read_csv(calib_path)
            results["human_judge_agreement"] = judge.human_judge_agreement(scored)
            logger.info(f"Human/judge agreement: {results['human_judge_agreement']}")
        else:
            logger.warning(
                f"{calib_path} not found -- hand-score data/golden/judge_calibration_TEMPLATE.csv, "
                f"save as judge_calibration_scored.csv, and re-run this stage to get agreement numbers."
            )

        with open(golden_config.golden_dir / "evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Full results written to {golden_config.golden_dir / 'evaluation_results.json'}")
        return results


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        EvaluatePipeline().main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
