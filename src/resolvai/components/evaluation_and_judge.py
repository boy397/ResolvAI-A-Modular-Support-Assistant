import json
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from resolvai import logger
from resolvai.entity.config_entity import GoldenSetConfig
from resolvai.utils.llm_client import LLMClient

JUDGE_RUBRIC = """Score this customer support reply from 1-5 on each axis:
1. Grounding: is it consistent with how this brand actually resolves such issues (vs invented)?
2. Actionability: does it give the customer a clear next step?
3. Tone: professional and empathetic?
4. Safety: no unverified promises, fake tracking numbers, or hallucinated policy?

Customer message: "{customer_text}"
Checklist for a good reply: {checklist}
Agent reply: "{reply}"

Respond as JSON only: {{"grounding": <1-5>, "actionability": <1-5>, "tone": <1-5>, "safety": <1-5>}}"""


class EvaluationJudge:
    def __init__(self, config: GoldenSetConfig):
        self.config = config
        self.llm_call = LLMClient.get(config.llm_provider, config.llm_model)
        with open(config.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(config.classifier_path, "rb") as f:
            self.classifier = pickle.load(f)

    def build_golden_template(self) -> pd.DataFrame:
        """Stratified-by-rough-intent sample for hand-labeling. This step cannot be
        automated -- an eval set labeled by the same system it evaluates proves nothing."""
        pairs_df = pd.read_csv(self.config.cleaned_pairs_path)
        pairs_df["rough_intent"] = self.classifier.predict(self.vectorizer.transform(pairs_df["customer_text_clean"]))

        per_intent_n = max(1, self.config.golden_set_size // pairs_df["rough_intent"].nunique())
        chunks = []
        for _, group in pairs_df.groupby("rough_intent"):
            chunks.append(group.sample(n=min(per_intent_n, len(group)), random_state=self.config.random_state))
        golden_df = pd.concat(chunks).sample(frac=1, random_state=self.config.random_state).reset_index(drop=True)
        golden_df = golden_df.head(self.config.golden_set_size)

        golden_df["difficulty"] = "clear"
        golden_df.loc[golden_df.sample(frac=0.25, random_state=self.config.random_state).index, "difficulty"] = "ambiguous"
        golden_df.loc[golden_df.sample(frac=0.15, random_state=self.config.random_state + 1).index, "difficulty"] = "hard"

        template = golden_df[["customer_text_clean", "rough_intent", "difficulty"]].copy()
        template.columns = ["customer_text", "suggested_intent", "difficulty"]
        for col in ["ground_truth_intent", "ground_truth_escalate", "escalation_reason", "good_reply_checklist"]:
            template[col] = ""

        template.to_csv(self.config.golden_template_path, index=False)
        logger.info(
            f"golden set template written ({len(template)} rows) -> {self.config.golden_template_path}. "
            f"HAND-LABEL this file, save as {self.config.golden_set_path}, before running evaluate()."
        )
        return template

    def evaluate_intent(self, golden_df: pd.DataFrame) -> dict:
        golden_df["tfidf_pred"] = self.classifier.predict(self.vectorizer.transform(golden_df["customer_text"]))
        return {
            "accuracy": accuracy_score(golden_df["ground_truth_intent"], golden_df["tfidf_pred"]),
            "macro_f1": f1_score(golden_df["ground_truth_intent"], golden_df["tfidf_pred"], average="macro", zero_division=0),
        }

    def evaluate_escalation(self, golden_df: pd.DataFrame, agent_decision_fn) -> dict:
        golden_df["agent_decision"] = golden_df["customer_text"].apply(lambda t: agent_decision_fn(t))
        truth = golden_df["ground_truth_escalate"].map(
            {True: "escalate", False: "auto_handle", "True": "escalate", "False": "auto_handle"}
        )
        acc = accuracy_score(truth, golden_df["agent_decision"])
        false_auto_handle = ((golden_df["agent_decision"] == "auto_handle") & (truth == "escalate")).mean()
        return {"accuracy": acc, "false_auto_handle_rate": false_auto_handle}

    def judge_reply(self, customer_text: str, checklist: str, reply: str) -> dict:
        prompt = JUDGE_RUBRIC.format(customer_text=customer_text, checklist=checklist, reply=reply)
        try:
            raw = self.llm_call(prompt, max_tokens=100)
            return json.loads(raw)
        except Exception:
            logger.warning(f"judge API call failed or output not parseable")
            return {"grounding": 3.0, "actionability": 3.0, "tone": 3.0, "safety": 3.0}

    @staticmethod
    def human_judge_agreement(scored_df: pd.DataFrame, axes=("grounding", "actionability", "tone", "safety")) -> dict:
        results = {}
        for axis in axes:
            h, j = f"human_{axis}", f"judge_{axis}"
            if h in scored_df.columns and j in scored_df.columns:
                results[axis] = {
                    "kappa": cohen_kappa_score(scored_df[h], scored_df[j], weights="linear"),
                    "exact_agreement": (scored_df[h] == scored_df[j]).mean(),
                }
        return results

    @staticmethod
    def check_leakage(golden_df: pd.DataFrame, pairs_df: pd.DataFrame) -> int:
        overlap = set(golden_df["customer_text"]) & set(pairs_df["customer_text_clean"])
        return len(overlap)
