import pickle
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from resolvai import logger
from resolvai.entity.config_entity import ClassificationConfig
from resolvai.utils.llm_client import LLMClient


class ClassificationBaselines:
    def __init__(self, config: ClassificationConfig):
        self.config = config
        with open(config.intents_config_path) as f:
            self.intents = yaml.safe_load(f)
        self.llm_call = LLMClient.get(config.llm_provider, config.llm_model)

    def classify_intent_llm(self, text: str) -> str:
        intent_list = "\n".join(f"- {k}: {v}" for k, v in self.intents.items())
        prompt = f"""Classify this customer support message into exactly one intent:
{intent_list}

Message: "{text}"
Respond with ONLY the intent key."""
        raw = self.llm_call(prompt, max_tokens=20).strip().lower().replace(" ", "_")
        return raw if raw in self.intents else "unmatched"

    def run_benchmarks(self) -> pd.DataFrame:
        pairs_df = pd.read_csv(self.config.cleaned_pairs_path)
        sample = pairs_df.sample(
            n=min(self.config.label_sample_size, len(pairs_df)), random_state=self.config.random_state
        ).reset_index(drop=True)

        logger.info(f"labeling {len(sample)} examples with the LLM (silver-standard labels)")
        sample["llm_intent"] = sample["customer_text_clean"].apply(self.classify_intent_llm)
        sample = sample[sample["llm_intent"] != "unmatched"].reset_index(drop=True)

        # Baseline 1: trivial
        majority_class = sample["llm_intent"].mode()[0]
        majority_acc = accuracy_score(sample["llm_intent"], [majority_class] * len(sample))

        # Baseline 2: TF-IDF + LogReg
        can_stratify = sample["llm_intent"].value_counts().min() > 1
        X_train, X_test, y_train, y_test = train_test_split(
            sample["customer_text_clean"], sample["llm_intent"],
            test_size=0.2, random_state=1, stratify=sample["llm_intent"] if can_stratify else None,
        )
        vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_vec, y_train)
        tfidf_preds = clf.predict(X_test_vec)
        tfidf_acc = accuracy_score(y_test, tfidf_preds)
        tfidf_f1 = f1_score(y_test, tfidf_preds, average="macro", zero_division=0)

        with open(self.config.vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)
        with open(self.config.classifier_path, "wb") as f:
            pickle.dump(clf, f)

        # Proposed: LLM few-shot, evaluated on the same held-out split
        test_preds = X_test.apply(self.classify_intent_llm)
        llm_acc = accuracy_score(y_test, test_preds)
        llm_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)

        comparison = pd.DataFrame([
            {"method": "Majority (trivial)", "accuracy": majority_acc, "macro_f1": None},
            {"method": "TF-IDF + LogReg (simple)", "accuracy": tfidf_acc, "macro_f1": tfidf_f1},
            {"method": "LLM few-shot (proposed)", "accuracy": llm_acc, "macro_f1": llm_f1},
        ])
        comparison.to_csv(self.config.comparison_output_path, index=False)
        logger.info(f"classifier comparison:\n{comparison.to_string(index=False)}")
        logger.warning(
            "NOTE: LLM labels used as ground truth here come from the same prompt family as the "
            "classifier being tested -- this is a sanity check, not a trustworthy absolute number. "
            "Use the hand-labeled golden set for that."
        )
        return comparison
