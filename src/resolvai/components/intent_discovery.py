import json
import time
import yaml
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from resolvai import logger
from resolvai.entity.config_entity import IntentDiscoveryConfig
from resolvai.utils.llm_client import LLMClient

DEFAULT_INTENTS = {
    "delayed_delivery": "Package or order has not arrived or is late",
    "refund_request": "Customer wants money back for an order or charge",
    "billing_dispute": "Incorrect or unexpected charge",
    "account_access": "Cannot log in or access account",
    "app_technical_issue": "App crashes, bugs, or doesn't function",
    "product_quality": "Item received damaged, wrong, or defective",
    "general_complaint": "Venting or dissatisfaction without a specific actionable issue",
    "positive_feedback": "Compliment or thanks, no action needed",
}


class IntentDiscovery:
    """Clusters customer messages and asks the LLM to name each cluster, so the
    pipeline can run non-interactively. IMPORTANT: this is a starting point, not
    a substitute for human review -- read config/intents.yaml after running this
    and correct anything that looks off before relying on it downstream."""

    def __init__(self, config: IntentDiscoveryConfig):
        self.config = config

    def discover_intents(self) -> dict:
        from sentence_transformers import SentenceTransformer

        pairs_df = pd.read_csv(self.config.cleaned_pairs_path)
        sample = pairs_df.sample(
            n=min(self.config.cluster_sample_size, len(pairs_df)), random_state=self.config.random_state
        ).reset_index(drop=True)

        embed_model = SentenceTransformer(self.config.embedding_model)
        embeddings = embed_model.encode(sample["customer_text_clean"].tolist(), show_progress_bar=False)

        km = KMeans(n_clusters=self.config.n_clusters, random_state=self.config.random_state, n_init=10)
        sample["cluster"] = km.fit_predict(embeddings)

        llm_call = LLMClient.get(self.config.llm_provider, self.config.llm_model)
        intents = {}
        for c in sorted(sample["cluster"].unique()):
            examples = sample[sample["cluster"] == c]["customer_text_clean"].sample(
                min(6, (sample["cluster"] == c).sum()), random_state=self.config.random_state
            ).tolist()
            intents.update(self._name_cluster(llm_call, examples))
            time.sleep(2.5)  # rate-limit guard

        if not intents:
            logger.warning("LLM cluster naming failed; falling back to default intent taxonomy")
            intents = DEFAULT_INTENTS

        with open(self.config.intents_config_path, "w") as f:
            yaml.dump(intents, f)
        logger.info(f"wrote {len(intents)} intents to {self.config.intents_config_path} "
                    f"-- REVIEW THESE BY HAND before trusting them downstream")
        return intents

    @staticmethod
    def _name_cluster(llm_call, examples: list) -> dict:
        prompt = f"""Here are customer support messages that were grouped together as similar:
{chr(10).join('- ' + e for e in examples)}

Suggest one short snake_case intent key (2-3 words) and a one-sentence definition for this group.
Respond as JSON only: {{"key": "...", "definition": "..."}}"""
        try:
            raw = llm_call(prompt, max_tokens=80)
            raw = raw.strip().strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()
            parsed = json.loads(raw)
            return {parsed["key"]: parsed["definition"]}
        except Exception as e:
            logger.warning(f"cluster naming failed for one cluster: {e}")
            return {}
