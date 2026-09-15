"""The serving-time agent: ties classification + retrieval + generation + escalation
together behind one call. This is what app.py imports -- keep this the single
source of truth for "what does a live request go through" so training-time
notebooks/stages and the deployed app never silently diverge."""
import pickle

from resolvai import logger
from resolvai.config.configuration import ConfigurationManager
from resolvai.components.retrieval_and_drafting import RetrievalDrafting
from resolvai.components.escalation_policy import EscalationPolicy


class SupportAgent:
    _instance = None

    def __init__(self):
        cm = ConfigurationManager()
        clf_config = cm.get_classification_config()
        retrieval_config = cm.get_retrieval_config()
        escalation_config = cm.get_escalation_config()

        with open(clf_config.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(clf_config.classifier_path, "rb") as f:
            self.classifier = pickle.load(f)

        self.retriever = RetrievalDrafting(retrieval_config)
        self.retriever.load_index()
        self.escalation = EscalationPolicy(escalation_config)
        logger.info("SupportAgent loaded and ready")

    @classmethod
    def instance(cls) -> "SupportAgent":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def handle(self, customer_text: str) -> dict:
        vec = self.vectorizer.transform([customer_text])
        intent = self.classifier.predict(vec)[0]
        # predict_proba gives a genuine confidence score -- prefer this over a placeholder
        intent_confidence = float(self.classifier.predict_proba(vec).max())

        retrieved = self.retriever.retrieve(customer_text)
        decision = self.escalation.decide(customer_text, intent_confidence, retrieved)
        reply = self.retriever.generate_reply(customer_text, retrieved)

        record = {
            "customer_text": customer_text,
            "intent": intent,
            "intent_confidence": intent_confidence,
            "retrieved": retrieved,
            "reply": reply,
            "decision": decision,
        }
        from resolvai.utils.common import append_jsonl
        append_jsonl(self.retriever.config.generation_log_path, record)
        return record
