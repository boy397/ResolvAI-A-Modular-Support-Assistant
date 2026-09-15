import re

from resolvai.entity.config_entity import EscalationConfig


class EscalationPolicy:
    """Rule-first escalation. An LLM call is deliberately NOT used here for the
    common cases -- keyword/threshold rules are cheaper, faster, and auditable.
    Reserve an LLM fallback (not implemented here) only for residual ambiguous
    cases the rules below don't catch, if you have time budget left."""

    def __init__(self, config: EscalationConfig):
        self.config = config
        self.risk_pattern = re.compile(
            "|".join(re.escape(k) for k in config.risk_keywords), re.IGNORECASE
        )

    def decide(self, customer_text: str, intent_confidence: float, retrieved: list) -> dict:
        top_similarity = retrieved[0]["similarity"] if retrieved else 0.0

        if self.risk_pattern.search(customer_text):
            return {"action": "escalate", "reason": "risk keyword matched (legal/financial/safety)", "confidence": 1.0}

        if intent_confidence < self.config.confidence_threshold:
            return {
                "action": "escalate",
                "reason": f"low intent confidence ({intent_confidence:.2f})",
                "confidence": intent_confidence,
            }

        if top_similarity < self.config.similarity_threshold:
            return {
                "action": "escalate",
                "reason": f"no close historical precedent (similarity {top_similarity:.2f})",
                "confidence": top_similarity,
            }

        return {
            "action": "auto_handle",
            "reason": "matched known pattern with sufficient confidence",
            "confidence": intent_confidence,
        }
