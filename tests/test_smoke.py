"""Smoke tests: does the pipeline hang together, not exhaustive correctness.
Run with: pytest tests/ -v
These assume main.py has already been run once so artifacts exist.
"""
import os
import pickle
import pytest

from resolvai.components.thread_reconstruction import clean_text
from resolvai.components.escalation_policy import EscalationPolicy
from resolvai.entity.config_entity import EscalationConfig


def test_clean_text_strips_mentions_and_urls():
    raw = "@AmazonHelp please help https://example.com/track thanks!!"
    cleaned = clean_text(raw)
    assert "@" not in cleaned
    assert "http" not in cleaned
    assert "thanks" in cleaned.lower()


def test_escalation_risk_keyword_forces_escalate():
    config = EscalationConfig(confidence_threshold=0.7, similarity_threshold=0.35, risk_keywords=["lawsuit", "fraud"])
    policy = EscalationPolicy(config)
    decision = policy.decide("I'm going to file a lawsuit over this", intent_confidence=0.95, retrieved=[{"similarity": 0.9}])
    assert decision["action"] == "escalate"


def test_escalation_low_confidence_escalates():
    config = EscalationConfig(confidence_threshold=0.7, similarity_threshold=0.35, risk_keywords=[])
    policy = EscalationPolicy(config)
    decision = policy.decide("hi", intent_confidence=0.3, retrieved=[{"similarity": 0.9}])
    assert decision["action"] == "escalate"


def test_escalation_auto_handle_when_confident_and_grounded():
    config = EscalationConfig(confidence_threshold=0.7, similarity_threshold=0.35, risk_keywords=[])
    policy = EscalationPolicy(config)
    decision = policy.decide("where's my order", intent_confidence=0.9, retrieved=[{"similarity": 0.8}])
    assert decision["action"] == "auto_handle"


@pytest.mark.skipif(not os.path.exists("models/logreg_clf.pkl"), reason="run main.py first to produce artifacts")
def test_classifier_artifact_loads():
    with open("models/logreg_clf.pkl", "rb") as f:
        clf = pickle.load(f)
    assert hasattr(clf, "predict")
