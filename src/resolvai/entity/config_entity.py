from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DataIngestionConfig:
    raw_data_path: Path
    subsample_path: Path
    candidate_brands: List[str]
    chosen_brand: str


@dataclass(frozen=True)
class ThreadReconstructionConfig:
    subsample_path: Path
    cleaned_pairs_path: Path


@dataclass(frozen=True)
class IntentDiscoveryConfig:
    cleaned_pairs_path: Path
    intents_config_path: Path
    embedding_model: str
    cluster_sample_size: int
    n_clusters: int
    random_state: int
    llm_provider: str
    llm_model: str


@dataclass(frozen=True)
class ClassificationConfig:
    cleaned_pairs_path: Path
    intents_config_path: Path
    models_dir: Path
    vectorizer_path: Path
    classifier_path: Path
    comparison_output_path: Path
    label_sample_size: int
    random_state: int
    llm_provider: str
    llm_model: str


@dataclass(frozen=True)
class RetrievalConfig:
    cleaned_pairs_path: Path
    intents_config_path: Path
    index_cache_path: Path
    generation_log_path: Path
    embedding_model: str
    top_k_retrieval: int
    llm_provider: str
    llm_model: str


@dataclass(frozen=True)
class EscalationConfig:
    confidence_threshold: float
    similarity_threshold: float
    risk_keywords: List[str]


@dataclass(frozen=True)
class GoldenSetConfig:
    golden_dir: Path
    golden_template_path: Path
    golden_set_path: Path
    judge_calibration_path: Path
    cleaned_pairs_path: Path
    vectorizer_path: Path
    classifier_path: Path
    golden_set_size: int
    judge_calibration_size: int
    random_state: int
    llm_provider: str
    llm_model: str
