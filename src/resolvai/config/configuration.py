from pathlib import Path

from resolvai.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from resolvai.utils.common import read_yaml, create_directories
from resolvai.entity.config_entity import (
    DataIngestionConfig,
    ThreadReconstructionConfig,
    IntentDiscoveryConfig,
    ClassificationConfig,
    RetrievalConfig,
    EscalationConfig,
    GoldenSetConfig,
)


class ConfigurationManager:
    def __init__(self, config_path: Path = CONFIG_FILE_PATH, params_path: Path = PARAMS_FILE_PATH):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        create_directories([self.config.artifacts_root, "data/processed", "data/golden", "models", "logs"])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        c = self.config.data_ingestion
        create_directories([Path(c.subsample_path).parent])
        return DataIngestionConfig(
            raw_data_path=Path(c.raw_data_path),
            subsample_path=Path(c.subsample_path),
            candidate_brands=list(c.candidate_brands),
            chosen_brand=c.chosen_brand,
        )

    def get_thread_reconstruction_config(self) -> ThreadReconstructionConfig:
        c = self.config.thread_reconstruction
        create_directories([Path(c.cleaned_pairs_path).parent])
        return ThreadReconstructionConfig(
            subsample_path=Path(c.subsample_path),
            cleaned_pairs_path=Path(c.cleaned_pairs_path),
        )

    def get_intent_discovery_config(self) -> IntentDiscoveryConfig:
        c = self.config.intent_discovery
        p = self.params.intent_discovery
        create_directories([Path(c.intents_config_path).parent])
        return IntentDiscoveryConfig(
            cleaned_pairs_path=Path(c.cleaned_pairs_path),
            intents_config_path=Path(c.intents_config_path),
            embedding_model=c.embedding_model,
            cluster_sample_size=p.cluster_sample_size,
            n_clusters=p.n_clusters,
            random_state=self.params.random_state,
            llm_provider=self.params.llm.provider,
            llm_model=self.params.llm.model,
        )

    def get_classification_config(self) -> ClassificationConfig:
        c = self.config.classification
        p = self.params.classification
        create_directories([Path(c.models_dir)])
        return ClassificationConfig(
            cleaned_pairs_path=Path(c.cleaned_pairs_path),
            intents_config_path=Path(c.intents_config_path),
            models_dir=Path(c.models_dir),
            vectorizer_path=Path(c.vectorizer_path),
            classifier_path=Path(c.classifier_path),
            comparison_output_path=Path(c.comparison_output_path),
            label_sample_size=p.label_sample_size,
            random_state=self.params.random_state,
            llm_provider=self.params.llm.provider,
            llm_model=self.params.llm.model,
        )

    def get_retrieval_config(self) -> RetrievalConfig:
        c = self.config.retrieval
        p = self.params.retrieval
        create_directories([Path(c.index_cache_path).parent, Path(c.generation_log_path).parent])
        return RetrievalConfig(
            cleaned_pairs_path=Path(c.cleaned_pairs_path),
            intents_config_path=Path(c.intents_config_path),
            index_cache_path=Path(c.index_cache_path),
            generation_log_path=Path(c.generation_log_path),
            embedding_model=c.embedding_model,
            top_k_retrieval=p.top_k_retrieval,
            llm_provider=self.params.llm.provider,
            llm_model=self.params.llm.model,
        )

    def get_escalation_config(self) -> EscalationConfig:
        p = self.params.escalation
        return EscalationConfig(
            confidence_threshold=p.confidence_threshold,
            similarity_threshold=p.similarity_threshold,
            risk_keywords=list(p.risk_keywords),
        )

    def get_golden_set_config(self) -> GoldenSetConfig:
        c = self.config.golden_set
        p = self.params.golden_set
        create_directories([Path(c.golden_dir)])
        return GoldenSetConfig(
            golden_dir=Path(c.golden_dir),
            golden_template_path=Path(c.golden_template_path),
            golden_set_path=Path(c.golden_set_path),
            judge_calibration_path=Path(c.judge_calibration_path),
            cleaned_pairs_path=Path(c.cleaned_pairs_path),
            vectorizer_path=Path(c.vectorizer_path),
            classifier_path=Path(c.classifier_path),
            golden_set_size=p.golden_set_size,
            judge_calibration_size=p.judge_calibration_size,
            random_state=self.params.random_state,
            llm_provider=self.params.llm.provider,
            llm_model=self.params.llm.model,
        )
