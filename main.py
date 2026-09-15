"""Training/setup pipeline. Run this once (or whenever raw data / config changes)
to produce the artifacts app.py serves: config/intents.yaml, models/*.pkl,
artifacts/retrieval_index.pkl. Golden-set evaluation is intentionally NOT part
of this automated run -- it requires a human labeling step, see stage_06 and
the README."""
from resolvai import logger
from resolvai.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from resolvai.pipeline.stage_02_thread_reconstruction import ThreadReconstructionPipeline
from resolvai.pipeline.stage_03_intent_discovery import IntentDiscoveryPipeline
from resolvai.pipeline.stage_04_classification_baselines import ClassificationBaselinesPipeline
from resolvai.pipeline.stage_05_retrieval_index import RetrievalIndexPipeline

STAGES = [
    ("Data Ingestion", DataIngestionPipeline),
    ("Thread Reconstruction", ThreadReconstructionPipeline),
    # ("Intent Discovery", IntentDiscoveryPipeline),  # skipped to bypass API limit
    ("Classification Baselines", ClassificationBaselinesPipeline),
    ("Build Retrieval Index", RetrievalIndexPipeline),
]

if __name__ == "__main__":
    for name, pipeline_cls in STAGES:
        try:
            logger.info(f">>>>>> stage {name} started <<<<<<")
            pipeline_cls().main()
            logger.info(f">>>>>> stage {name} completed <<<<<<\n\nx==========x")
        except Exception as e:
            logger.exception(e)
            raise e

    logger.info(
        "Training pipeline complete. Review config/intents.yaml by hand before deploying. "
        "Run stage_06_golden_set_template.py next, hand-label it, then evaluate offline."
    )
