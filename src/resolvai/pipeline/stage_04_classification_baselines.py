from resolvai import logger
from resolvai.config.configuration import ConfigurationManager
from resolvai.components.classification_baselines import ClassificationBaselines

STAGE_NAME = "Classification Baselines"


class ClassificationBaselinesPipeline:
    def main(self):
        config = ConfigurationManager().get_classification_config()
        ClassificationBaselines(config).run_benchmarks()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        ClassificationBaselinesPipeline().main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
