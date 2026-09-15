from resolvai import logger
from resolvai.config.configuration import ConfigurationManager
from resolvai.components.retrieval_and_drafting import RetrievalDrafting

STAGE_NAME = "Build Retrieval Index"


class RetrievalIndexPipeline:
    def main(self):
        config = ConfigurationManager().get_retrieval_config()
        RetrievalDrafting(config).build_index()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        RetrievalIndexPipeline().main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
