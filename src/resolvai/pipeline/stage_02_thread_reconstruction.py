from resolvai import logger
from resolvai.config.configuration import ConfigurationManager
from resolvai.components.thread_reconstruction import ThreadReconstruction

STAGE_NAME = "Thread Reconstruction"


class ThreadReconstructionPipeline:
    def main(self):
        config = ConfigurationManager().get_thread_reconstruction_config()
        ThreadReconstruction(config).reconstruct_and_clean()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        ThreadReconstructionPipeline().main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
