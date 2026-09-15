from resolvai import logger
from resolvai.config.configuration import ConfigurationManager
from resolvai.components.data_ingestion import DataIngestion

STAGE_NAME = "Data Ingestion"


class DataIngestionPipeline:
    def main(self):
        config = ConfigurationManager().get_data_ingestion_config()
        ingestion = DataIngestion(config)
        ingestion.profile_brands()   # logs the table -- check it before trusting chosen_brand
        ingestion.create_subsample()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        DataIngestionPipeline().main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
