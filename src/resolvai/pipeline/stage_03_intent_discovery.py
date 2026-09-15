from resolvai import logger
from resolvai.config.configuration import ConfigurationManager
from resolvai.components.intent_discovery import IntentDiscovery

STAGE_NAME = "Intent Discovery"


class IntentDiscoveryPipeline:
    def main(self):
        config = ConfigurationManager().get_intent_discovery_config()
        IntentDiscovery(config).discover_intents()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        IntentDiscoveryPipeline().main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
