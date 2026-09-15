from dotenv import load_dotenv
load_dotenv()

from resolvai import logger
from resolvai.config.configuration import ConfigurationManager
from resolvai.components.evaluation_and_judge import EvaluationJudge

STAGE_NAME = "Golden Set Template Generation"


class GoldenSetTemplatePipeline:
    def main(self):
        config = ConfigurationManager().get_golden_set_config()
        EvaluationJudge(config).build_golden_template()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        GoldenSetTemplatePipeline().main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
        logger.info("HAND-LABEL data/golden/golden_set_TEMPLATE.csv, save as golden_set.csv, "
                    "then run evaluate_pipeline.py separately (see README).")
    except Exception as e:
        logger.exception(e)
        raise e
