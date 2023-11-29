from pathlib import Path

from src.logs import logger
from src.pipelines.S1_Dataprocessing import DataProcessingPipeline

STAGE_NAME1 = str("Data Ingestion")

    
if __name__ == "__main__":
    try:
        logger.info(f">>>>> Stage {STAGE_NAME1} started <<<<<")
        data_processor = DataProcessingPipeline()
        data_processor.run()
        logger.info(f">>>>> Stage {STAGE_NAME1} completed. <<<<< \n")
        
    except Exception as e:
        logger.exception(e)
        raise e
    