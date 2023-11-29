from pathlib import Path

from src.logs import logger
from src.pipelines.S1_Dataprocessing import DataProcessingPipeline
from src.pipelines.S2_TrainingPipeline import ModelTrainingPipeline

STAGE_NAME1 = str("Data Ingestion")
STAGE_NAME2 = str("Model Training")

    
if __name__ == "__main__":
    try:
        logger.info(f">>>>> Stage {STAGE_NAME1} started <<<<<")
        data_processor = DataProcessingPipeline()
        data_processor.run()
        logger.info(f">>>>> Stage {STAGE_NAME1} completed. <<<<< \n")
        
    except Exception as e:
        logger.exception(e)
        raise e
    
    
    try:
        logger.info(f">>>>> Stage {STAGE_NAME2} started <<<<<")
        model_training = ModelTrainingPipeline()
        model_training.run()
        logger.info(f">>>>> Stage {STAGE_NAME2} completed. <<<<< \n")
    
    except Exception as e:  
        logger.exception(e)
        raise e