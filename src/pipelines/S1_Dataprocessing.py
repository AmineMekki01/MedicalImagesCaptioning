from pathlib import Path
import yaml


from src.component.dataPreProcessing import dataUploadMerge, preprocessData
from src.config.configuration import ConfigurationManager
from src.logs import logger
import os 


# Now you can use these variables in your program

STAGE1 = 'Data Processing'

class  DataProcessingPipeline:
    def __init__(self):
        pass
    
    def run(self):
        config = ConfigurationManager()
        data_processor_config = config.get_data_processor_config()
        merged_data = dataUploadMerge(data_processor_config.projections_data_path, data_processor_config.reports_data_path)
        dataProcessed = preprocessData(merged_data, data_processor_config.images_base_path)
        dataProcessed.to_csv(data_processor_config.processed_data_output_path, index=False, sep=';')

   

if __name__ == '__main__':

    logger.info(f'Running pipeline: {STAGE1}')
    try:
        images_base_path = Path('./artifacts/data/raw/images')
        projections_path = Path('./artifacts/data/raw/caption/indiana_projections.csv')
        reports_path = Path('./artifacts/data/raw/caption/indiana_reports.csv')
        
        dataProcessingPipeline = DataProcessingPipeline(images_base_path, projections_path, reports_path)
        dataProcessingPipeline.run()
    except Exception as e:  
        logger.error(f'Error in {STAGE1} pipeline: {e}')
        raise e 
    logger.info(f'Completed pipeline: {STAGE1}')