from pathlib import Path
import yaml


from src.component.data_processing.data_pre_processing import dataUploadMerge, preprocessData, convert_txt_to_csv
from src.config.configuration import ConfigurationManager
from src.logs import logger
from src.component.data_processing.get_data import download_and_extract
import os


# Now you can use these variables in your program

STAGE1 = 'Data Processing'


class DataProcessingPipeline:
    def __init__(self):
        pass

    def run(self):
        
        
        # DATA_FILE_ID = '1oVpVPyH66gN4kL4lQRXcghc-ELgNHdHz'
        # DATA_DEST_FOLDER = './artifacts/data'
        
        # try:
        #     logger.info(f'Downloading data from google drive')
        #     download_and_extract(DATA_FILE_ID, DATA_DEST_FOLDER)
        # except:  
        #     logger.error(f'Error in {STAGE1} pipeline while downloading the data: {e}')
            
        config = ConfigurationManager()
        ChestXray_data_processor_config = config.get_ChestXray_data_processor_config()
        merged_data = dataUploadMerge(
            ChestXray_data_processor_config.projections_data_path, ChestXray_data_processor_config.reports_data_path)
        dataProcessed = preprocessData(
            merged_data, ChestXray_data_processor_config.images_base_path, ChestXray_data_processor_config.processed_data_path)

        if not ChestXray_data_processor_config.processed_first:
            """
                This means the we want to fine tune the model on general data then we will fine tune it on specific data
            """
            ROCO_data_processor_config = config.get_ROCO_data_processing_config()
            convert_txt_to_csv(text_folder_path=ROCO_data_processor_config.raw_data_path, image_path_prefix=ROCO_data_processor_config.images_base_path,
                               output_path=ROCO_data_processor_config.processed_data_path)


if __name__ == '__main__':

    logger.info(f'Running pipeline: {STAGE1}')
    try:
        images_base_path = Path('./artifacts/data/raw/images')
        projections_path = Path(
            './artifacts/data/raw/caption/indiana_projections.csv')
        reports_path = Path('./artifacts/data/raw/caption/indiana_reports.csv')

        dataProcessingPipeline = DataProcessingPipeline(
            images_base_path, projections_path, reports_path)
        dataProcessingPipeline.run()
    except Exception as e:
        logger.error(f'Error in {STAGE1} pipeline: {e}')
        raise e
    logger.info(f'Completed pipeline: {STAGE1}')
