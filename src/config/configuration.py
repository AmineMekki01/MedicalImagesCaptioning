from src.constants import *
from src.utils.commonFunctions import read_yaml, create_directories

from src.entity.configEntity import DataProcessingConfig


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH):
        self.config = read_yaml(config_filepath)
    
    def get_data_processor_config(self) -> DataProcessingConfig:
        config = self.config.data_processing_config
        create_directories([Path(config.processed_data_path)])
        data_processor_config = DataProcessingConfig(
            processed_data_path = Path(config.processed_data_path),
            images_base_path = Path(config.images_base_path),
            projections_data_path = Path(config.projections_path),
            reports_data_path = Path(config.reports_path),
            processed_data_output_path = Path(config.processed_data_output_path)
        ) 
        return data_processor_config
    
