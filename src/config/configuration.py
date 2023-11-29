from src.constants import *
from src.utils.commonFunctions import read_yaml, create_directories

from src.entity.configEntity import DataProcessingConfig, TrainingConfig, ModelConfig


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
    
    def get_training_config(self) -> TrainingConfig:
        config = self.config.train_config
        create_directories([Path(config.trained_model_output_path), Path(config.metrics_folder_path)])
        training_config = TrainingConfig(
            training_data_path = Path(config.training_data_path),
            trained_model_output_path = Path(config.trained_model_output_path),
            metrics_folder_path = Path(config.metrics_folder_path),
            metrics_path = Path(config.metrics_path),
            batch_size = config.batch_size,
            epochs = config.epochs,
            learning_rate = config.learning_rate,
            freeze_epochs_gpt = config.freeze_epochs_gpt,
            freeze_epochs_all = config.freeze_epochs_all,
            device = config.device
        )
        return training_config
    
    def get_model_config(self) -> ModelConfig:
        config = self.config.model_config
        model_config = ModelConfig(
            vocab_size= config.vocab_size,
            embed_dim= config.embed_dim,
            num_heads= config.num_heads,
            seq_len= config.seq_len,
            depth= config.depth,
            attention_dropout= config.attention_dropout,
            residual_dropout= config.residual_dropout,
            mlp_ratio= config.mlp_ratio,
            mlp_dropout= config.mlp_dropout,
            emb_dropout= config.emb_dropout
        )
        return model_config