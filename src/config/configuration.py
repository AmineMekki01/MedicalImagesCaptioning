"""
Script : 
    configuration.py

Description :
    The 'configuration.py' module in this project serves as a central point for managing and setting up configurations for various components of the system. It leverages a combination of custom configuration entities and utility functions to streamline the process of reading, processing, and setting up configurations for different aspects of the project, including data processing, training, model parameters, and inference. This module is critical for ensuring that all parts of the system can access consistent and well-defined configuration parameters.

Classes:
    ConfigurationManager: A class that provides methods to retrieve various configuration entities based on a YAML configuration file.

Methods:
    __init__(self, config_filepath=CONFIG_FILE_PATH): Initializes the ConfigurationManager with a path to the YAML configuration file.

    get_ChestXray_data_processor_config(self) -> ChestXrayDataProcessingConfig: Retrieves and returns the configuration for processing Chest X-ray data, including paths and processing flags.

    get_ROCO_data_processing_config(self) -> ROCODataProcessingConfig: Retrieves and returns the configuration for processing ROCO dataset, including raw and processed data paths and image base path.

    get_training_config(self) -> [TrainingConfig, ChestXrayTrainingConfig, ROCOTrainingConfig]: Retrieves and returns the training configuration which includes general training parameters as well as specific configurations for Chest X-ray and ROCO datasets.

    get_model_config(self) -> ModelConfig: Retrieves and returns the configuration for the model, including parameters like vocabulary size, embedding dimensions, and attention mechanisms.

    get_inference_config(self) -> InferenceConfig: Retrieves and returns the configuration for the inference process, including paths for the trained model, inference data, and metrics.

Each of these methods utilizes a combination of reading from a YAML configuration file and creating necessary directories to facilitate smooth execution of data processing, training, and inference workflows in the project.

Dependencies:
    - src.constants: Module containing constants used across the project.
    - src.utils.commonFunctions: Module providing utility functions like reading YAML files and creating directories.
    - src.entity.configEntity: Module containing various configuration entities for data processing, training, model, and inference.

"""
from pathlib import Path
from typing import Tuple, Union, Dict, Any

from src.constants import CONFIG_FILE_PATH
from src.utils.common_functions import read_yaml, create_directories
from src.entity.config_entity import (
    ChestXrayDataProcessingConfig,
    ROCODataProcessingConfig,
    TrainingConfig,
    ChestXrayTrainingConfig,
    ROCOTrainingConfig,
    ModelConfig,
    InferenceConfig
)


class ConfigurationManager:
    def __init__(self, config_filepath: Path = CONFIG_FILE_PATH) -> None:
        self.config: Dict[str, Any] = read_yaml(config_filepath)
    
    def get_ChestXray_data_processor_config(self) -> ChestXrayDataProcessingConfig:
        """ Retrieve the configuration for processing Chest X-ray data.
        
        Returns:
            ChestXrayDataProcessingConfig: The configuration for processing Chest X-ray data.
        """
        config_ChestXray = self.config.ChestXray_data_processing_config
        create_directories([Path(config_ChestXray.processed_data_path)])
        data_processor_config = ChestXrayDataProcessingConfig(
            processed_data_path=Path(config_ChestXray.processed_data_path),
            images_base_path=Path(config_ChestXray.images_base_path),
            projections_data_path=Path(config_ChestXray.projections_path),
            reports_data_path=Path(config_ChestXray.reports_path),
            processed_data_output_path=Path(
                config_ChestXray.processed_data_output_path),
            processed_first=config_ChestXray.processed_first
        )
        return data_processor_config

    def get_ROCO_data_processing_config(self) -> ROCODataProcessingConfig:
        """ Retrieve the configuration for processing ROCO dataset.
        
        Returns:
            ROCODataProcessingConfig: The configuration for processing ROCO dataset.
        """
        config_roco = self.config.ROCO_data_processing_config
        create_directories([Path(config_roco.processed_data_path)])
        data_processor_config = ROCODataProcessingConfig(
            raw_data_path=Path(config_roco.raw_data_path),
            processed_data_path=Path(config_roco.processed_data_path),
            images_base_path=Path(config_roco.images_base_path),
        )
        return data_processor_config

    def get_training_config(self) -> Union[TrainingConfig, Tuple[TrainingConfig, ChestXrayTrainingConfig, ROCOTrainingConfig]]:
        """ Retrieve the training configuration for the model.
        
        Returns:
            Union[TrainingConfig, Tuple[TrainingConfig, ChestXrayTrainingConfig, ROCOTrainingConfig]]: The training configuration for the model.
        """
        config = self.config.train_config
        create_directories(
            [Path(config.ChestXray.metrics_folder_path), Path(config.ROCO.metrics_folder_path)])
        training_config = TrainingConfig(
            learning_rate=config.training_params.learning_rate,
            epochs=config.training_params.epochs,
            device=config.training_params.device,
            batch_size=config.training_params.batch_size,
            freeze_epochs_gpt=config.training_params.freeze_epochs_gpt,
            freeze_epochs_all=config.training_params.freeze_epochs_all,
            general_specific_fine_tuning=config.training_params.general_specific_fine_tuning
        )

        ChestXray_config = ChestXrayTrainingConfig(
            train_data_path=Path(config.ChestXray.train_data_path),
            val_data_path=Path(config.ChestXray.val_data_path),
            test_data_path=Path(config.ChestXray.test_data_path),
            training_data_path=Path(config.ChestXray.training_data_path),
            roco_trained_model_path=Path(
                config.ChestXray.roco_trained_model_path),
            trained_model_output_folder_path=Path(
                config.ChestXray.trained_model_output_folder_path),
            trained_model_output_path=Path(
                config.ChestXray.trained_model_output_path),
            metrics_folder_path=Path(config.ChestXray.metrics_folder_path),
            metrics_path=Path(config.ChestXray.metrics_path),
            processed_data_path=Path(config.ChestXray.processed_data_path),
        )

        ROCO_config = ROCOTrainingConfig(
            train_data_path=Path(config.ROCO.train_data_path),
            val_data_path=Path(config.ROCO.val_data_path),
            test_data_path=Path(config.ROCO.test_data_path),
            trained_model_output_folder_path=Path(
                config.ROCO.trained_model_output_folder_path),
            trained_model_output_path=Path(
                config.ROCO.trained_model_output_path),
            metrics_folder_path=Path(config.ROCO.metrics_folder_path),
            metrics_path=Path(config.ROCO.metrics_path),
            processed_data_path=Path(config.ROCO.processed_data_path),
        )
        if not training_config.general_specific_fine_tuning:
            return training_config, ChestXray_config
        else:
            return training_config, ChestXray_config, ROCO_config

    def get_model_config(self) -> ModelConfig:
        """ Retrieve the configuration for the model.
        
        Returns:
            ModelConfig: The configuration for the model.
        """
        config = self.config.model_config
        model_config = ModelConfig(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            seq_len=config.seq_len,
            depth=config.depth,
            attention_dropout=config.attention_dropout,
            residual_dropout=config.residual_dropout,
            mlp_ratio=config.mlp_ratio,
            mlp_dropout=config.mlp_dropout,
            emb_dropout=config.emb_dropout,
            fine_tune=config.fine_tune,
            encoder_type=config.encoder_type
        )
        return model_config

    def get_inference_config(self) -> InferenceConfig:
        """ Retrieve the configuration for the inference process.
        
        Returns:
            InferenceConfig: The configuration for the inference process.
        """
        config = self.config.inference_config
        create_directories([Path(config.metrics_folder_path)])
        inference_config = InferenceConfig(
            trained_model_path=Path(config.trained_model_path),
            inference_data_path=Path(config.inference_data_path),
            metrics_folder_path=Path(config.metrics_folder_path),
            metrics_path=Path(config.metrics_path),
            device=config.device
        )
        return inference_config
