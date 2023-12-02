from src.constants import *
from src.utils.commonFunctions import read_yaml, create_directories

from src.entity.configEntity import ChestXrayDataProcessingConfig, ROCODataProcessingConfig, TrainingConfig, ChestXrayTrainingConfig, ROCOTrainingConfig, ModelConfig, InferenceConfig


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH):
        self.config = read_yaml(config_filepath)

    def get_ChestXray_data_processor_config(self) -> ChestXrayDataProcessingConfig:
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
        config_roco = self.config.ROCO_data_processing_config
        create_directories([Path(config_roco.processed_data_path)])
        data_processor_config = ROCODataProcessingConfig(
            raw_data_path=Path(config_roco.raw_data_path),
            processed_data_path=Path(config_roco.processed_data_path),
            images_base_path=Path(config_roco.images_base_path),
        )
        return data_processor_config

    def get_training_config(self) -> [TrainingConfig, ChestXrayTrainingConfig, ROCOTrainingConfig]:
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
            general_specific_fine_tuning=config.training_params.general_specific_fine_tuning,
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
            emb_dropout=config.emb_dropout
        )
        return model_config

    def get_inference_config(self) -> InferenceConfig:
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
