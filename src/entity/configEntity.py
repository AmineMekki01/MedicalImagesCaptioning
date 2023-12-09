"""
Script : 
    configEntity.py
    
Description :
    This module defines a series of dataclasses that represent configuration entities for various components of a machine learning system focused on processing and analyzing medical imaging data, specifically Chest X-rays and the ROCO dataset. These configurations are crucial for setting up the data processing, training, and inference stages of the machine learning pipeline.

Dataclasses:
    ChestXrayDataProcessingConfig: Configuration for processing Chest X-ray data. 

    ROCODataProcessingConfig: Configuration for processing ROCO dataset.

    TrainingConfig: General training configuration.
        Attributes: This class contain the attributes for the training configuration.

    ChestXrayTrainingConfig: Training configuration specific to Chest X-ray data.
        Attributes: 
            [Paths for train, validation, test data, and various model and metrics-related paths]

    ROCOTrainingConfig: Training configuration specific to the ROCO dataset.
        Attributes:
            [Paths for train, validation, test data, and various model and metrics-related paths]

    ModelConfig: Configuration for the ML model.
        Attributes: This class contain the attributes for the model configuration.

    InferenceConfig: Configuration for the inference stage.
        Attributes: This class contain the attributes for the inference configuration.
 

Each dataclass is immutable (frozen=True), ensuring that configuration values remain constant throughout the runtime of the system. This structure provides a clear, organized, and efficient way to manage and access various configuration parameters needed across different stages of the machine learning pipeline.
"""

from dataclasses import dataclass
from pathlib import Path

# @dataclass(frozen=True)
# class GetData:
#     images_base_path: Path
#     projections_path: Path
#     reports_path: Path

@dataclass(frozen=True)
class ChestXrayDataProcessingConfig:
    processed_data_path: Path
    images_base_path: Path
    projections_data_path: str
    reports_data_path: Path
    processed_data_output_path: Path
    processed_first: bool


@dataclass(frozen=True)
class ROCODataProcessingConfig:
    raw_data_path: Path
    processed_data_path: Path
    images_base_path: Path


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    freeze_epochs_gpt: int
    freeze_epochs_all: int
    device: str
    general_specific_fine_tuning: Path


@dataclass(frozen=True)
class ChestXrayTrainingConfig:
    train_data_path: Path
    val_data_path: Path
    test_data_path: Path
    training_data_path: Path
    roco_trained_model_path: Path
    trained_model_output_folder_path: Path
    trained_model_output_path: Path
    metrics_path: Path
    metrics_folder_path: Path
    processed_data_path: Path


@dataclass(frozen=True)
class ROCOTrainingConfig:
    train_data_path: Path
    val_data_path: Path
    test_data_path: Path
    trained_model_output_folder_path: Path
    trained_model_output_path: Path
    metrics_path: Path
    metrics_folder_path: Path
    processed_data_path: Path


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    embed_dim: int
    num_heads: int
    seq_len: int
    depth: int
    attention_dropout: float
    residual_dropout: float
    mlp_ratio: int
    mlp_dropout: float
    emb_dropout: float
    fine_tune: bool
    encoder_type : str


@dataclass(frozen=True)
class InferenceConfig:
    trained_model_path: Path
    inference_data_path: Path
    metrics_folder_path: Path
    metrics_path: Path
    device: str
