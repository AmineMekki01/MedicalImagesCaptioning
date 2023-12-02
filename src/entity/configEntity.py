from dataclasses import dataclass
from pathlib import Path


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


@dataclass(frozen=True)
class InferenceConfig:
    trained_model_path: Path
    inference_data_path: Path
    metrics_folder_path: Path
    metrics_path: Path
    device: str
