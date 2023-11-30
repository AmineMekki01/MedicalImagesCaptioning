from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataProcessingConfig:
    processed_data_path: Path
    images_base_path: Path
    projections_data_path: str
    reports_data_path: Path
    processed_data_output_path: Path


@dataclass(frozen=True)
class TrainingConfig:
    training_data_path: Path
    trained_model_output_path: Path
    metrics_folder_path: Path
    metrics_path: Path
    batch_size: int
    epochs: int
    learning_rate: float
    freeze_epochs_gpt: int
    freeze_epochs_all: int
    device: str


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
