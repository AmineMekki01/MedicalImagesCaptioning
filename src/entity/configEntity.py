from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataProcessingConfig:
    processed_data_path : Path
    images_base_path: Path
    projections_data_path: str
    reports_data_path: Path
    processed_data_output_path: Path
