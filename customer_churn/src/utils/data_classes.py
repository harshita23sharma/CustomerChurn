from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PreprocessingData:
    train_path: Optional[Path] = None
    test_path: Optional[Path] = None
    batch_path: Optional[Path] = None


@dataclass
class MlFlowConfig:
    uri: Optional[Path] = None
    experiment_name: str = "churn_predictor"
    artifact_path: Optional[Path] = None
    registered_model_name: str = "churn_predictor"


class TrainerConfig:
    model_name: str = "churn_predictor"
    random_state: int = 42
    train_size: float = 0.2
    shuffle: bool = True
    params = dict = {}
