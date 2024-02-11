import os
from pathlib import Path

REPO_DIR = Path(os.path.realpath(""))
DATA_PATH = "data/"
print(f"REPO_DIR = {REPO_DIR}")
INFERENCE_DATA_PATH = REPO_DIR / "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
TRAINING_DATA_PATH = REPO_DIR / "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = REPO_DIR / "model"
print(f"TRAINING_DATA_PATH={TRAINING_DATA_PATH}")


class PreprocessConfig:
    train_path = REPO_DIR / DATA_PATH / "preprocessed/train.csv"
    test_path = REPO_DIR / DATA_PATH / "preprocessed/test.csv"
    batch_path = REPO_DIR / DATA_PATH / "preprocessed/batch.csv"


class TrainerInputConfig:
    train_path = REPO_DIR / DATA_PATH / "preprocessed/train.csv"
    test_path = REPO_DIR / DATA_PATH / "preprocessed/test.csv"
    target = "Churn"


class TrainerConfig:
    model_name = "logistic-regression"
    random_state = 42
    train_size = 0.2
    shuffle = True
    params = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
    }


class MlFlowConfig:
    model_path = REPO_DIR / MODEL_PATH / "model.sav"
    uri = "http://0.0.0.0:8000"
    experiment_name = "churn_predictor"
    artifact_path = "model-artifact"
    registered_model_name = "churn_predictor"
