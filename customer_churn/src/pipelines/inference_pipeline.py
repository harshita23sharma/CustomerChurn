import json
import logging
from pathlib import Path
from typing import List

import joblib
import mlflow
import pandas as pd
from src.pipelines.configs import MlFlowConfig

LOGGER = logging.getLogger(__name__)

from typing import List


class InferencePipeline:
    "Get the model from the model registry and predict in batch"

    def __call__(self, batch_path: Path) -> List[int]:
        """Use the MLFlow artifact built-in predict.

        Args:
            batch_path (Path): Input batch_path

        Return (List[int]):
            Predictions
        """
        model = self._load_model(registered_model_name=MlFlowConfig.model_path)
        batch = self._load_batch(batch_path)
        batch = batch.drop(columns=["Churn"])
        if model:
            # Transform np.ndarray into list for serialization
            prediction = model.predict(batch).tolist()
            LOGGER.info(f"Prediction: {prediction}")
            return json.dumps(prediction)
        else:
            LOGGER.warning(
                "No model used for prediction. Model registry probably empty."
            )

    @staticmethod
    def _load_model(registered_model_name: str):
        """Load model from model registry.

        Args:
            registered_model_name (str): Name

        Returns:
            Model artifact
        """
        # mlflow.set_tracking_uri(MlFlowConfig.uri)
        # models = mlflow.search_registered_models(
        #     filter_string=f"name = '{MlFlowConfig.registered_model_name}'"
        # )
        # print("Model Loaded from Registry")
        models = joblib.load(registered_model_name)
        return models
        # if models:
        #     latest_model_version = models[0].latest_versions[0].version
        #     LOGGER.info(
        #         f"Latest model version in the model registry used for prediction: {latest_model_version}"
        #     )
        #     model = mlflow.sklearn.load_model(
        #         model_uri=f"models:/{registered_model_name}/{latest_model_version}"
        #     )
        #     return model
        # else:
        #     LOGGER.warning(
        #         f"No model in the model registry under the name: {MlFlowConfig.registered_model_name}."
        #     )

    @staticmethod
    def _load_batch(batch_path: Path) -> pd.DataFrame:
        """Load dataframe from path"""
        batch = pd.read_csv(batch_path)
        LOGGER.info(f"Batch columns: {batch.columns}")
        return batch
