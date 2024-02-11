from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from src.pipelines.configs import INFERENCE_DATA_PATH, PreprocessConfig
from src.pipelines.data_preprocessor import DataPreprocessor
from src.pipelines.inference_pipeline import InferencePipeline

# Preparation
IS_TRAIN = False


# Steps

preprocess_step = DataPreprocessor(is_train=IS_TRAIN, data_path=INFERENCE_DATA_PATH)

inference_step = InferencePipeline()


default_args = {
    "owner": "user",
    "depends_on_past": False,
    "retries": 0,
    "catchup": False,
}

with DAG(
    "inference-pipeline",
    default_args=default_args,
    start_date=datetime(2023, 12, 20),
    tags=["inference"],
    schedule=None,
) as dag:
    preprocessing_task = PythonOperator(
        task_id="preprocessing",
        python_callable=preprocess_step,
        op_kwargs={"data_path": INFERENCE_DATA_PATH},
    )

    inference_task = PythonOperator(
        task_id="inference",
        python_callable=inference_step,
        op_kwargs={"batch_path": PreprocessConfig.test_path},
    )

    preprocessing_task >> inference_task
