import os
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from src.pipelines.configs import (TRAINING_DATA_PATH, TrainerConfig,
                                   TrainerInputConfig)
from src.pipelines.data_preprocessor import DataPreprocessor
from src.pipelines.training_pipeline import TrainingPipeline

# Preparation
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IS_TRAIN = True
preprocess_step = DataPreprocessor(is_train=IS_TRAIN, data_path=TRAINING_DATA_PATH)
training_step = TrainingPipeline(
    params=TrainerConfig.params,
    train_path=TrainerInputConfig.train_path,
    test_path=TrainerInputConfig.test_path,
    target=TrainerInputConfig.target,
    model_name=TrainerConfig.model_name,
)

default_args = {
    "owner": "user",  # user's name
    "depends_on_past": False,  # keeps a task from getting triggered if the previous schedule for the task hasn’t succeeded.
    "retries": 0,  # Number of retries for a dag
    "catchup": False,  # Run the dag from the start_date to today in respect to the trigger frequency
}

with DAG(
    "training-pipeline",  # Dag name
    default_args=default_args,  # Default dag's arguments that can be share accross dags
    start_date=datetime(2023, 12, 19),  # Reference date for the scheduler (mandatory)
    tags=["training"],  # tags
    schedule=None,  # No repetition
) as dag:
    preprocessing_task = PythonOperator(
        task_id="preprocessing",
        python_callable=preprocess_step,
        op_kwargs={"data_path": TRAINING_DATA_PATH},
    )
    training_task = PythonOperator(
        task_id="training",
        python_callable=training_step,
        # op_kwargs={train_path=TrainerInputConfig.train_path,
    )
    preprocessing_task >> training_task
