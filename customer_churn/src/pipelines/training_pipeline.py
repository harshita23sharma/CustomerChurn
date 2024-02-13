import time
from typing import Any, Dict

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from src.pipelines.configs import MlFlowConfig, TrainerConfig


class TrainingPipeline:
    def __init__(
        self,
        params: Dict[str, Any],
        train_path: str,
        test_path: str,
        target: str,
        model_name: str = TrainerConfig.model_name,
    ) -> None:
        self.params = params
        self.model_name = model_name
        self.train_path = train_path
        self.test_path = test_path
        self.target = target

    def __call__(self):
        mlflow.set_tracking_uri(MlFlowConfig.uri)
        experiment = mlflow.get_experiment_by_name(MlFlowConfig.experiment_name)

        if experiment is None:
            experiment_id = mlflow.create_experiment(
                name=MlFlowConfig.experiment_name, artifact_location=MlFlowConfig.artifact_path
            )
        else:
            experiment_id = experiment.experiment_id
        mlflow.set_experiment(experiment_id=experiment_id)

        with mlflow.start_run():
            print("MLFLOW..........")
            train_df = pd.read_csv(self.train_path)
            test_df = pd.read_csv(self.test_path)
            X_train, y_train, X_test, y_test = (
                train_df.drop(columns=[self.target]),
                train_df[self.target],
                test_df.drop(columns=[self.target]),
                test_df[self.target],
            )
            lr_clf = LogisticRegression()
            lr_model, train_lr, test_lr, f1_lr, pred_lr, time_lr = self.parameter_finder(
                lr_clf, self.params, X_train, y_train, X_test, y_test
            )

            model = lr_model
            # Evaluate
            y_test = test_df[self.target]
            y_pred = model.predict(test_df.drop(self.target, axis=1))

            # Metrics
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)
            print(classification_report(y_test, y_pred))

            metrics = {"precision": precision, "recall": recall, "roc_auc": roc_auc}
            # Mlflow
            mlflow.log_params(self.params)
            mlflow.log_metrics(metrics)
            mlflow.set_tag(key="model", value=self.model_name)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=MlFlowConfig.artifact_path,
            )
            # save the model to disk
            joblib.dump(model, filename=MlFlowConfig.model_path)
            model = joblib.load(filename=MlFlowConfig.model_path)
            return metrics

    def parameter_finder(self, model, parameters, X_train, y_train, X_test, y_test):
        start = time.time()

        grid = GridSearchCV(
            model,
            param_grid=parameters,
            refit=True,
            cv=KFold(shuffle=True, random_state=1),
            n_jobs=-1,
        )
        grid_fit = grid.fit(X_train, y_train)
        best = grid_fit.best_estimator_
        y_pred = best.predict(X_test)

        train_score = best.score(X_train, y_train)
        test_score = best.score(X_test, y_test)
        F1_score = f1_score(y_test, y_pred).round(2)

        model_name = str(model).split("(")[0]

        end = time.time()
        takes_time = np.round(end - start, 2)

        print(f"The best parameters for {model_name} model is: {grid_fit.best_params_}")
        print("--" * 10)
        print(
            f"(R2 score) in the training set is {train_score:0.2%} for {model_name} model."
        )
        print(
            f"(R2 score) in the testing set is {test_score:0.2%} for {model_name} model."
        )
        print(f"F1 score is {F1_score:,} for {model_name} model.")
        print("--" * 10)
        print(f"Runtime of the program is: {end - start:0.2f}")

        return best, train_score, test_score, F1_score, y_pred, takes_time
