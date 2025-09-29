import os
from contextlib import contextmanager
import mlflow


def setup_mlflow(tracking_uri: str | None = None, experiment_name: str = "churn-ml") -> None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


@contextmanager
def start_run(run_name: str | None = None):
    with mlflow.start_run(run_name=run_name):
        yield


def log_params(params: dict) -> None:
    if params:
        mlflow.log_params(params)


def log_metrics(metrics: dict) -> None:
    if metrics:
        mlflow.log_metrics(metrics)


def log_artifact(path: str) -> None:
    if os.path.exists(path):
        mlflow.log_artifact(path)


