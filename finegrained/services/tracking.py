"""Log metrics and models to MLflow.
"""
from pathlib import Path
from typing import Optional

import mlflow
import onnx

from finegrained.utils import mlflow_server
from finegrained.utils.mlflow_server import get_tensorboard_files
from finegrained.utils.os_utils import read_yaml


def log_events(path: str | list) -> None:
    """Log events history as metrics to MLflow

    Args:
        path: tensorboard events path
    """
    if isinstance(path, str):
        path = [path]

    for one in path:
        scalars = mlflow_server.read_tensorboard_scalars(one)
        metrics = mlflow_server.parse_tensorboard_scalars(scalars)

        for step, vals in enumerate(metrics):
            step = vals.pop("epoch", step)
            mlflow.log_metrics(vals, step=step)


def log_hparams(path: str) -> None:
    """Log hyperparameters to MLflow

    Args:
        path: path to hparams.yaml file
    """
    hparams = read_yaml(path)
    labels = hparams.pop("labels")

    mlflow.log_params(hparams)
    if bool(labels):
        mlflow.log_text("\n".join(labels), artifact_file="labels.txt")


def log_ckpt(path: str) -> None:
    """Log a torch checkpoint to MLflow

    Args:
        path: path to a checkpoint file
    """
    path = Path(path)
    if path.is_file():
        mlflow.log_artifact(path, "checkpoints")
    else:
        mlflow.log_artifacts(path, "checkpoints")


def log_run(
    name: Optional[str] = None,
    events: Optional[str] = None,
    hparams: Optional[str] = None,
    ckpt: Optional[str] = None,
    model: Optional[str] = None,
    log_dir: Optional[str] = None,
    tracking_uri: str = "./mlruns",
    experiment_name: Optional[str] = None,
    artifact_location: Optional[str] = None,
):
    """Connect to MLflow and log metrics, params and files.

    Either pass paths to events, hparams and ckpt or a single path to log_dir.

    Args:
        name: mlflow run name
        events: path to tensorboard events file as metrics
        hparams: path to hparams.yaml file as params and labels
        ckpt: path to torch checkpoint file or a dir with checkpoint files
        model: path to onnx model to log as MLflow models
        log_dir: provide path to tensorboard log dir in order to automatically
            get events, hparams and ckpt paths.
        tracking_uri: mlflow tracking uri
        experiment_name: if set, use this experiment name (if does not exist, creates a new one)
        artifact_location: set artifact location for this experiment
                            (applicable only when a new experiment created)

    Returns:
        MLflow run id
    """
    assert log_dir or events, "Either log_dir or events must be specified"

    mlflow_server.connect_to_mlflow(tracking_uri,
                                    experiment_name=experiment_name,
                                    artifact_location=artifact_location)

    if log_dir:
        events, hparams, ckpt = get_tensorboard_files(log_dir)

    with mlflow.start_run(run_name=name) as run:
        run_id = run.info.run_id

        if events:
            log_events(events)
        if hparams:
            log_hparams(hparams)
        if ckpt:
            log_ckpt(ckpt)
        if model:
            log_model(model)

    return run_id


def log_model(path: str) -> None:
    """Log ONNX model to MLflow run.

    Args:
        path: path to onnx model file
    """
    assert path.endswith(".onnx")
    model = onnx.load_model(path)
    mlflow.onnx.log_model(model, artifact_path="model")
