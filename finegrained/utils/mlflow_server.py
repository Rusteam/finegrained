"""MLflow and Tensorboard utils.
"""
from pathlib import Path
from typing import Optional

import mlflow
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

SCALAR_TYPE = dict[str, list[event_accumulator.ScalarEvent]]


def read_tensorboard_scalars(events: str) -> SCALAR_TYPE:
    """Read Tensorboard scalars.

    Args:
        events: path to tensorboard events file.

    Returns:
         a dict of scalar events.
    """
    scalars = {}
    events = str(events) if isinstance(events, Path) else events
    event_acc = event_accumulator.EventAccumulator(events)
    event_acc.Reload()
    for tag in event_acc.Tags()["scalars"]:
        scalars[tag] = event_acc.Scalars(tag)
    return scalars


def _get_scalar_values(scalars: SCALAR_TYPE, tag: str) -> Optional[list[float]]:
    """Get scalar values from a dict of scalar events.

    Args:
        scalars: a dict of scalar events.
        tag: the tag of the scalar event.

    Returns:
        a list of scalar values.
    """
    if tag in scalars:
        return [acc.value for acc in scalars[tag]]
    else:
        return None


def parse_tensorboard_scalars(scalars: SCALAR_TYPE) -> list[dict]:
    """Parse Tensorboard scalars into a format to log to mlflow.

    Args:
        scalars: a dict of scalar events.

    Returns:
        a list of dict with metrics and step value
    """
    train_accuracy = _get_scalar_values(scalars, tag="train_accuracy_epoch")
    train_ce = _get_scalar_values(scalars, tag="train_cross_entropy_epoch")
    val_accuracy = _get_scalar_values(scalars, tag="val_accuracy")
    val_ce = _get_scalar_values(scalars, tag="val_cross_entropy")
    test_accuracy = _get_scalar_values(scalars, tag="test_accuracy")
    test_ce = _get_scalar_values(scalars, tag="test_cross_entropy")
    epoch = sorted(list(set(_get_scalar_values(scalars, tag="epoch"))))
    epoch = list(map(int, epoch))

    df = pd.DataFrame(
        {
            "train_accuracy": train_accuracy,
            "train_cross_entropy": train_ce,
            "val_accuracy": val_accuracy,
            "val_cross_entropy": val_ce,
            "test_accuracy": test_accuracy,
            "test_cross_entropy": test_ce,
            "epoch": epoch,
        }
    )
    cols = df.columns[df.isnull().sum(0) == 0]
    parsed = df[cols].to_dict("records")

    return parsed


def connect_to_mlflow(
    tracking_uri: str,
    experiment_name: Optional[str] = None,
    artifact_location: Optional[str] = None,
) -> None:
    """Connect to MLflow.

    Args:
        tracking_uri: MLflow tracking uri.
        experiment_name: if given, set MLflow experiment name.
        artifact_location: if experiment given, set its default artifact location
    """
    mlflow.set_tracking_uri(tracking_uri)

    if experiment_name:
        experiments = mlflow.search_experiments()
        if experiment_name in [exp.name for exp in experiments]:
            mlflow.set_experiment(experiment_name)
        else:
            mlflow.create_experiment(experiment_name, artifact_location)
            mlflow.set_experiment(experiment_name)


def get_tensorboard_files(path: str) -> tuple[list[Path], Path, Path]:
    """Get tensorboard files from a path.

    Args:
        path: path to a tensorboard log directory

    Returns:
        a tuple of paths: a list of events, hparams.yaml and latest checkpoint
    """
    path = Path(path)

    events = list(path.glob("events.*"))
    assert len(events) > 0, "Expected at least one event file"

    hparams = path / "hparams.yaml"
    assert hparams.exists(), f"{hparams=} does not exist."

    ckpt = list(path.glob("checkpoints/*.ckpt"))
    assert len(ckpt) >= 1, f"no checkpoint found in {str(path)}"
    ckpt = ckpt[-1]

    return events, hparams, ckpt
