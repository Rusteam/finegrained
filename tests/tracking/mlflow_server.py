from pathlib import Path

import pytest
import mlflow
import torch

import finegrained.utils.mlflow_server
from finegrained.services import tracking
from finegrained.utils import mlflow_server
from finegrained.utils.os_utils import read_yaml


TENSORBOARD_FOLDER = Path(__file__).parents[1] / "files/tensorboard"
EVENTS = [
    TENSORBOARD_FOLDER / "events.out.tfevents.1663861466.rgaliullin0lm.35424.0",
    TENSORBOARD_FOLDER / "events.out.tfevents.1663864826.rgaliullin0lm.35424.1",
]
HPARAMS = TENSORBOARD_FOLDER / "hparams.yaml"
CKPT = TENSORBOARD_FOLDER / "checkpoints" / "epoch=24-step=5725.ckpt"


@pytest.fixture(scope="session")
def mlflow_tracking(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("mlflow")
    tracking_uri = f"sqlite:///{str(tmp_dir)}/tracking.db"
    artifact_location = str(tmp_dir / "artifacts")
    return tracking_uri, artifact_location


@pytest.fixture(scope="module")
def tfevents():
    return EVENTS, HPARAMS, CKPT


@pytest.fixture(scope="function")
def onnx_model(tmp_path):
    dest = tmp_path / "model.onnx"
    model = torch.nn.Softmax(dim=1)
    torch.onnx.export(model, torch.randn(1, 10), str(dest))
    return dest


@pytest.mark.parametrize("file", EVENTS)
def test_read_tensorboard_scalars(file):
    scalars = mlflow_server.read_tensorboard_scalars(str(file))

    for tag in ["val_accuracy", "epoch", "train_cross_entropy_epoch", "test_accuracy"]:
        if tag in scalars:
            for event in scalars[tag]:
                assert isinstance(event.wall_time, float)
                assert isinstance(event.step, int)
                assert isinstance(event.value, float)

    parsed = mlflow_server.parse_tensorboard_scalars(scalars)
    for step in parsed:
        for key in ["train_accuracy", "val_accuracy", "test_accuracy",
                    "train_cross_entropy", "val_cross_entropy", "test_cross_entropy",
                    "epoch"]:
            if key in step:
                assert isinstance(step[key], float)


def test_get_tensorboard_files(tfevents):
    files = mlflow_server.get_tensorboard_files(str(
        Path(__file__).parents[1]
        / "files/tensorboard"
    ))
    assert len(files) == 3

    events, hparams, ckpt = files

    assert len(events) == 2
    assert all([e.name.startswith("events") for e in events])
    assert hparams.name == "hparams.yaml"
    assert ckpt.suffix == ".ckpt"


def test_log_run(mlflow_tracking, tfevents, onnx_model, tmp_path):
    tracking_uri, artifact_location = mlflow_tracking
    events, hparams, ckpt = tfevents
    events = events[0]
    exp_name = "new_test_experiment"
    run_name = "new_test_run"
    run_id = tracking.log_run(
        name=run_name,
        tracking_uri=tracking_uri,
        artifact_location=artifact_location,
        events=str(events),
        hparams=str(hparams),
        ckpt=str(ckpt),
        model=str(onnx_model),
        experiment_name=exp_name,
    )

    assert mlflow.get_tracking_uri() == tracking_uri

    # experiment
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(exp_name)
    assert exp.artifact_location == artifact_location
    assert exp.lifecycle_stage == "active"

    # run
    exp_runs = client.search_runs(exp.experiment_id)
    assert len(exp_runs) == 1
    run = exp_runs[0]
    assert run.data.tags["mlflow.runName"] == run_name
    assert run.info.status == "FINISHED"
    assert run.info.run_id == run_id

    # metrics
    scalars = mlflow_server.read_tensorboard_scalars(events)
    expected = mlflow_server.parse_tensorboard_scalars(scalars)[-1]
    expected.pop("epoch")
    assert run.data.metrics == expected

    # params
    expected_params = read_yaml(hparams)
    expected_labels = expected_params.pop("labels")
    actual_params = {k: v for k, v in run.data.params.items()}
    expected_params = {k: str(v) for k, v in expected_params.items()}
    assert actual_params == expected_params

    # artifacts
    artifacts = client.list_artifacts(run_id, path=".")
    artifact_paths = [art.path for art in artifacts]

    # labels
    dest_path = tmp_path / "labels.txt"
    assert "labels.txt" in artifact_paths
    client.download_artifacts(run_id, path="labels.txt", dst_path=tmp_path)
    actual_labels = dest_path.read_text().strip().split("\n")
    assert actual_labels == expected_labels

    # checkpoint
    ckpt_path = tmp_path / "checkpoints" / ckpt.name
    assert "checkpoints" in artifact_paths
    client.download_artifacts(run_id, path=f"checkpoints/{ckpt.name}", dst_path=tmp_path)
    actual_keys = torch.load(ckpt_path).keys()
    expected_keys = torch.load(ckpt).keys()
    assert actual_keys == expected_keys

    # model
    assert "model" in artifact_paths
    model_artifacts = client.list_artifacts(run_id, path="model")
    assert len(model_artifacts) == 5
    assert any([a.path == "model/model.onnx" for a in model_artifacts])
