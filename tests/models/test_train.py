"""Test training models.
"""
from pathlib import Path

import pytest
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.splits as fous

from finegrained.models import train
from finegrained.utils.os_utils import write_yaml


@pytest.fixture
def clf_dataset():
    dataset = foz.load_zoo_dataset("caltech101").take(100).clone("clf_train_test")
    dataset.tags = []
    fous.random_split(dataset, {"train": 0.7, "val": 0.1, "test": 0.2})
    yield dataset.name
    fo.delete_dataset(dataset.name)


@pytest.fixture
def clf_config(clf_dataset, tmp_path):
    cfg_path = Path(tmp_path) / "config.yaml"
    model_path = Path(tmp_path) / "model.pt"
    cfg = dict(
        data=dict(
            dataset=clf_dataset,
            label_field="ground_truth",
            batch_size=2,
        ),
        model=dict(
            backbone="efficientnet_b0",
        ),
        trainer=dict(
            epochs=2,
            limit_train_batches=3,
            limit_val_batches=2,
            save_checkpoint=str(model_path),
            strategy="freeze"
        )
    )
    write_yaml(cfg, cfg_path)
    yield cfg_path, model_path
    cfg_path.unlink(False)
    model_path.unlink(True)


def test_classification(clf_config):
    cfg, model_path = clf_config
    train.finetune_classifier(str(cfg))

    assert model_path.exists()
