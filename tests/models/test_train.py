"""Test training models.
"""
import random
from pathlib import Path

import pytest
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.splits as fous
import torch.cuda
from flash import Trainer
from flash.image import ImageClassifier

from finegrained.data.brain import compute_hardness
from finegrained.data.dataset_utils import get_unique_labels
from finegrained.data.tag import split_classes, split_dataset
from finegrained.models import (
    image_classification,
    image_embedder,
    image_metalearn,
)
from finegrained.utils.os_utils import write_yaml, read_yaml


@pytest.fixture(scope="module")
def clf_dataset():
    if fo.dataset_exists("train_test_temp"):
        fo.delete_dataset("train_test_temp")
    dataset = (
        foz.load_zoo_dataset("quickstart").take(100).clone("train_test_temp")
    )
    yield dataset
    fo.delete_dataset(dataset.name)


@pytest.fixture
def clf_config(clf_dataset, tmp_path):
    cfg_path = Path(tmp_path) / "config.yaml"
    model_path = Path(tmp_path) / "model.pt"
    cfg = dict(
        data=dict(
            dataset=clf_dataset.name,
            label_field="ground_truth",
            batch_size=2,
            train_transform="randaugment",
            val_transform="nothing",
            transform_kwargs=dict(image_size=(224, 112)),
        ),
        model=dict(
            backbone="efficientnet_b0",
        ),
        trainer=dict(
            epochs=2,
            limit_train_batches=3,
            limit_val_batches=2,
            save_checkpoint=str(model_path),
            strategy="freeze",
        ),
    )
    write_yaml(cfg, cfg_path)

    clf_dataset.tags = []
    fous.random_split(clf_dataset, {"train": 0.7, "val": 0.1, "test": 0.2})

    yield cfg_path, model_path
    cfg_path.unlink(False)
    model_path.unlink(True)


@pytest.fixture()
def clf_ckpt(tmp_path):
    labels = fo.load_dataset("caltech101").distinct("ground_truth.label")
    clf = ImageClassifier(backbone="efficientnet_b0", labels=labels)
    trainer = Trainer(gpus=torch.cuda.device_count(), max_epochs=0)
    trainer.model = clf
    path = Path(tmp_path) / "clf_ckpt.pt"
    trainer.save_checkpoint(path)
    yield path
    path.unlink()


def test_classification_finetune(clf_config):
    cfg, model_path = clf_config
    image_classification.finetune(str(cfg))
    assert model_path.exists()


def test_classification_predict(clf_dataset, clf_ckpt):
    image_classification.predict(
        clf_dataset,
        label_field="test_temp_predictions",
        ckpt_path=clf_ckpt,
        include_tags=["test", "val"],
        image_size=(224, 112),
    )

    dataset = fo.load_dataset(clf_dataset)
    _ = dataset.evaluate_classifications(
        "test_temp_predictions",
        gt_field="ground_truth",
        eval_key="test_eval_temp",
    )


@pytest.fixture
def meta_learn_cfg(clf_dataset, tmp_path):
    cfg_path = Path(tmp_path) / "meta_learn.yaml"
    model_path = Path(tmp_path) / "meta_learn.pt"

    label_field = "resnet50"
    clf_dataset.tags = []
    split_classes(
        clf_dataset.name,
        label_field,
        train_size=0.5,
        val_size=0.5,
        min_samples=3,
    )

    cfg = dict(
        data=dict(
            dataset=clf_dataset.name,
            label_field=label_field,
            batch_size=2,
        ),
        model=dict(
            backbone="efficientnet_b0",
            training_strategy="prototypicalnetworks",
            training_strategy_kwargs=dict(
                ways=len(
                    get_unique_labels(
                        clf_dataset.match_tags("train"), label_field
                    )
                ),
                meta_batch_size=2,
                shots=2,
                queries=1,
                num_task=-1,
                epoch_length=2,
                test_ways=len(
                    get_unique_labels(
                        clf_dataset.match_tags("test"), label_field
                    )
                ),
                test_shots=2,
                test_queries=1,
            ),
            optimizer="AdamW",
            learning_rate=0.003,
        ),
        trainer=dict(
            epochs=2,
            save_checkpoint=str(model_path),
            strategy="freeze",
        ),
    )
    write_yaml(cfg, cfg_path)

    yield cfg_path, model_path, label_field
    cfg_path.unlink(False)
    model_path.unlink(True)


def test_metalearning_finetune(meta_learn_cfg):
    cfg, model_path, label_field = meta_learn_cfg
    image_metalearn.finetune(cfg)
    assert model_path.exists()

    conf = read_yaml(cfg)
    split_dataset(conf["data"]["dataset"], splits=dict(support=0.6, query=0.4))
    image_metalearn.predict(
        support_dataset=conf["data"]["dataset"],
        support_label_field="resnet50",
        ckpt_path=str(model_path),
        query_dataset=conf["data"]["dataset"],
        query_label_field="new_label_prediction",
        image_size=(280, 140),
        batch_size=4,
        support_kwargs=dict(include_tags=["support"]),
        query_kwargs=dict(include_tags=["query"]),
    )
    compute_hardness(
        conf["data"]["dataset"], "new_label_prediction", include_tags=["query"]
    )


@pytest.fixture
def embed_config(clf_dataset, tmp_path):
    cfg_path = Path(tmp_path) / "embed_config.yaml"
    model_path = Path(tmp_path) / "embed_model.pt"
    cfg = dict(
        data=dict(
            dataset=clf_dataset.name,
            batch_size=2,
            transform_kwargs=dict(image_size=(224, 112)),
        ),
        model=dict(
            backbone="vision_transformer",
            pretrained=True,
            training_strategy="simclr",
            head="simclr_head",
            pretraining_transform="simclr_transform",
        ),
        trainer=dict(
            epochs=2,
            save_checkpoint=str(model_path),
            limit_train_batches=2,
            strategy="freeze",
        ),
    )
    write_yaml(cfg, cfg_path)
    yield cfg_path, model_path
    cfg_path.unlink(False)
    model_path.unlink(True)


def test_embedder_finetune(embed_config):
    cfg_path, model_path = embed_config
    image_embedder.finetune(cfg_path)
    assert model_path.exists()
