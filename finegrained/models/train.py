from typing import Tuple

import torch.cuda
from flash import Trainer, DataModule
from flash.core.data.utilities.classification import SingleLabelTargetFormatter
from flash.image import ImageClassificationData, ImageClassifier

from finegrained.data.dataset_utils import (
    load_fiftyone_dataset,
)
from finegrained.utils import types
from finegrained.utils.os_utils import read_yaml


def _init_classification_datamodule(
    dataset: str, label_field: str, **kwargs
) -> Tuple[DataModule, types.LIST_STR]:
    dataset = load_fiftyone_dataset(dataset)
    labels = dataset.distinct(f"{label_field}.label")
    data = ImageClassificationData.from_fiftyone(
        train_dataset=dataset.match_tags("train"),
        val_dataset=dataset.match_tags("val"),
        test_dataset=dataset.match_tags("test"),
        label_field=label_field,
        batch_size=kwargs.get("batch_size", 16),
        target_formatter=SingleLabelTargetFormatter(
            labels=labels, num_classes=len(labels)
        ),
    )
    return data, labels


def _init_classifier(
    backbone: str, classes: types.LIST_STR, **kwargs
) -> ImageClassifier:
    clf = ImageClassifier(
        num_classes=len(classes),
        backbone=backbone,
        labels=classes,
        **kwargs,
    )
    return clf


def _finetune(model, data: DataModule, epochs, **kwargs):
    trainer = Trainer(
        max_epochs=epochs,
        limit_train_batches=kwargs.get("limit_train_batches"),
        limit_val_batches=kwargs.get("limit_val_batches"),
        gpus=torch.cuda.device_count(),
    )

    trainer.finetune(
        model,
        datamodule=data,
        strategy=kwargs.get("strategy", ("freeze_unfreeze", 1)),
    )

    trainer.save_checkpoint(kwargs.get("save_checkpoint", "model.pt"))


def _validate_config(cfg: dict):
    for key in ["data", "model", "trainer"]:
        assert key in cfg, f"config file has to contain {key=}"
    for key in ["dataset", "label_field"]:
        assert key in cfg["data"], f"data has to contain {key=}"
    for key in ["backbone"]:
        assert key in cfg["model"], f"model has to contain {key=}"
    for key in ["epochs"]:
        assert key in cfg["trainer"], f"trainer has to contain {key=}"


def finetune_classifier(cfg: str):
    cfg = read_yaml(cfg)
    _validate_config(cfg)
    data, class_labels = _init_classification_datamodule(**cfg["data"])
    model = _init_classifier(classes=class_labels, **cfg["model"])
    _finetune(model, data, **cfg["trainer"])
