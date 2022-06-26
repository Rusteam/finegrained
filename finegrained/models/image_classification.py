"""Train, evaluate and predict with image classification models.
"""
from itertools import chain
from typing import Tuple

from fiftyone import Dataset
from flash import DataModule, Trainer
from flash.core.classification import FiftyOneLabelsOutput
from flash.core.data.utilities.classification import \
    SingleLabelTargetFormatter, SingleNumericTargetFormatter
from flash.image import (
    ImageClassificationData,
    ImageClassifier,
)

from finegrained.data.dataset_utils import load_fiftyone_dataset
from finegrained.models.flash import get_transform
from finegrained.models.torch_utils import get_cuda_count
from finegrained.models.utils import validate_train_config
from finegrained.utils import types
from finegrained.utils.os_utils import read_yaml


def _init_training_datamodule(
    dataset: str, label_field: str, **kwargs
) -> Tuple[DataModule, types.LIST_STR]:
    dataset = load_fiftyone_dataset(dataset)
    labels = dataset.match_tags(["train", "val"]).distinct(f"{label_field}.label")
    data = ImageClassificationData.from_fiftyone(
        train_dataset=dataset.match_tags("train"),
        val_dataset=dataset.match_tags("val"),
        label_field=label_field,
        batch_size=kwargs.get("batch_size", 16),
        target_formatter=SingleLabelTargetFormatter(
            labels=labels, num_classes=len(labels)
        ),
        train_transform=get_transform(kwargs.get("train_transform")),
        val_transform=get_transform(kwargs.get("val_transform")),
        transform_kwargs=kwargs.get("transform_kwargs"),
    )
    return data, labels


def _init_prediction_datamodule(
    dataset: str, image_size: Tuple[int, int], batch_size: int = 4, **kwargs
) -> Tuple[DataModule, Dataset]:
    dataset = load_fiftyone_dataset(dataset, **kwargs)
    data = ImageClassificationData.from_fiftyone(
        predict_dataset=dataset,
        batch_size=batch_size,
        transform_kwargs=dict(image_size=image_size),
    )
    return data, dataset


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


def _load_pretrained_classifier(ckpt_path: str) -> ImageClassifier:
    model = ImageClassifier.load_from_checkpoint(ckpt_path)
    return model


def _finetune(model, data: DataModule, epochs, **kwargs):
    trainer = Trainer(
        max_epochs=epochs,
        limit_train_batches=kwargs.get("limit_train_batches"),
        limit_val_batches=kwargs.get("limit_val_batches"),
        gpus=get_cuda_count(),
    )

    trainer.finetune(
        model,
        datamodule=data,
        strategy=kwargs.get("strategy", ("freeze_unfreeze", 1)),
    )

    trainer.save_checkpoint(kwargs.get("save_checkpoint", "model.pt"))


def _predict(model: ImageClassifier, data: DataModule):
    trainer = Trainer(gpus=get_cuda_count())
    predictions = trainer.predict(
        model,
        datamodule=data,
        output=FiftyOneLabelsOutput(model.labels, return_filepath=False),
    )
    predictions = list(chain.from_iterable(predictions))
    return predictions


def predict(
    dataset: str,
    label_field: str,
    ckpt_path: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 4,
    **kwargs,
):
    """Classify samples from a dataset and assign values to label field

    Args:
        dataset: which dataset to run samples on
        label_field: which field to assign predictions to
        ckpt_path: flash model checkpoint path
        image_size: Image size for inference
        batch_size: predictions batch size
        **kwargs: dataset loading filters

    Returns:
        none
    """
    data, dataset = _init_prediction_datamodule(
        dataset, image_size=image_size, batch_size=batch_size, **kwargs
    )
    model = _load_pretrained_classifier(ckpt_path)
    predictions = _predict(model, data)
    dataset.set_values(label_field, predictions)


def finetune(cfg: str):
    """Fine-tune model with given data

    Args:
        cfg: path to yaml file with data, model and trainer configs
    """
    cfg = read_yaml(cfg)
    validate_train_config(cfg)
    data, class_labels = _init_training_datamodule(**cfg["data"])
    model = _init_classifier(classes=class_labels, **cfg["model"])
    _finetune(model, data, **cfg["trainer"])
    return model


def report(
    dataset: str,
    predictions: str,
    gt_field: str = "ground_truth",
    cmat: bool = False,
    **kwargs,
):
    """Print classification report

    Args:
        dataset: fiftyone dataset name
        predictions: a field with predictions
        gt_field: a field with ground truth labels
        cmat: if True, plot a confusion matrix
        **kwargs: dataset loading filters

    Returns:
        none
    """
    dataset = load_fiftyone_dataset(dataset, **kwargs)
    labels = dataset.distinct(f"{gt_field}.label")
    results = dataset.evaluate_classifications(
        predictions, gt_field=gt_field, classes=labels
    )
    results.print_report()
    if cmat:
        cm = results.plot_confusion_matrix(backend="matplotlib")
        cm.show()
