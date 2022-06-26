"""Train, evaluate and predict with image meta-learning .
"""
from itertools import chain
from typing import Tuple

import torch
from fiftyone import Dataset
from flash import DataModule, Trainer
from flash.core.classification import FiftyOneLabelsOutput
from flash.core.data.utilities.classification import (
    SingleLabelTargetFormatter,
    SingleNumericTargetFormatter,
)
from flash.image import (
    ImageClassificationData,
    ImageClassifier,
)
import fiftyone.core.labels as fol
from learn2learn.nn import PrototypicalClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    labels = dataset.match_tags(["train", "val"]).distinct(
        f"{label_field}.label"
    )

    # TODO optimize this
    train_files = [
        smp.filepath
        for smp in dataset.match_tags("train").select_fields("filepath")
    ]
    train_targets = [
        labels.index(smp[label_field].label)
        for smp in dataset.match_tags("train").select_fields(label_field)
    ]
    val_files = [
        smp.filepath
        for smp in dataset.match_tags("val").select_fields("filepath")
    ]
    val_targets = [
        labels.index(smp[label_field].label)
        for smp in dataset.match_tags("val").select_fields(label_field)
    ]

    data = ImageClassificationData.from_files(
        train_files=train_files,
        train_targets=train_targets,
        val_files=val_files,
        val_targets=val_targets,
        batch_size=kwargs.get("batch_size", 16),
        train_transform=get_transform(kwargs.get("train_transform")),
        val_transform=get_transform(kwargs.get("val_transform")),
        transform_kwargs=kwargs.get("transform_kwargs"),
    )
    return data, labels


def _init_support_datamodule(
    dataset: str,
    label_field: str,
    image_size: Tuple[int, int],
    batch_size: int = 4,
    **kwargs,
) -> Tuple[DataModule, Dataset]:
    dataset = load_fiftyone_dataset(dataset, **kwargs)

    data = ImageClassificationData.from_fiftyone(
        train_dataset=dataset,
        label_field=label_field,
        batch_size=batch_size,
        transform_kwargs=dict(image_size=image_size),
    )
    return data


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


def _calculate_features(
    model: ImageClassifier, dataloader: DataLoader
) -> Tuple[torch.Tensor, torch.Tensor]:
    feature_extractor = model.adapter.backbone.eval()
    features = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="extracting features batch"):
            features.append(feature_extractor(batch["input"]))
            if "target" in batch.keys():
                labels.append(batch["target"].argmax(dim=1))

    features = torch.cat(features)
    labels = torch.cat(labels) if len(labels) > 0 else labels
    return features, labels


def _predict(
    model: ImageClassifier, support_data: DataModule, query_data: DataModule
):

    support_features, support_labels = _calculate_features(
        model, support_data.train_dataloader()
    )
    query_features, _ = _calculate_features(
        model, query_data.predict_dataloader()
    )

    clf = PrototypicalClassifier(
        support=support_features, labels=support_labels
    )
    dist = clf(query_features)
    class_probs = dist.softmax(dim=1)
    conf, index = class_probs.max(dim=1)

    _get_class_label = lambda index: support_data.train_dataset.labels[index]
    predictions = [
        fol.Classification(
            label=_get_class_label(i.item()),
            confidence=c.item(),
            logits=l.cpu().numpy(),
        )
        for c, i, l in zip(conf, index, dist)
    ]

    return predictions


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


def predict(
    support_dataset: str,
    support_label_field: str,
    query_dataset: str,
    query_label_field: str,
    ckpt_path: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 4,
    support_kwargs={},
    query_kwargs={},
):
    """Classify samples from a dataset and assign values to label field

    Args:
        support_dataset: support fiftyone dataset
        support_label_field: label field for the support dataset
        query_dataset: query fiftyone dataset
        query_label_field: where to assign predictions on the query dataset
        ckpt_path: flash model checkpoint path
        image_size: Image size for inference
        batch_size: predictions batch size

    Returns:
        none
    """
    query_data, query_dataset = _init_prediction_datamodule(
        query_dataset,
        image_size=image_size,
        batch_size=batch_size,
        **query_kwargs,
    )
    support_data = _init_support_datamodule(
        support_dataset,
        support_label_field,
        image_size=image_size,
        batch_size=batch_size,
        **support_kwargs,
    )
    model = _load_pretrained_classifier(ckpt_path)
    predictions = _predict(model, support_data, query_data)
    query_dataset.set_values(query_label_field, predictions)
    print(f"{len(predictions)} predictions saved to {query_label_field} field in {query_dataset.name}")


def report(
    dataset: str,
    predictions: str,
    gt_field: str = "ground_truth",
    cmat: bool = False,
    **kwargs,
):
    """Print classification report.

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
