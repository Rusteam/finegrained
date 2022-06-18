"""Run predictions with a model and save to fiftyone dataset.
"""
from itertools import chain
from typing import Tuple

import torch.cuda
from fiftyone import Dataset
from flash import DataModule, Trainer
from flash.core.classification import FiftyOneLabelsOutput
from flash.image import ImageClassificationData, ImageClassifier


from finegrained.data.dataset_utils import load_fiftyone_dataset


def _init_classification_datamodule(
    dataset: str, batch_size: int = 4, **kwargs
) -> Tuple[DataModule, Dataset]:
    dataset = load_fiftyone_dataset(dataset, **kwargs)
    data = ImageClassificationData.from_fiftyone(
        predict_dataset=dataset,
        batch_size=batch_size,
    )
    return data, dataset


def _load_pretrained_classifier(ckpt_path: str) -> ImageClassifier:
    model = ImageClassifier.load_from_checkpoint(ckpt_path)
    return model


def _predict(model: ImageClassifier, data: DataModule):
    trainer = Trainer(gpus=torch.cuda.device_count())
    predictions = trainer.predict(
        model,
        datamodule=data,
        output=FiftyOneLabelsOutput(model.labels, return_filepath=False),
    )
    predictions = list(chain.from_iterable(predictions))
    return predictions


def predict_classes(
    dataset: str,
    label_field: str,
    ckpt_path: str,
    batch_size: int = 4,
    **kwargs,
):
    """Classify samples from a dataset and assign values to label field

    Args:
        dataset: which dataset to run samples on
        label_field: which field to assign predictions to
        ckpt_path: flash model checkpoint path
        batch_size: predictions batch size
        **kwargs: dataset loading filters

    Returns:
        none
    """
    data, dataset = _init_classification_datamodule(
        dataset, batch_size=batch_size, **kwargs
    )
    model = _load_pretrained_classifier(ckpt_path)
    predictions = _predict(model, data)
    dataset.set_values(label_field, predictions)


def classification_report(
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
