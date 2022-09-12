"""Run fiftyone.brain operations on a dataset.
"""
import fiftyone.brain as fob

from finegrained.utils.dataset import load_fiftyone_dataset


def compute_mistakenness(
    dataset: str, predictions: str, gt_field: str = "ground_truth", **kwargs
):
    """Estimate a probability that a ground truth label is wrong

    Args:
        dataset: fiftyone dataset name
        predictions: a field that contains model predictions
        gt_field: a field that contains ground truth data
        **kwargs: dataset loading filters

    Returns:
        none
    """
    dataset = load_fiftyone_dataset(dataset, **kwargs)
    fob.compute_mistakenness(dataset, predictions, label_field=gt_field)


def compute_hardness(dataset: str, predictions: str, **kwargs):
    """Estimate how difficult is this sample to predict.

    Args:
        dataset: fiftyone dataset name
        predictions: field with predictions
        **kwargs: dataset filters

    Returns:
        None
    """
    dataset = load_fiftyone_dataset(dataset, **kwargs)
    fob.compute_hardness(dataset, predictions)


def compute_uniqueness(dataset: str, label_field: str = "uniqueness", **kwargs):
    dataset = load_fiftyone_dataset(dataset, **kwargs)
    fob.compute_uniqueness(dataset, uniqueness_field=label_field)
    return dataset
