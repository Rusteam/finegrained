"""Run fiftyone.brain operations on a dataset.
"""
from typing import Optional

import fiftyone.brain as fob
import fiftyone.zoo as foz

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


def compute_uniqueness(
    dataset: str,
    field: str = "uniqueness",
    model: Optional[str] = None,
    model_kwargs: dict = {},
    **kwargs
):
    """Create a field that estimates uniqueness of each sample in the dataset.

    Args:
        dataset: fiftyone dataset name
        field: where to store uniqueness values
        model: fiftyone zoo model name
        model_kwargs: model loading kwargs
        **kwargs: dataset loading kwargs
    """
    import fiftyone.utils.torch as fout

    dataset = load_fiftyone_dataset(dataset, **kwargs)
    if isinstance(model, str) and "/" in model:
        repo, model = model.rsplit("/", 1)
        model = fout.load_torch_hub_image_model(repo, model, **model_kwargs)
    else:
        model = foz.load_zoo_model(model, **model_kwargs) if model else None
    fob.compute_uniqueness(dataset, uniqueness_field=field, model=model)
    return dataset
