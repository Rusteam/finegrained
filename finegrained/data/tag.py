"""Tag or untag samples with specific filters or condition
"""
from fiftyone import ViewField as F
from fiftyone.utils import random as four
from sklearn.model_selection import train_test_split

from finegrained.utils.dataset import (
    load_fiftyone_dataset,
)
from finegrained.utils import types
from finegrained.utils.general import parse_list_str


def tag_samples(dataset: str, tags: types.LIST_STR_STR, **kwargs) -> dict:
    """Tag each sample in dataset with given tags

    Args:
        dataset: fiftyone dataset name
        tags: tags to apply
        kwargs: dataset loading kwargs, i.e. filters

    Returns:
        a dict of sample tag counts
    """
    dataset = load_fiftyone_dataset(dataset, **kwargs)
    dataset.tag_samples(parse_list_str(tags))
    return dataset.count_sample_tags()


def split_dataset(
    dataset: str,
    splits: types.DICT_STR_FLOAT = {"train": 0.8, "val": 0.1, "test": 0.1},
):
    """Create data split tags for a dataset

    Args:
        dataset: fiftyone dataset
        splits: a dict of split names and relative sizes

    Returns:
        a dict of split counts
    """
    dataset = load_fiftyone_dataset(dataset)
    four.random_split(dataset, splits)
    return dataset.count_sample_tags()


def split_classes(
    dataset: str,
    label_field: str,
    train_size: float = 0.5,
    val_size: float = 0.5,
    min_samples: int = 3,
) -> types.DICT_STR_FLOAT:
    """Split classes in a dataset into train and val.

    Used for meta-learning.

    Args:
        dataset: fiftyone dataset name
        label_field: which field to use for classes
        train_size: fraction of classes to tag as train
        val_size: fraction of classes to tag as val
        min_samples: minimum number of samples
            per class to include a class into a split

    Returns:
        a dict of tag counts
    """
    dataset = load_fiftyone_dataset(dataset)
    label_counts = dataset.count_values(f"{label_field}.label")
    labels = list(filter(lambda x: label_counts[x] >= min_samples,
                         label_counts))
    train_labels, val_labels = train_test_split(
        labels, test_size=val_size, train_size=train_size, shuffle=True
    )
    train_view = dataset.filter_labels(
        label_field, F("label").is_in(train_labels)
    )
    train_view.tag_samples("train")
    val_view = dataset.filter_labels(label_field, F("label").is_in(val_labels))
    val_view.tag_samples("val")
    return dataset.count_sample_tags()


def tag_vertical(dataset: str, tag: str = "vertical", **kwargs) -> dict:
    """Add a tag to samples where height is larger than width

    Args:
        dataset: fiftyone dataset name
        tag: a tag to add
        **kwargs: dataset filter kwargs

    Returns:
        a dict with sample tag counts
    """
    dataset = load_fiftyone_dataset(dataset, **kwargs)
    dataset.compute_metadata()
    vertical_view = dataset.match(F("metadata.height") > F("metadata.width"))
    vertical_view.tag_samples(tag)
    return vertical_view.count_sample_tags()
