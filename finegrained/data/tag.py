"""Tag or untag samples with specific filters or condition
"""
from pathlib import Path
from typing import Optional

from fiftyone import ViewField as F
from fiftyone.utils import random as four
from sklearn.model_selection import train_test_split

from finegrained.data.display import label_diff
from finegrained.utils import types
from finegrained.utils.dataset import load_fiftyone_dataset
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
    **kwargs,
):
    """Create data split tags for a dataset

    Args:
        dataset: fiftyone dataset
        splits: a dict of split names and relative sizes
        kwargs: dataset loading filters

    Returns:
        a dict of split counts
    """
    dataset = load_fiftyone_dataset(dataset, **kwargs)
    four.random_split(dataset, splits)
    return dataset.count_sample_tags()


def split_classes(
    dataset: str,
    label_field: str,
    train_size: float = 0.5,
    val_size: float = 0.5,
    min_samples: int = 3,
    split_names: tuple[str, str] = ("train", "val"),
    overwrite: bool = False,
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
        split_names: splits will be tagged with these names
        overwrite: if True, existing tags are removed

    Returns:
        a dict of tag counts
    """
    dataset = load_fiftyone_dataset(dataset)
    label_counts = dataset.count_values(f"{label_field}.label")
    labels = list(filter(lambda x: label_counts[x] >= min_samples, label_counts))
    train_labels, val_labels = train_test_split(
        labels, test_size=val_size, train_size=train_size, shuffle=True
    )
    if overwrite:
        dataset.untag_samples(split_names)
    train_view = dataset.filter_labels(label_field, F("label").is_in(train_labels))
    train_view.tag_samples(split_names[0])
    val_view = dataset.filter_labels(label_field, F("label").is_in(val_labels))
    val_view.tag_samples(split_names[1])
    return dataset.count_sample_tags()


def tag_alignment(
    dataset: str, vertical: bool = True, tag: Optional[str] = None, **kwargs
) -> dict:
    """Add a vertical/horizontal tag each sample.

    Args:
        dataset: fiftyone dataset name
        vertical: if True, vertical images are tagged.
            If False, horizontal images are tagged.
        tag: overwrite default 'vertical' or 'horizontal' tag.
        **kwargs: dataset filter kwargs

    Returns:
        a dict with sample tag counts
    """
    dataset = load_fiftyone_dataset(dataset, **kwargs)
    dataset.compute_metadata()
    if vertical:
        tag = "vertical" if tag is None else tag
        tag_view = dataset.match(F("metadata.height") > F("metadata.width"))
    else:
        tag = "horizontal" if tag is None else tag
        tag_view = dataset.match(F("metadata.width") >= F("metadata.height"))
    tag_view.tag_samples(tag)
    return tag_view.count_sample_tags()


def retag_missing_labels(
    dataset: str,
    label_field: str,
    from_tags: types.LIST_STR_STR,
    to_tags: types.LIST_STR_STR,
) -> dict:
    """Remove from_tags and add to_tags for labels that are present in
        from_tags but absent in to_tags.

    Args:
        dataset: fiftyone dataset name
        label_field: a label field
        from_tags: tags with base list of class labels
        to_tags: tags with intersection of class labels

    Returns:
        a count of sample tags for a subset
    """
    # TODO test this
    diff = label_diff(dataset, label_field, tags_left=from_tags, tags_right=to_tags)
    assert len(diff) > 0, "No samples to retag"

    dataset = load_fiftyone_dataset(dataset, include_labels={label_field: diff})
    dataset.untag_samples(from_tags)
    dataset.tag_samples(to_tags)

    return dataset.count_sample_tags()


def tag_labels(
    dataset: str,
    label_field: str,
    labels: types.LIST_STR_STR,
    tags: types.LIST_STR_STR,
) -> dict:
    """Tag labels with given tags.

    Args:
        dataset: fiftyone dataset name
        label_field: a label field
        labels: labels to filter, can be a txt file with labels
        tags: tags to apply

    Returns:
        a count of label tags for a subset
    """
    if (lab := Path(labels)).is_file():
        labels = lab.read_text().strip().split("\n")
    dataset = load_fiftyone_dataset(dataset, include_labels={label_field: labels})
    dataset.tag_labels(tags, label_fields=label_field)
    return dataset.count_label_tags(label_fields=label_field)
