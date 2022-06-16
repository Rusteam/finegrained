"""Data transforms on top of fiftyone datasets.
"""
import fiftyone as fo
from fiftyone.types import ImageClassificationDirectoryTree
import fiftyone.utils.splits as fous

from .dataset_utils import load_fiftyone_dataset, create_fiftyone_dataset
from ..utils.general import parse_list_str
from ..utils import types


def _export_patches(
    dataset: fo.Dataset,
    label_field: str,
    export_dir: str,
) -> None:
    patches = dataset.to_patches(label_field)
    patches.export(export_dir, dataset_type=ImageClassificationDirectoryTree)


def to_patches(
    dataset: str,
    label_field: str,
    to_name: str,
    export_dir: str,
    overwrite: bool = False,
    **kwargs,
):
    """Crop out patches from a dataset and create a new one

    Args:
        dataset: a fiftyone dataset with detections
        label_field: detections label field
        to_name: a new dataset name for patches
        export_dir: where to save crops
        overwrite: if True and that name already exists, delete it
        **kwargs: dataset filters

    Returns:
        none
    """
    dataset = load_fiftyone_dataset(dataset, **kwargs)
    _export_patches(dataset, label_field, export_dir)
    create_fiftyone_dataset(
        to_name, export_dir, ImageClassificationDirectoryTree, overwrite
    )


def tag_samples(dataset: str, tags: types.LIST_STR_STR, **kwargs):
    """Tag each sample in dataset with given tags

    Args:
        dataset: fiftyone dataset name
        tags: tags to apply
        kwargs: dataset loading kwargs, i.e. filters

    Returns:
        none
    """
    dataset = load_fiftyone_dataset(dataset, **kwargs)
    dataset.tag_samples(parse_list_str(tags))


def delete_field(dataset: str, fields: types.LIST_STR_STR):
    """Delete one or more fields from a dataset

    Args:
        dataset: fiftyone dataset name
        fields: fields to delete

    Returns:
        a fiftyone dataset
    """
    dataset = load_fiftyone_dataset(dataset)
    fields = parse_list_str(fields)
    for field in fields:
        dataset.delete_sample_field(field)
        print(f"{field=} deleted from {dataset.name=}")
    return dataset


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
    fous.random_split(dataset, splits)
    return dataset.count_sample_tags()
