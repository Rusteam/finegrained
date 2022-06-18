"""Data transforms on top of fiftyone datasets.
"""
import fiftyone as fo
from fiftyone import ViewField as F
import fiftyone.utils.splits as fous
from fiftyone.types import ImageClassificationDirectoryTree

from .dataset_utils import load_fiftyone_dataset, create_fiftyone_dataset
from ..utils import types
from ..utils.general import parse_list_str


def _export_patches(
    dataset: fo.Dataset,
    label_field: str,
    export_dir: str,
) -> None:
    label_type = dataset.get_field(label_field)
    if label_type is None:
        raise KeyError(f"{label_field=} does not exist in {dataset.name=}")
    label_type = label_type.document_type
    if label_type == fo.Classification:
        patches = dataset.exists(label_field)
    elif label_type in [fo.Detections, fo.Polylines]:
        patches = dataset.to_patches(label_field)
    else:
        raise ValueError(f"{label_type=} cannot be exported as patches")
    patches.export(
        export_dir,
        dataset_type=ImageClassificationDirectoryTree,
        label_field=label_field,
    )


def to_patches(
    dataset: str,
    label_field: str,
    to_name: str,
    export_dir: str,
    overwrite: bool = False,
    **kwargs,
) -> fo.Dataset:
    """Crop out patches from a dataset and create a new one

    Args:
        dataset: a fiftyone dataset with detections
        label_field: detections label field
        to_name: a new dataset name for patches
        export_dir: where to save crops
        overwrite: if True and that name already exists, delete it
        **kwargs: dataset filters

    Returns:
        fiftyone dataset object
    """
    dataset = load_fiftyone_dataset(dataset, **kwargs)
    label_field = parse_list_str(label_field)
    for field in label_field:
        _export_patches(dataset, field, export_dir)
    new = create_fiftyone_dataset(
        to_name, export_dir, ImageClassificationDirectoryTree, overwrite
    )
    return new


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


def prefix_label(dataset: str, label_field: str, dest_field: str, prefix: str):
    """Prepend each label with given prefix

    Args:
        dataset: fiftyone dataset name
        label_field: a field with class labels
        dest_field: a new field to create with '<prefix>_<label>' values
        prefix: a prefix value

    Returns:
        fiftyone dataset object
    """
    dataset = load_fiftyone_dataset(dataset)
    values = [
        fo.Classification(label=f"{prefix}_{smp[label_field].label}")
        for smp in dataset.select_fields(label_field)
    ]
    dataset.set_values(dest_field, values)
    return dataset


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
