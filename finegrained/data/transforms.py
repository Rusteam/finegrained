"""Data transforms on top of fiftyone datasets.
"""
from pathlib import Path

import fiftyone as fo
from PIL import Image, ImageOps
from fiftyone.types import ImageClassificationDirectoryTree
from tqdm import tqdm

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


def merge_diff(
    dataset: str,
    image_dir: str,
    tags: types.LIST_STR_STR = None,
    recursive: bool = True,
):
    """Merge new files into an existing dataset.

    Existing files will be skipped.
    No labels for new files are expected.
    Merger happens based on an absolute filepath.

    Args:
        dataset: existing fiftyone dataset
        image_dir: a folder with new files
        tags: tag new samples
        recursive: search for files in subfolders as well

    Returns:
        an updated fiftyone dataset
    """
    dataset = load_fiftyone_dataset(dataset)
    second = fo.Dataset.from_images_dir(
        image_dir, tags=tags, recursive=recursive
    )
    dataset.merge_samples(second, skip_existing=True)
    return dataset


def delete_samples(dataset: str, **kwargs):
    """Delete samples and associated files from a dataset

    Args:
        dataset: fiftyone dataset name
        **kwargs: dataset filters to select samples for deletion
            (must be provided)

    Returns:
        None
    """
    assert bool(kwargs), "Danger: provide dataset filters to select a subset"
    subset = load_fiftyone_dataset(dataset, **kwargs)
    delete_ids = []
    for smp in subset.select_fields(["id", "filepath"]):
        Path(smp.filepath).unlink()
        delete_ids.append(smp.id)

    full_dataset = fo.load_dataset(dataset)
    full_dataset.delete_samples(delete_ids)
    print(f"{len(delete_ids)} files deleted and removed from {dataset=}")


def exif_transpose(dataset: str, **kwargs):
    """Rotate images that have a PIL rotate tag

    Args:
        dataset: fiftyone dataset name
        **kwargs: dataset loading filters

    Returns:
        None
    """
    dataset = load_fiftyone_dataset(dataset, **kwargs)
    for smp in tqdm(dataset.select_fields("filepath"), desc="transposing"):
        orig = Image.open(smp.filepath)
        transposed = ImageOps.exif_transpose(orig)
        transposed.save(smp.filepath)


def map_labels(
    dataset: str, from_field: str, to_field: str, label_mapping: dict, **kwargs
) -> fo.DatasetView:
    """Create a new dataset field with mapped labels.

    Args:
        dataset: fiftyone dataset name
        from_field: source label field
        to_field: a new label field
        label_mapping: label mapping (use {}/None for creating a field copy)
        **kwargs: dataset loading kwargs

    Returns:
        dataset view
    """
    dataset = load_fiftyone_dataset(dataset, **kwargs)
    dataset.clone_sample_field(from_field, to_field)
    if bool(label_mapping):
        dataset = dataset.map_labels(to_field, label_mapping)
        dataset.save(to_field)
    return dataset


def _update_labels(labels: fo.Label, new_label: str):
    for one in labels.detections:
        one.label = new_label


def from_labels(dataset: str, label_field: str, from_field: str, **kwargs):
    """Re-assign classification label to detection labels.

    Args:
        dataset: fiftyone dataset name
        label_field: a field with detections to be updated
        from_field: a field with classification to get labels from
        **kwargs: dataset loading filters
    """
    dataset = load_fiftyone_dataset(dataset, **kwargs)
    dataset = dataset.exists(label_field)

    assert dataset.has_sample_field(label_field), f"Dataset does not contain {label_field=}."
    assert (doc_type := dataset.get_field(label_field).document_type) == fo.Detections, f"{label_field=} has to be of type Detections, got {doc_type=}."
    assert dataset.has_sample_field(from_field), f"Dataset does not contain {from_field=}."
    assert (doc_type := dataset.get_field(from_field).document_type) == fo.Classification, f"{from_field=} has to be of type Detections, got {doc_type=}."

    for smp in tqdm(dataset.select_fields([label_field, from_field])):
        _update_labels(smp[label_field], smp[from_field].label)
        smp.save()
