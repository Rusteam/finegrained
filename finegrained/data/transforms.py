"""Data transforms on top of fiftyone datasets.
"""
import shutil
from pathlib import Path
from typing import Optional

import fiftyone as fo
from fiftyone.types import ImageClassificationDirectoryTree
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm

from ..utils import types
from ..utils.dataset import create_fiftyone_dataset, load_fiftyone_dataset
from ..utils.general import parse_list_str
from ..utils.os_utils import read_json, read_yaml, write_json


def _export_patches(
    dataset: fo.Dataset,
    label_field: str,
    export_dir: Path,
    splits: Optional[list[str]] = None,
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
    if splits:
        for tag in splits:
            patches.match_tags(tag).export(
                str(export_dir / tag),
                dataset_type=ImageClassificationDirectoryTree,
                label_field=label_field,
            )
    else:
        patches.export(
            str(export_dir),
            dataset_type=ImageClassificationDirectoryTree,
            label_field=label_field,
        )


def to_patches(
    dataset: str,
    label_field: str | list[str],
    to_name: str,
    export_dir: str,
    overwrite: bool = False,
    splits: Optional[list[str]] = None,
    **kwargs,
) -> fo.Dataset:
    """Crop out patches from a dataset and create a new one.

    Args:
        dataset: a fiftyone dataset with detections
        label_field: label field(s) with detection, classification or polylines
        to_name: a new dataset name for patches
        export_dir: where to save crops
        overwrite: if True and that name already exists, delete it
        splits: if provided, these tags will be used to split patches into subsets
        **kwargs: dataset filters

    Returns:
        fiftyone dataset object
    """
    export_dir = Path(export_dir)

    # prompt overwriting if dataset or folder exist
    if not overwrite:
        if fo.dataset_exists(to_name):
            raise ValueError(
                f"{to_name=} dataset already exists. Use --overwrite or delete it."
            )
        if export_dir.exists():
            raise ValueError(
                f"{str(export_dir)=} already exists. "
                "User --overwrite or delete it manually"
            )
    else:
        if export_dir.exists():
            shutil.rmtree(export_dir)

    dataset = load_fiftyone_dataset(dataset, **kwargs)
    label_field = parse_list_str(label_field)

    # make sure splits are present if given
    if splits:
        splits = parse_list_str(splits)
        tag_counts = dataset.count_sample_tags()
        assert all(
            [s in tag_counts for s in splits]
        ), f"{dataset.name=} does not contain all {splits=}"

    # export each label field
    for field in label_field:
        assert dataset.has_sample_field(
            field
        ), f"{dataset.name=} does not contain {field=}"
        _export_patches(dataset, field, export_dir, splits)

    # import all together, tag if needed
    new = create_fiftyone_dataset(
        name=to_name,
        src=export_dir if splits is None else None,
        dataset_type=ImageClassificationDirectoryTree,
        overwrite=overwrite,
    )
    if splits:
        for tag in splits:
            new.add_dir(
                dataset_dir=str(export_dir / tag),
                dataset_type=ImageClassificationDirectoryTree,
                tags=tag,
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
    second = fo.Dataset.from_images_dir(image_dir, tags=tags, recursive=recursive)
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
        try:
            orig = Image.open(smp.filepath)
            transposed = ImageOps.exif_transpose(orig)
            transposed.save(smp.filepath)
        except UnidentifiedImageError as e:
            print(e, "at", smp.filepath)


def map_labels(
    dataset: str,
    from_field: str,
    to_field: str,
    label_mapping: Optional[dict] = None,
    overwrite: bool = False,
    **kwargs,
) -> fo.DatasetView:
    """Create a new dataset field with mapped labels.

    Args:
        dataset: fiftyone dataset name
        from_field: source label field
        to_field: a new label field
        label_mapping: label mapping (use {}/None for creating a field copy)
        overwrite: if to_field already exists, then overwrite it
        **kwargs: dataset loading kwargs

    Returns:
        dataset view
    """
    dataset = load_fiftyone_dataset(dataset, **kwargs)

    if overwrite and dataset.has_sample_field(to_field):
        delete_field(dataset.dataset_name, to_field)
    elif not overwrite and dataset.has_sample_field(to_field):
        raise ValueError(f"{to_field=} already exists")

    dataset.clone_sample_field(from_field, to_field)
    if bool(label_mapping):
        dataset = dataset.map_labels(to_field, label_mapping)
        dataset.save(to_field)
    return dataset


def _update_labels(labels: fo.Detections, new_label: str):
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

    assert dataset.has_sample_field(
        label_field
    ), f"Dataset does not contain {label_field=}."
    assert (
        doc_type := dataset.get_field(label_field).document_type
    ) == fo.Detections, f"{label_field=} has to be of type Detections, got {doc_type=}."
    assert dataset.has_sample_field(
        from_field
    ), f"Dataset does not contain {from_field=}."
    assert (
        doc_type := dataset.get_field(from_field).document_type
    ) == fo.Classification, (
        f"{from_field=} has to be of type Detections, got {doc_type=}."
    )

    for smp in tqdm(dataset.select_fields([label_field, from_field])):
        _update_labels(smp[label_field], smp[from_field].label)
        smp.save()


def from_label_tag(dataset: str, label_field: str, label_tag: str, **kwargs) -> dict:
    """Update a label_field label with its label_tag.

    Args:
        dataset: fiftyone dataset name
        label_field: a field that contains detections labels.
        label_tag: labels that contain this tag, will be renamed to it.
        **kwargs: dataset loading filters

    Returns:
        updated label values
    """
    kwargs = kwargs | {"label_tags": label_tag}
    dataset = load_fiftyone_dataset(dataset, **kwargs)

    for smp in tqdm(dataset.select_fields(label_field), desc="updating samples"):
        for det in smp[label_field].detections:
            if label_tag in det.tags:
                det.label = label_tag
                smp.save()

    return dataset.count_values(f"{label_field}.detections.label")


def combine_datasets(
    dest_name: str,
    label_field: str,
    cfg: str,
    persistent: bool = True,
    overwrite: bool = False,
):
    """Create a new dataset by adding samples from multiple datasets.

    List of datasets and filters are specified in a yaml config file.
    Source label fields will be renamed to a destination label field.

    Args:
        dest_name: a new dataset name
        label_field: a new label field
        cfg: path to yaml config
        persistent: whether to persist destination dataset (False for testing)
        overwrite: if dataset exists, overwrite it

    Returns:
        a dataset instance
    """
    cfg = read_yaml(cfg)
    assert "datasets" in cfg and isinstance(dataset_cfg := cfg["datasets"], list)
    assert len(dataset_cfg) > 0

    dataset = create_fiftyone_dataset(
        dest_name, src=None, persistent=persistent, overwrite=overwrite
    )
    for one in dataset_cfg:
        assert "name" in one and isinstance(one["name"], str)
        if "filters" in one:
            assert isinstance(one["filters"], dict)
        else:
            one["filters"] = {}
        assert "label_field" in one and isinstance(one["label_field"], str)

        temp_name = f"{dest_name}_{one['name']}"
        temp = load_fiftyone_dataset(one["name"], **one["filters"]).clone(
            name=temp_name, persistent=False
        )
        temp.clone_sample_field(one["label_field"], label_field)
        if "tags" in one:
            temp.tag_samples(one["tags"])

        dataset.add_samples(temp.select_fields([label_field, "tags"]))
        fo.delete_dataset(temp_name)

    return dataset


def fix_filepath(src: str, from_dir: str, to_dir: str) -> None:
    """Replace from_dir part to to_dir in each sample's filepath in samples.json file.

    Samples.json file is updated inplace.

    Args:
        src: sample.json file export for fiftyone.types.FiftyOneDataset
        from_dir: relative directory to replace
        to_dir: new relative directory
    """
    # TODO test this
    src = Path(src)
    assert src.exists() and src.suffix == ".json"
    samples = read_json(src)

    to_dir = Path(to_dir)

    def fix_path(path):
        return str(to_dir / Path(path).relative_to(from_dir))

    for smp in samples["samples"]:
        smp["filepath"] = fix_path(smp["filepath"])

    write_json(samples, src)


def transpose_images(dataset: str, **kwargs) -> fo.DatasetView:
    """Rotate images 90 degrees.

    Args:
        dataset: fiftyone dataset name
        **kwargs: dataset loading filters

    Returns:
        a dataset view instance
    """
    assert len(kwargs) > 0, "Danger: provide dataset filters"

    dataset = load_fiftyone_dataset(dataset, **kwargs)

    for smp in tqdm(dataset.select_fields("filepath"), desc="transposing"):
        Image.open(smp.filepath).transpose(Image.ROTATE_90).save(smp.filepath)

    return dataset
