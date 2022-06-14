"""Some dataset transforms.
"""
import fiftyone as fo
from fiftyone.types import ImageClassificationDirectoryTree

from .utils import load_fiftyone_dataset, create_fiftyone_dataset
from ..utils.general import LIST_STR, parse_list_str


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
    dataset = load_fiftyone_dataset(dataset, **kwargs)
    _export_patches(dataset, label_field, export_dir)
    create_fiftyone_dataset(
        to_name, export_dir, ImageClassificationDirectoryTree, overwrite
    )


def tag_samples(dataset: str, tags: LIST_STR, **kwargs):
    dataset = load_fiftyone_dataset(dataset, **kwargs)
    dataset.tag_samples(parse_list_str(tags))
