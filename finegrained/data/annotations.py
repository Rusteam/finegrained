"""Send, query and get annotation results.
"""
from pathlib import Path
from typing import Optional, Any

from .dataset_utils import load_fiftyone_dataset
from ..utils import types
from ..utils.os_utils import read_txt, read_file_config


def _load_backend_config(src: Any) -> dict:
    if isinstance(src, dict):
        config = src
    elif isinstance(src, str) and Path(src).is_file():
        # if a txt file as file
        config = read_file_config(src, section="annotation_backend")
    else:
        raise ValueError(f"{type(src)!r} type has not been implemented.")
    return config


def _parse_classes(src: Any) -> list[str]:
    if isinstance(src, list):
        return src
    elif isinstance(src, str) and Path(src).is_file():
        return read_txt(src)
    else:
        raise ValueError(
            f"Classes has to be a list of str or a path to a txt file. Received {type(src)}"
        )


def annotate(
    dataset: str,
    annotation_key: str,
    label_field: str,
    backend: Any,
    overwrite: bool = False,
    dataset_kwargs: Optional[dict] = None,
    **kwargs,
):
    """Send samples to annotations

    Args:
        dataset: fiftyone dataset with samples
        annotation_key: assign this key for annotation run
        label_field: if exists, upload labels
        backend: backend name or filepath to configs
        overwrite: overwrite existing annotation run if True
        dataset_kwargs: dataset loading filters
        **kwargs: annotation kwargs

    Returns:
        none
    """
    dataset = load_fiftyone_dataset(
        dataset, **dataset_kwargs if bool(dataset_kwargs) else {}
    )
    backend_conf = _load_backend_config(backend)
    if dataset.has_annotation_run(annotation_key) and overwrite:
        dataset.delete_annotation_run(annotation_key)
    if "classes" in kwargs:
        kwargs["classes"] = _parse_classes(kwargs["classes"])
    dataset.annotate(
        annotation_key,
        label_field=label_field,
        **backend_conf | kwargs,
    )


def load(
    dataset: str,
    annotation_key: str,
    backend: Any,
    dest_field: str = None,
    dataset_kwargs: Optional[dict] = None,
):
    """Download annotations from an annotation backend.

    Args:
        dataset: fiftyone dataset name
        annotation_key: annotation key used to send for annotations
        backend: annotation backend name or filepath with configs
        dest_field: if given, annotations will be stored in a new field
        dataset_kwargs: dataset loading filters

    Returns:
        none
    """
    backend_conf = _load_backend_config(backend)
    backend_conf.pop("backend")
    dataset = load_fiftyone_dataset(
        dataset, **dataset_kwargs if bool(dataset_kwargs) else {}
    )
    dataset.load_annotations(
        annotation_key, dest_field=dest_field, **backend_conf
    )


def list_keys(dataset: str) -> types.LIST_STR:
    """List annotation keys attributed to the dataset

    Args:
        dataset: fiftyone dataset name

    Returns:
        a list of keys
    """
    dataset = load_fiftyone_dataset(dataset)
    keys = dataset.list_annotation_runs()
    return keys
