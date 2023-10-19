"""Send, query and get annotation results.
"""
from pathlib import Path
from typing import Any, Optional

import fiftyone as fo
import fiftyone.utils.cvat as fouc

from finegrained.utils import types
from finegrained.utils.dataset import load_fiftyone_dataset
from finegrained.utils.os_utils import read_file_config, read_txt


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
            "Classes has to be a list of str or a path to a txt file."
            f"Received {type(src)}"
        )


def annotate(
    dataset: str,
    annotation_key: str,
    backend: Any,
    label_field: Optional[str] = None,
    overwrite: bool = False,
    label_type: Optional[str] = None,
    project_id: Optional[int] = None,
    segment_size: int = 10,
    task_name: Optional[str] = None,
    image_quality: int = 75,
    task_asignee: Optional[str] = None,
    organization: Optional[str] = None,
    classes: Optional[str] = None,
    frame_start: Optional[int] = None,
    frame_stop: Optional[int] = None,
    frame_step: Optional[int] = None,
    **kwargs,
):
    """Send samples to annotations

    Args:
        dataset: fiftyone dataset with samples
        annotation_key: assign this key for annotation run
        label_field: if exists, upload labels
        label_type: if label_field does not exist, this has to be specified
        backend: backend name or filepath to configs
        overwrite: overwrite existing annotation run if True
        classes: list of classes or path to labels.txt file
        image_quality: image upload quality
        task_name: custom task name, by default dataset name + annotation key
        segment_size: number of frames/images per one job
        project_id: which cvat project to connect to
        task_asignee: assignee for the task
        organization: cvat organization name
        frame_start: start frame index for all videos
        frame_stop: stop frame index for all videos
        frame_step: step frame index for all videos
        **kwargs: dataset loading filters
    """
    dataset = load_fiftyone_dataset(dataset, **kwargs)
    backend_conf = _load_backend_config(backend)
    if dataset.has_annotation_run(annotation_key) and overwrite:
        dataset.delete_annotation_run(annotation_key)
    if label_field and not dataset.has_sample_field(label_field) and label_type is None:
        raise ValueError(
            f"{label_field=} does not exist in {dataset.name}. Specify 'label_type'"
        )
    if classes:
        classes = _parse_classes(classes)
    if task_name is None:
        dataset_name = getattr(dataset, "dataset_name", dataset.name)
        task_name = dataset_name + " - " + annotation_key
    dataset.annotate(
        annotation_key,
        label_field=label_field,
        label_type=label_type,
        project_id=project_id,
        segment_size=segment_size,
        task_name=task_name,
        image_quality=image_quality,
        classes=classes,
        task_asignee=task_asignee,
        organization=organization,
        frame_start=frame_start,
        frame_stop=frame_stop,
        frame_step=frame_step,
        **backend_conf,
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
    dataset.load_annotations(annotation_key, dest_field=dest_field, **backend_conf)


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


def delete_key(dataset: str, key: str):
    """Delete an annotation key.

    Args:
        dataset: fiftyone dataset name
        key: annotation key

    Returns:
        none
    """
    dataset = load_fiftyone_dataset(dataset)
    dataset.delete_annotation_run(key)


def print_status(dataset: str, key: str, backend: str) -> None:
    dataset = load_fiftyone_dataset(dataset)
    assert dataset.has_annotation_run(key)

    backend_conf = _load_backend_config(backend)
    backend_conf.pop("backend")

    results = dataset.load_annotation_results(key, **backend_conf)
    results.print_status()


def _from_cvat_box(box: fouc.CVATImageBox, img_w: int, img_h: int) -> fo.Detection:
    """Convert CVAT box to fiftyone detection"""
    xmin, ymin, xmax, ymax = box.xtl, box.ytl, box.xbr, box.ybr
    return fo.Detection(
        label=box.label,
        bounding_box=[
            xmin / img_w,
            ymin / img_h,
            (xmax - xmin) / img_w,
            (ymax - ymin) / img_h,
        ],
    )


def from_cvat_annotations_file(
    dataset: str, annotations_file: str, label_field: str = "ground_truth"
) -> dict[str, int]:
    """Load annotations from CVAT annotations file export.

    Args:
        dataset: fiftyone dataset name
        annotations_file: path to CVAT annotations file
        label_field: which field to store annotations
    """
    info, task_labels, images = fouc.load_cvat_image_annotations(annotations_file)
    dataset = fo.load_dataset(dataset)
    for img in images:
        fname = img.name.split("_", 1)[1]
        sample = dataset.match(fo.ViewField("filepath").ends_with(fname)).first()
        sample[label_field] = fo.Detections(
            detections=[_from_cvat_box(box, img.width, img.height) for box in img.boxes]
        )
        sample.save()

    return dataset.count_values(f"{label_field}.detections.label")
