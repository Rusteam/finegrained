"""Dataset converting and exporting utils.
"""
from typing import List

import fiftyone.types as fot

from finegrained.utils.dataset import get_unique_labels, load_fiftyone_dataset


def to_yolov5(
    dataset: str,
    label_field: str,
    export_dir: str,
    splits: List[str],
    **kwargs,
):
    """Export a dataset into yolov5 format for training

    Args:
        dataset: fiftyone dataset name
        label_field: field that contains labels
        export_dir: where to write data
        splits: which splits to export
        **kwargs: dataset loading filters
    """
    dataset = load_fiftyone_dataset(dataset, **kwargs)
    labels = get_unique_labels(dataset, label_field)
    for tag in splits:
        subset = dataset.match_tags(tag)
        assert len(subset) > 0, f"No samples in the subset with {tag=}"
        subset.export(
            export_dir=export_dir,
            split=tag,
            dataset_type=fot.YOLOv5Dataset,
            label_field=label_field,
            classes=labels,
        )


def to_cvat(dataset: str, label_field: str, export_dir: str, **kwargs):
    """Export a dataset into CVAT format for annotation.

    Args:
        dataset: fiftyone dataset name
        label_field: field that contains labels
        export_dir: where to write data
        **kwargs: dataset loading filters
    """
    dataset = load_fiftyone_dataset(dataset, **kwargs)
    dataset.export(
        export_dir=export_dir,
        dataset_type=fot.CVATImageDataset,
        label_field=label_field,
    )
