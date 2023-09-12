"""Dataset converting and exporting utils.
"""
from pathlib import Path
from typing import List, Optional

import fiftyone.types as fot

from finegrained.utils.dataset import (
    create_fiftyone_dataset,
    get_unique_labels,
    load_fiftyone_dataset,
)


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


def to_csv(
    dataset: str,
    label_field: str,
    export_path: str,
    extra_fields: Optional[list[str]] = None,
    **kwargs,
):
    """Export a dataset into CSV format for uploading to external sources.

    Args:
        dataset: fiftyone dataset name
        label_field: field that contains labels (will be mapped to 'label')
        export_path: where to write csv file
        extra_fields: extra fields to be added to csv
        **kwargs: dataset loading filters
    """
    dataset = load_fiftyone_dataset(dataset, **kwargs)
    label_field = f"{label_field}.label"
    fields = {"filepath": "image", label_field: "label"}
    if extra_fields:
        fields.update({k: k for k in extra_fields})
    dataset.export(
        dataset_type=fot.CSVDataset,
        abs_paths=True,
        export_media=False,
        labels_path=export_path,
        fields=fields,
    )


def to_clf_dir(
    dataset: str,
    export_dir: str,
    tags: list[str],
    label_field: str = "ground_truth",
    **kwargs,
) -> None:
    """Export a dataset into classification directory structure
    with tags as split names.

    Args:
        dataset: fiftyone dataset name
        export_dir: where to write data
        tags: which tags to export, tags will be used as dataset split names
        label_field: field that contains labels
        **kwargs: dataset loading filters
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_fiftyone_dataset(dataset, include_tags=tags, **kwargs)
    for tag in tags:
        subset = dataset.match_tags(tag)
        assert len(subset) > 0, f"No samples in the subset with {tag=}"
        subset.export(
            export_dir=str(export_dir / tag),
            dataset_type=fot.ImageClassificationDirectoryTree,
            label_field=label_field,
        )


def from_clf_dir(
    dataset: str,
    data_path: str,
    label_field: str = "ground_truth",
    overwrite: bool = False,
) -> dict:
    """Load a dataset from a classification directory of images and tag with split name.

    Expected folder structure:
        data_path
        ├── test
        │   ├── class1
        │   │   ├── img1.jpg
        │   │   ├── img2.jpg
        │   │   └── ...
        │   ├── class2
        │   │   ├── img1.jpg
        │   │   ├── img2.jpg
        │   │   └── ...
        │   └── ...
        ├── train
        │   ├── class1
        │   │   ├── img1.jpg
        │   │   ├── img2.jpg
        │   │   └── ...
        │   ├── class2
        │   │   ├── img1.jpg
        │   │   ├── img2.jpg
        │   │   └── ...
        │   └── ...
        |── ...

    Args:
        dataset: fiftyone dataset name
        data_path: path to the dataset
        label_field: label field name
        overwrite: whether to overwrite if that name is already taken

    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} does not exist")

    splits = [p.name for p in data_path.glob("*") if p.is_dir()]
    if not splits:
        raise ValueError(f"No splits found in {data_path}")

    dataset = create_fiftyone_dataset(
        name=dataset, src=None, overwrite=overwrite, persistent=True
    )
    for spl in splits:
        dataset.add_dir(
            dataset_dir=str(data_path / spl),
            tags=spl,
            label_field=label_field,
            dataset_type=fot.ImageClassificationDirectoryTree,
        )

    dataset.save()
    return dataset.count_sample_tags()
