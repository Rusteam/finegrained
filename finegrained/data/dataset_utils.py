import fiftyone as fo
from fiftyone import ViewField as F
from fiftyone.types import ImageClassificationDirectoryTree

from finegrained.utils import types
from finegrained.utils.general import parse_list_str
from finegrained.utils.types import LIST_STR


def load_fiftyone_dataset(
    dataset: str,
    include_labels: dict[str, LIST_STR] = {},
    include_tags: LIST_STR = None,
    exclude_tags: LIST_STR = None,
    max_samples: int = None,
    label_conf: float = 0.0,
    fields_exist: LIST_STR = None,
    not_exist: LIST_STR = None,
) -> fo.Dataset:
    """Load a dataset and apply view filters

    Args:
        dataset: fiftyone dataset name
        include_labels: keep samples that have fields with these values
        include_tags: keep samples that match these sample tags
        exclude_tags: exclude samples that match these sample tags
        max_samples: randomly select this number of samples if specified
        label_conf: if 'include_labels' specified, apply confidence threshold
        fields_exist: keep samples that contain these fields
        not_exist: keep samples that do Not contain these fields

    Returns:
        a fiftyone dataset
    """
    if include_tags is None:
        include_tags = []
    dataset = fo.load_dataset(dataset)
    if bool(include_tags):
        dataset = dataset.match_tags(include_tags)
    if bool(exclude_tags):
        dataset = dataset.match_tags(exclude_tags, False)
    if bool(include_labels):
        for field, values in include_labels.items():
            filter_fn = F("label").is_in(parse_list_str(values))
            if label_conf > 0:
                filter_fn = filter_fn & F("confidence") >= label_conf
            dataset = dataset.filter_labels(field, filter_fn)
    if bool(fields_exist):
        for field in parse_list_str(fields_exist):
            dataset = dataset.exists(field)
    if bool(not_exist):
        for field in parse_list_str(not_exist):
            dataset = dataset.exists(field, False)
    if bool(max_samples):
        dataset = dataset.take(min(max_samples, len(dataset)))

    assert len(dataset) > 0, "No samples left after filtering the dataset"
    return dataset


def create_fiftyone_dataset(
    name: str,
    src: str,
    dataset_type=ImageClassificationDirectoryTree,
    overwrite: bool = False,
    persistent: bool = True,
) -> fo.Dataset:
    """Create a fiftyone dataset

    Args:
        name: a name for this dataset
        src: dataset directory
        dataset_type: fiftyone dataset format
        overwrite: whether to overwrite if that name is already taken
        persistent: whether to persist this dataset in mongo

    Returns:
        a fiftyone dataset
    """
    if (exists := fo.dataset_exists(name)) and not overwrite:
        raise ValueError(
            f"Dataset {name!r} already exists!. User overwrite=True to delete it"
        )
    elif overwrite and exists:
        fo.delete_dataset(name, verbose=True)
    dataset = fo.Dataset.from_dir(
        dataset_dir=src,
        dataset_type=dataset_type,
        name=name,
    )
    if persistent:
        dataset.persistent = True
    return dataset


def get_unique_labels(dataset: fo.Dataset, label_field: str) -> list[str]:
    """Extract a list of labels from a field

    Args:
        dataset: a fiftyone dataset object
        label_field: a field to extract labels from

    Returns:
        a list of class labels
    """
    label_fields = parse_list_str(label_field)
    labels = []
    for field in label_fields:
        if not dataset.has_sample_field(field):
            raise KeyError(f"{dataset.name} does not contain {field=}")
        label_type = dataset.get_field(field).document_type
        if label_type == fo.Classification:
            view_field = f"{field}.label"
        elif label_type == fo.Detections:
            view_field = f"{field}.detections.label"
        elif label_type == fo.Polylines:
            view_field = f"{field}.polylines.label"
        else:
            raise NotImplementedError(f"Not implemented for {label_type=}.")
        labels.extend(dataset.distinct(F(view_field)))
    return labels


def get_all_filepaths(dataset: fo.Dataset) -> types.LIST_STR:
    """Retrieve all filepaths from a sample collection

    Args:
        dataset: fiftyone dataset object or sample collection

    Returns:
        a list of absolute paths
    """
    files = [
        smp.filepath
        for smp in dataset.select_fields("filepath")
    ]
    return files
