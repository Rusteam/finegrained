import fiftyone as fo
from fiftyone import ViewField as F
from fiftyone.types import ImageClassificationDirectoryTree

from finegrained.utils.general import parse_list_str, LIST_STR


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
    return dataset


def create_fiftyone_dataset(
    name: str,
    src: str,
    dataset_type=ImageClassificationDirectoryTree,
    overwrite: bool = False,
    persistent: bool = True,
):
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
    for smp in dataset.select_fields(label_field):
        break
    if isinstance(smp[label_field], fo.Detections):
        field = f"{label_field}.detections.label"
    elif isinstance(smp[label_field], fo.Classification):
        field = f"{label_field}.label"
    else:
        raise NotImplementedError(
            f"Not implemented for type {type(smp[label_field])}."
        )
    labels = dataset.distinct(F(field))
    return labels
