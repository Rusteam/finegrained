"""Display various data about datasets.
"""
from finegrained.data.transforms import delete_field
from finegrained.utils import types
from finegrained.utils.dataset import get_unique_labels, load_fiftyone_dataset
from finegrained.utils.general import find_diff


def print_labels(dataset: str, label_field: str, **kwargs) -> None:
    """Print all classes in the dataset.

    Args:
        dataset: fiftyone dataset name
        label_field: field that contains labels
        **kwargs: dataset loading filters
    """
    dataset = load_fiftyone_dataset(dataset, **kwargs)
    labels = get_unique_labels(dataset, label_field)
    print("\n".join(labels))


def classification_report(
    dataset: str,
    predictions: str,
    gt_field: str = "ground_truth",
    cmat: bool = False,
    **kwargs,
):
    """Print classification report.

    Args:
        dataset: fiftyone dataset name
        predictions: a field with predictions
        gt_field: a field with ground truth labels
        cmat: if True, plot a confusion matrix
        **kwargs: dataset loading filters

    Returns:
        none
    """
    dataset = load_fiftyone_dataset(dataset, **kwargs)
    labels = dataset.distinct(f"{gt_field}.label")
    results = dataset.evaluate_classifications(
        predictions, gt_field=gt_field, classes=labels
    )
    results.print_report()
    if cmat:
        cm = results.plot_confusion_matrix(backend="matplotlib")
        cm.show()


def label_diff(
    dataset: str,
    label_field: str,
    tags_left: types.LIST_STR_STR,
    tags_right: types.LIST_STR_STR,
):
    """Compute difference between two sets of labels.

    Args:
        dataset: fiftyone dataset name
        label_field: field with labels
        tags_left: list of tags for base list of labels
        tags_right: list of tags for intersection comparison
    """
    # TODO test this
    dataset = load_fiftyone_dataset(dataset)
    assert dataset.has_sample_field(label_field)
    assert len(tags_left) > 0
    assert len(tags_right) > 0

    left_labels = get_unique_labels(dataset.match_tags(tags_left), label_field)
    right_labels = get_unique_labels(dataset.match_tags(tags_right), label_field)

    diff = find_diff(left_labels, right_labels)
    return diff


def compute_area(
    dataset: str,
    field: str = "area",
    average_size: bool = False,
    overwrite_metadata: bool = False,
    overwrite: bool = False,
    **kwargs,
) -> tuple[int, int]:
    """Calculate area of an image based on metadata

    Args:
        dataset: fiftyone dataset name
        field: field where to assign area values
        average_size: if True, calculate (width + height)/2 instead
        overwrite_metadata: whether to overwrite metadata
        overwrite: delete field if already exists
        **kwargs: dataset loading filters

    Returns:
        area bounds
    """
    dataset = load_fiftyone_dataset(dataset, **kwargs)
    if dataset.has_sample_field(field):
        if overwrite:
            delete_field(dataset.name, field)
        else:
            raise ValueError(f"{field=} already exists.")

    dataset.compute_metadata(overwrite=overwrite_metadata)
    for smp in dataset.select_fields("metadata"):
        val = (
            (smp.metadata.width + smp.metadata.height) / 2
            if average_size
            else smp.metadata.width * smp.metadata.height
        )
        smp[field] = val
        smp.save()

    return dataset.bounds(field)
