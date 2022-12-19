"""Display various data about datasets.
"""
from finegrained.utils.dataset import load_fiftyone_dataset, get_unique_labels


def print_labels(dataset: str, label_field: str) -> None:
    """Print all classes in the dataset.

    Args:
        dataset: fiftyone dataset name
        label_field: field that contains labels

    Returns:
        none
    """
    dataset = load_fiftyone_dataset(dataset)
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
