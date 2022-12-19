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
