import pytest

from finegrained.data import tag
from finegrained.utils.dataset import get_unique_labels


@pytest.fixture(scope="module")
def labels_txt(tmpdir_factory):
    labels = tmpdir_factory.mktemp("labels").join("labels.txt")
    labels.write("\n".join(["airplane", "car", "dog"]))
    return labels


def test_tag_samples(temp_dataset):
    tag_name = "new_tag"
    tag_counts = tag.tag_samples(temp_dataset.name, tag_name)

    assert tag_name in tag_counts
    assert tag_counts[tag_name] == len(temp_dataset)
    assert "not_existing_tag" not in tag_counts


def test_tag_labels(temp_dataset, labels_txt):
    tag_name = "new_tag"

    for label in ["airplane", labels_txt]:
        tag_counts = tag.tag_labels(temp_dataset.name, "ground_truth", label, tag_name)

        assert tag_name in tag_counts
        assert tag_counts[tag_name] > 0
        assert "not_existing_tag" not in tag_counts


@pytest.mark.parametrize(
    "splits", [{"rain": 0.65, "text": 0.35}, {"a": 0.5, "b": 0.5, "c": 0.5}]
)
def test_split_dataset(temp_dataset, splits):
    tag_counts = tag.split_dataset(temp_dataset.name, splits)

    data_len = len(temp_dataset)
    total = sum(list(splits.values()))
    for k, v in splits.items():
        assert v / total * data_len - tag_counts[k] <= 1


@pytest.mark.parametrize("vertical", [True, False])
def test_alignment(temp_dataset, vertical):
    tag_counts = tag.tag_alignment(temp_dataset.name, vertical=vertical)

    aligment_tag = "vertical" if vertical else "horizontal"
    assert aligment_tag in tag_counts
    assert tag_counts[aligment_tag] > 0


def test_split_classes(temp_dataset):
    tag.split_classes(
        temp_dataset.name,
        "resnet50",
        train_size=0.6,
        val_size=0.4,
        min_samples=2,
    )

    train_labels = get_unique_labels(temp_dataset.match_tags("train"), "resnet50")
    val_labels = get_unique_labels(temp_dataset.match_tags("val"), "resnet50")
    num_intersect = len(set(train_labels).intersection(val_labels))
    assert num_intersect == 0

    label_counts = temp_dataset.match_tags("train").count_values("resnet50.label")
    assert min(label_counts.values()) >= 2

    label_counts = temp_dataset.match_tags("val").count_values("resnet50.label")
    assert min(label_counts.values()) >= 2
