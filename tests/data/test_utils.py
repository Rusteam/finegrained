import random

import fiftyone.utils.random as four
import pytest

import finegrained.utils.dataset as dataset_utils
from finegrained.utils.general import find_diff


def test_load_dataset_tags(temp_dataset):
    four.random_split(temp_dataset, {"spl": 0.65, "rest": 0.35})

    dataset = dataset_utils.load_fiftyone_dataset(temp_dataset.name, include_tags="spl")
    tags = dataset.count_sample_tags()
    assert "spl" in tags
    assert "rest" not in tags

    dataset = dataset_utils.load_fiftyone_dataset(
        temp_dataset.name, exclude_tags=["spl"]
    )
    tags = dataset.count_sample_tags()
    assert "spl" not in tags
    assert "rest" in tags


def test_load_dataset_label_tags(temp_dataset):
    n = 15
    tag = "new_label_tag"
    label_field = "predictions"

    for smp in temp_dataset.take(n):
        det = smp[label_field].detections[-1]
        det.tags.append(tag)
        smp.save()

    dataset = dataset_utils.load_fiftyone_dataset(temp_dataset.name, label_tags=tag)
    tags = dataset.count_label_tags()
    assert tag in tags and tags[tag] == n
    assert len(dataset) == n


def test_load_dataset_fields(temp_dataset):
    field = "new-field"
    n = 25

    for smp in temp_dataset.take(n):
        smp[field] = random.random()
        smp.save()

    dataset = dataset_utils.load_fiftyone_dataset(temp_dataset.name, fields_exist=field)
    assert len(dataset) == n
    assert dataset.has_sample_field(field)

    dataset = dataset_utils.load_fiftyone_dataset(temp_dataset.name, not_exist=[field])
    assert len(dataset) == len(temp_dataset) - n
    assert not any(dataset.values(field))


@pytest.mark.parametrize(
    "label_field,filter_labels,label_conf",
    [
        ("predictions", ["person", "dog", "carrot", "car", "kite"], 0.0),
        (
            "resnet18-imagenet-torch",
            ["parachute", "sandbar", "flagpole", "ski"],
            0.0,
        ),
        ("predictions", "all", 0.75),
        (
            "resnet18-imagenet-torch",
            ["parachute", "sandbar", "flagpole", "ski"],
            0.05,
        ),
    ],
)
def test_load_dataset_labels(temp_dataset, label_field, filter_labels, label_conf):
    dataset = dataset_utils.load_fiftyone_dataset(
        temp_dataset.name,
        include_labels={label_field: filter_labels},
        label_conf=label_conf,
    )

    labels = dataset_utils.get_unique_labels(dataset, label_field)
    if filter_labels != "all":
        diff = find_diff(filter_labels, labels)
        assert len(diff) == 0
    else:
        assert len(labels) > 0
    if label_conf > 0:
        _, p = temp_dataset._get_label_field_path(label_field)
        conf_vals = dataset.values(f"{p}.confidence")
        conf_vals = sum(conf_vals, []) if isinstance(conf_vals[0], list) else conf_vals
        assert min(conf_vals) >= label_conf

    # test exclude labels
    if filter_labels != "all":
        dataset = dataset_utils.load_fiftyone_dataset(
            temp_dataset.name, exclude_labels={label_field: filter_labels}
        )

        labels = dataset_utils.get_unique_labels(dataset, label_field)

        diff = find_diff(filter_labels, labels)
        assert len(diff) == len(filter_labels)
