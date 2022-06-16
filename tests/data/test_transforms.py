import shutil

import pytest
import fiftyone as fo

from finegrained.data import transforms
from finegrained.data.dataset_utils import get_unique_labels


COCO_NAME = "coco-2017-100"


@pytest.fixture()
def coco_dataset():
    assert COCO_NAME in fo.list_datasets()
    return fo.load_dataset(COCO_NAME)


@pytest.fixture()
def coco_clone_temp(coco_dataset):
    name = "coco_clone_test"
    new = coco_dataset.clone(name)
    yield new, name
    fo.delete_dataset(name)


@pytest.fixture()
def temp_dataset_name():
    name = "temp_dataset_test"
    yield name
    fo.delete_dataset(name)


@ pytest.fixture()
def temp_export_dir(tmp_path):
    yield tmp_path
    shutil.rmtree(tmp_path)


def test_to_patches(temp_export_dir, temp_dataset_name, coco_dataset):
    params = dict(
        dataset=COCO_NAME,
        label_field="ground_truth",
        to_name=temp_dataset_name,
        export_dir=str(temp_export_dir),
        overwrite=False,
        max_samples=5,
    )
    transforms.to_patches(**params)

    assert temp_dataset_name in fo.list_datasets()

    new = fo.load_dataset(temp_dataset_name)
    new_labels = get_unique_labels(new, "ground_truth")
    existing_labels = get_unique_labels(coco_dataset, "ground_truth")

    assert len(set(new_labels).difference(existing_labels)) == 0


def test_tag_samples(coco_clone_temp):
    dataset, name = coco_clone_temp
    transforms.tag_samples(name, "new_tag")
    transforms.tag_samples(name, ["two", "four"])

    for tag in ["new_tag", "two", "four"]:
        tagged = dataset.match_tags(tag)
        assert len(tagged) == len(dataset)


def test_delete_field(coco_clone_temp):
    dataset, name = coco_clone_temp
    dataset.clone_sample_field("ground_truth", "new")
    dataset.clone_sample_field("new", "new_clone")
    dataset.clone_sample_field("new", "last_clone")

    transforms.delete_field(name, "last_clone")
    transforms.delete_field(name, ["new", "new_clone"])
    for field in ["new", "new_clone", "last_clone"]:
        assert not dataset.has_sample_field(field)


@pytest.mark.parametrize("splits", [{"rain": 0.65, "text": 0.35},
                                    {"a": 0.5, "b": 0.5, "c": 0.5}])
def test_split_dataset(coco_clone_temp, splits):
    dataset, name = coco_clone_temp
    tag_counts = transforms.split_dataset(name, splits)

    data_len = len(dataset)
    total = sum(list(splits.values()))
    for k, v in splits.items():
        assert v / total * data_len == tag_counts[k]



