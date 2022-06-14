import shutil

import pytest
import fiftyone as fo

from finegrained.data import transforms
from finegrained.data.utils import get_unique_labels


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
    dataset, name = coco_dataset
    transforms.tag_samples(name, )
