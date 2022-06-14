import pytest
import fiftyone as fo

from finegrained.data import utils


@pytest.fixture()
def fake_dataset():
    new = fo.load_dataset("cavist_KB_bottles").clone("test_load_dataset")
    yield new
    fo.delete_dataset(new.name)


# TODO change this
def test_load_exists(fake_dataset):
    dataset = utils.load_fiftyone_dataset(fake_dataset.name, fields_exist=['cavist_class_id'])
    assert len(dataset) == 115

    dataset = utils.load_fiftyone_dataset(fake_dataset.name, fields_exist='cavist_class_id_polyline')
    assert len(dataset) == 133

    dataset = utils.load_fiftyone_dataset(fake_dataset.name, not_exist=["cavist_class_id", "cavist_class_id_polyline",
                                                                        "cavist_class_id_detection"])
    assert len(dataset) == 2008 - 115 - 133 - 1


def test_load_labels(fake_dataset):
    dataset = utils.load_fiftyone_dataset(fake_dataset.name,
                                          include_labels={"ground_truth": ["bottle"]})

    assert len(dataset) == 1818

    dataset = utils.load_fiftyone_dataset(fake_dataset.name,
                                          include_labels={"ground_truth":
                                                              ["book", "cup", "motorcycle"]})

    assert len(dataset) == 73


def test_load_tags(fake_dataset):
    dataset = utils.load_fiftyone_dataset(fake_dataset.name,
                                          include_tags='cvat')
    assert len(dataset) == 249

    dataset = utils.load_fiftyone_dataset(fake_dataset.name,
                                          exclude_tags='cvat')
    assert len(dataset) == 2008 - 249
