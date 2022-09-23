import pytest
import fiftyone as fo
import fiftyone.zoo as foz

from finegrained.data import zoo


@pytest.fixture(scope="module")
def temp_dataset():
    dataset = (
        foz.load_zoo_dataset("quickstart")
        .take(10)
        .clone("temp_torchvison_dataset")
    )
    yield dataset
    fo.delete_dataset(dataset.name)


def test_object_detection(temp_dataset):
    field = "temp_detection"
    zoo.object_detection(
        temp_dataset_anno.name,
        label_field=field,
        conf=0.5,
        image_size=320
    )

    assert len(temp_dataset_anno.exists(field)) == len(temp_dataset_anno)
