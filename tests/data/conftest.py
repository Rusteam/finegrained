import fiftyone as fo
import pytest
from fiftyone import zoo as foz


@pytest.fixture(scope="module")
def temp_dataset():
    dataset = foz.load_zoo_dataset("quickstart").take(100).clone("test_transforms_temp")
    model_name = "resnet18-imagenet-torch"
    if len(dataset.exists(model_name)) < len(dataset):
        model = foz.load_zoo_model(model_name)
        dataset.exists(model_name, False).apply_model(
            model, label_field=model_name, batch_size=4
        )
    yield dataset
    if fo.dataset_exists(dataset.name):
        fo.delete_dataset(dataset.name)


@pytest.fixture(scope="module")
def temp_dataset_video():
    dataset = (
        foz.load_zoo_dataset("quickstart-video", max_samples=5)
        .take(5)
        .clone("test_transforms_video_temp")
    )
    yield dataset
    if fo.dataset_exists(dataset.name):
        fo.delete_dataset(dataset.name)


@pytest.fixture(scope="function")
def new_dataset_name_temp():
    name = "fg-unittest-temp"
    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    yield name
    if fo.dataset_exists(name):
        fo.delete_dataset(name)
