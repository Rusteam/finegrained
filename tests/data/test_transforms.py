import shutil
from pathlib import Path

import pytest
import fiftyone as fo
import fiftyone.zoo as foz

from finegrained.data import transforms
from finegrained.data.dataset_utils import get_unique_labels


@pytest.fixture(scope="module")
def temp_dataset():
    dataset = foz.load_zoo_dataset("quickstart").take(15) \
        .clone('test_transforms_temp')
    if not dataset.has_sample_field("resnet50"):
        model = foz.load_zoo_model("resnet50")
        dataset.apply(model, label_field="resnet50", batch_size=4)
    yield dataset
    fo.delete_dataset(dataset.name)


@pytest.fixture(scope="function")
def to_patches_conf(tmp_path):
    path = Path(tmp_path) / "export"
    name = "test_to_patches_temp"
    yield str(path), name
    shutil.rmtree(path)
    fo.delete_dataset(name)


@pytest.mark.parametrize("label_field", ["predictions",
                                         ("predictions", "resnet50")])
def test_to_patches(to_patches_conf, temp_dataset, label_field):
    export_dir, name = to_patches_conf
    params = dict(
        dataset=temp_dataset.name,
        label_field=label_field,
        to_name=name,
        export_dir=export_dir,
        overwrite=False,
    )
    new = transforms.to_patches(**params)

    new_labels = get_unique_labels(new, "ground_truth")
    existing_labels = get_unique_labels(temp_dataset,
                                        label_field)

    assert len(set(new_labels).difference(existing_labels)) == 0


def test_tag_samples(temp_dataset):
    transforms.tag_samples(temp_dataset.name, "new_tag")

    for tag in ["new_tag", "two", "four"]:
        tagged = temp_dataset.match_tags(tag)
        assert len(tagged) == len(temp_dataset)


def test_delete_field(temp_dataset):
    temp_dataset.clone_sample_field("ground_truth", "new")
    temp_dataset.clone_sample_field("new", "new_clone")
    temp_dataset.clone_sample_field("new", "last_clone")

    transforms.delete_field(temp_dataset.name, "last_clone")
    transforms.delete_field(temp_dataset.name, ["new", "new_clone"])
    for field in ["new", "new_clone", "last_clone"]:
        assert not temp_dataset.has_sample_field(field)


@pytest.mark.parametrize("splits", [{"rain": 0.65, "text": 0.35},
                                    {"a": 0.5, "b": 0.5, "c": 0.5}])
def test_split_dataset(temp_dataset, splits):
    tag_counts = transforms.split_dataset(temp_dataset.name, splits)

    data_len = len(temp_dataset)
    total = sum(list(splits.values()))
    for k, v in splits.items():
        assert v / total * data_len - tag_counts[k] <= 1


def test_prefix_label(temp_dataset):
    dataset = transforms.prefix_label(temp_dataset.name,
                                      label_field="resnet50",
                                      dest_field="with_prefix",
                                      prefix="new")

    for smp in dataset.select_fields("with_prefix"):
        assert smp["with_prefix"].label.startswith("new")


def test_is_vertical(temp_dataset):
    tag_counts = transforms.tag_vertical(temp_dataset.name,
                                         tag="vert")

    assert "vert" in tag_counts
    assert tag_counts["vert"] > 0
