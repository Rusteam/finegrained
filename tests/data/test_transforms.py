import shutil
from pathlib import Path

import pytest
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.types as fot

from finegrained.data import transforms, tag
from finegrained.utils.dataset import get_unique_labels
from finegrained.utils.os_utils import write_yaml


@pytest.fixture(scope="module")
def temp_dataset():
    dataset = (
        foz.load_zoo_dataset("quickstart")
        .take(100)
        .clone("test_transforms_temp")
    )
    if len(dataset.exists("resnet50")) < len(dataset):
        model = foz.load_zoo_model("resnet50")
        dataset.exists("resnet50", False).apply(
            model, label_field="resnet50", batch_size=4
        )
    yield dataset
    fo.delete_dataset(dataset.name)


@pytest.fixture(scope="function")
def to_patches_conf(tmp_path):
    path = Path(tmp_path) / "export"
    name = "test_to_patches_temp"
    yield str(path), name
    shutil.rmtree(path)
    fo.delete_dataset(name)


@pytest.mark.parametrize(
    "label_field", ["predictions", ("predictions", "resnet50")]
)
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
    existing_labels = get_unique_labels(temp_dataset, label_field)

    assert len(set(new_labels).difference(existing_labels)) == 0


def test_tag_samples(temp_dataset):
    tag.tag_samples(temp_dataset.name, "new_tag")

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


@pytest.mark.parametrize(
    "splits", [{"rain": 0.65, "text": 0.35}, {"a": 0.5, "b": 0.5, "c": 0.5}]
)
def test_split_dataset(temp_dataset, splits):
    tag_counts = tag.split_dataset(temp_dataset.name, splits)

    data_len = len(temp_dataset)
    total = sum(list(splits.values()))
    for k, v in splits.items():
        assert v / total * data_len - tag_counts[k] <= 1


def test_prefix_label(temp_dataset):
    dataset = transforms.prefix_label(
        temp_dataset.name,
        label_field="resnet50",
        dest_field="with_prefix",
        prefix="new",
    )

    for smp in dataset.select_fields("with_prefix"):
        assert smp["with_prefix"].label.startswith("new")


def test_is_vertical(temp_dataset):
    tag_counts = tag.tag_vertical(temp_dataset.name, tag="vert")

    assert "vert" in tag_counts
    assert tag_counts["vert"] > 0


def test_split_classes(temp_dataset):
    tag.split_classes(
        temp_dataset.name,
        "resnet50",
        train_size=0.6,
        val_size=0.4,
        min_samples=2,
    )

    train_labels = get_unique_labels(
        temp_dataset.match_tags("train"), "resnet50"
    )
    val_labels = get_unique_labels(temp_dataset.match_tags("val"), "resnet50")
    num_intersect = len(set(train_labels).intersection(val_labels))
    assert num_intersect == 0

    label_counts = temp_dataset.match_tags("train").count_values(
        "resnet50.label"
    )
    assert min(label_counts.values()) >= 2

    label_counts = temp_dataset.match_tags("val").count_values(
        "resnet50.label"
    )
    assert min(label_counts.values()) >= 2


def test_merge_diff(temp_dataset, tmp_path):
    export_dir = tmp_path / "export"

    N = len(temp_dataset)
    n = 10
    subset = temp_dataset.take(n)

    # merge all new samples
    subset.export(export_dir=str(export_dir), dataset_type=fot.ImageDirectory)

    transforms.merge_diff(
        temp_dataset.name, image_dir=export_dir, tags=["merged_test"]
    )
    # extra n samples added
    assert len(temp_dataset) == N + n
    assert len(temp_dataset.match_tags("merged_test")) == n
    # original label fields are not deleted
    assert len(temp_dataset.exists("resnet50")) == N
    assert len(temp_dataset.exists("ground_truth")) == N

    # half new, half existing
    for i, smp in enumerate(subset.take(n // 2).select_fields("filepath")):
        src_file = Path(smp.filepath)
        dest_file = export_dir / src_file.name
        # change name to be considered as a new sample
        dest_file = dest_file.with_stem(dest_file.stem + "_new")
        shutil.copy(src_file, dest_file)

    transforms.merge_diff(
        temp_dataset.name, export_dir, tags=["merged_test_2"]
    )
    # extract n//2 samples added
    assert len(temp_dataset) == N + n + n // 2
    assert len(temp_dataset.match_tags("merged_test")) == n
    assert len(temp_dataset.match_tags("merged_test_2")) == n // 2
    # original samples are unchanged
    assert len(temp_dataset.exists("resnet50")) == N
    assert len(temp_dataset.exists("ground_truth")) == N


def test_delete_samples(temp_dataset, tmp_path):
    export_dir = tmp_path / "test_delete"
    temp_dataset.export(
        str(export_dir),
        dataset_type=fot.ImageClassificationDirectoryTree,
        label_field="resnet50",
    )
    dataset = fo.Dataset.from_dir(
        str(export_dir),
        dataset_type=fot.ImageClassificationDirectoryTree,
        label_field="ground_truth",
    )
    N = len(dataset)

    transforms.delete_samples(
        dataset.name, include_labels={"ground_truth": ["ski", "zebra"]}
    )

    assert len(dataset) < N
    assert len(list(Path(export_dir).glob("*.*"))) < N
    values = dataset.count_values("ground_truth.label")
    assert len(values) > 0
    assert "ski" not in values
    assert "zebra" not in values


@pytest.mark.parametrize("to_field,mapping", [
    ("new_labels", dict(
        carrot="veg", car="transport", boat="transport", broccoli="veg"
    )),
    ("same_field", {}),
])
def test_map_labels(temp_dataset, to_field, mapping):
    transforms.map_labels(
        dataset=temp_dataset.name,
        from_field="predictions",
        to_field=to_field,
        label_mapping=mapping,
    )

    assert temp_dataset.has_sample_field("predictions")
    assert temp_dataset.has_sample_field(to_field)

    uniq_src = get_unique_labels(temp_dataset, "predictions")
    uniq_dest = get_unique_labels(temp_dataset, to_field)

    if bool(mapping):
        assert all([k in uniq_src for k in mapping.keys()])
        assert all([k not in uniq_src for k in mapping.values()])

        assert all([k in uniq_dest for k in mapping.values()])
        assert all([k not in uniq_dest for k in mapping.keys()])
    else:
        assert uniq_src == uniq_dest


def test_from_labels(temp_dataset):
    new_field = "test_from_fields"
    transforms.map_labels(
        dataset=temp_dataset.name,
        from_field="predictions",
        to_field=new_field,
        label_mapping=None
    )

    transforms.from_labels(
        dataset=temp_dataset.name,
        label_field=new_field,
        from_field="resnet50"
    )

    expected_labels = get_unique_labels(temp_dataset, "resnet50")
    actual_labels = get_unique_labels(temp_dataset, new_field)

    assert all([l in expected_labels for l in actual_labels])



@pytest.fixture(scope="function")
def combine_cfg_path(tmp_path):
    cfg = dict(
        datasets=[
            dict(
                name="quickstart",
                filters=dict(
                    include_tags="test"
                ),
                label_field="ground_truth",
                tag="test"
            ),
            dict(
                name="quickstart",
                filters=dict(
                    include_labels=dict(ground_truth=["broccoli", "cake", "car", "cell phone", "chair", "clock"])
                ),
                label_field="ground_truth",
                tag="train",
            ),
            dict(
                name="quickstart",
                filters=dict(
                    include_labels=dict(ground_truth=["bear", "bed", "banana"]),
                ),
                label_field="ground_truth",
                tag="val",
            ),
        ]
    )
    write_path = tmp_path / "cfg.yaml"
    write_yaml(cfg, write_path)
    return write_path


def test_combine_datasets(combine_cfg_path):
    new_name = "combo_test_delete"
    label_field = "combo_field"
    dataset = transforms.combine_datasets(new_name, label_field,
                                          combine_cfg_path, persistent=False)

    tags = dataset.count_sample_tags()
    assert all([x in tags for x in ["train", "val", "test"]])

    assert dataset.has_sample_field(label_field)
