import shutil
from pathlib import Path

import pytest
import fiftyone as fo
import fiftyone.types as fot
import fiftyone.utils.random as four
from PIL import Image

from finegrained.data import transforms
from finegrained.utils.dataset import get_unique_labels
from finegrained.utils.os_utils import write_yaml


@pytest.fixture(scope="function")
def to_patches_conf(tmp_path):
    path = Path(tmp_path) / "export"
    name = "test_to_patches_temp"
    yield str(path), name
    shutil.rmtree(path)
    if fo.dataset_exists(name):
        fo.delete_dataset(name)


@pytest.mark.parametrize(
    "label_field,splits",
    [
        ("predictions", None),
        (("predictions", "resnet18-imagenet-torch"), None),
        ("predictions", ["train_subset_del", "test_subset_del"]),
    ],
)
def test_to_patches(to_patches_conf, temp_dataset, label_field, splits):
    export_dir, name = to_patches_conf
    if splits:
        split_fracs = {s: 1 / len(splits) for s in splits}
        four.random_split(temp_dataset, split_fracs)

    params = dict(
        dataset=temp_dataset.name,
        label_field=label_field,
        to_name=name,
        export_dir=export_dir,
        splits=splits,
        overwrite=False,
    )
    new = transforms.to_patches(**params)

    new_labels = get_unique_labels(new, "ground_truth")
    existing_labels = get_unique_labels(temp_dataset, label_field)

    assert len(set(new_labels).difference(existing_labels)) == 0

    tag_counts = new.count_sample_tags()
    if splits:
        assert all([s in tag_counts for s in splits])


def test_delete_field(temp_dataset):
    temp_dataset.clone_sample_field("ground_truth", "new")
    temp_dataset.clone_sample_field("new", "new_clone")
    temp_dataset.clone_sample_field("new", "last_clone")

    transforms.delete_field(temp_dataset.name, "last_clone")
    transforms.delete_field(temp_dataset.name, ["new", "new_clone"])
    for field in ["new", "new_clone", "last_clone"]:
        assert not temp_dataset.has_sample_field(field)


def test_prefix_label(temp_dataset):
    dataset = transforms.prefix_label(
        temp_dataset.name,
        label_field="resnet50",
        dest_field="with_prefix",
        prefix="new",
    )

    for smp in dataset.select_fields("with_prefix"):
        assert smp["with_prefix"].label.startswith("new")


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


@pytest.mark.parametrize(
    "to_field,mapping",
    [
        (
            "new_labels",
            dict(
                carrot="veg", car="transport", boat="transport", broccoli="veg"
            ),
        ),
        ("same_field", {}),
    ],
)
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
    from_field = "resnet18-imagenet-torch"
    transforms.map_labels(
        dataset=temp_dataset.name,
        from_field="predictions",
        to_field=new_field,
        label_mapping=None,
    )

    transforms.from_labels(
        dataset=temp_dataset.name, label_field=new_field, from_field=from_field
    )

    expected_labels = get_unique_labels(temp_dataset, from_field)
    actual_labels = get_unique_labels(temp_dataset, new_field)

    assert all([l in expected_labels for l in actual_labels])


def test_from_label_tag(temp_dataset):
    label_tag = "test_label_tag"
    label_field = "predictions"

    no_tag, to_tag = four.random_split(temp_dataset, [0.6, 0.4])
    for smp in to_tag:
        det = smp[label_field].detections[0]
        det.tags.append(label_tag)
        smp.save()
    # to_tag.tag_labels(label_tag)

    label_values = transforms.from_label_tag(
        dataset=temp_dataset.name,
        label_field=label_field,
        label_tag=label_tag
    )

    assert label_tag in label_values, f"{label_tag} not in {label_values}"
    assert len(label_values) > 1


@pytest.fixture(scope="function")
def combine_cfg_path(tmp_path):
    cfg = dict(
        datasets=[
            dict(
                name="quickstart",
                filters=dict(include_tags="test"),
                label_field="ground_truth",
                tags="test",
            ),
            dict(
                name="quickstart",
                filters=dict(
                    include_labels=dict(
                        ground_truth=[
                            "broccoli",
                            "cake",
                            "car",
                            "cell phone",
                            "chair",
                            "clock",
                        ]
                    )
                ),
                label_field="ground_truth",
                tags=["train", "quick"],
            ),
            dict(
                name="quickstart",
                filters=dict(
                    include_labels=dict(
                        ground_truth=["bear", "bed", "banana"]
                    ),
                ),
                label_field="ground_truth",
                tags=["val", "quick"],
            ),
        ]
    )
    write_path = tmp_path / "cfg.yaml"
    write_yaml(cfg, write_path)
    return write_path


def test_combine_datasets(combine_cfg_path):
    new_name = "combo_test_delete"
    label_field = "combo_field"
    dataset = transforms.combine_datasets(
        new_name, label_field, combine_cfg_path, persistent=False
    )

    tags = dataset.count_sample_tags()
    assert all([x in tags for x in ["train", "val", "test", "quick"]])
    assert tags["train"] < tags["quick"] < len(dataset)

    assert dataset.has_sample_field(label_field)


def test_transpose_images(temp_dataset, tmp_path):
    dataset_dir = str(tmp_path / "transpose")
    dataset_type = fo.types.ImageClassificationDirectoryTree
    temp_dataset.take(2).export(
        export_dir=dataset_dir,
        dataset_type=dataset_type,
        label_field="ground_truth",
    )

    new = fo.Dataset.from_dir(
        dataset_dir=dataset_dir, dataset_type=dataset_type
    )
    new.persistent = False

    sizes = [Image.open(p).size for p in new.values("filepath")]
    transforms.transpose_images(new.name, label_conf=0.5)

    for p, size in zip(new.values("filepath"), sizes):
        new_size = Image.open(p).size
        assert new_size[1] == size[0]
        assert new_size[0] == size[1]
