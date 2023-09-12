from pathlib import Path

import fiftyone as fo
import fiftyone.types as fot
import pandas as pd
import pytest

from finegrained.data import export


def test_to_csv(temp_dataset, tmp_path):
    export_path = tmp_path / "export.csv"
    export.to_csv(
        temp_dataset.name,
        label_field="resnet18-imagenet-torch",
        export_path=str(export_path),
        extra_fields=["id"],
    )

    assert export_path.exists()
    data = pd.read_csv(str(export_path))
    assert data.shape == (100, 3)
    assert data.columns.tolist() == ["image", "label", "id"]

    assert all([Path(p).is_absolute() for p in data["image"].tolist()])


@pytest.fixture(scope="module")
def img_clf_dir(temp_dataset, tmp_path_factory):
    img_clf_dir = tmp_path_factory.mktemp("img_clf")
    for tag in ["test", "validation"]:
        temp_dataset.match_tags("test").take(5).export(
            export_dir=str(img_clf_dir / tag),
            dataset_type=fot.ImageClassificationDirectoryTree,
            label_field="resnet18-imagenet-torch",
        )
    return img_clf_dir


@pytest.fixture(scope="module")
def img_clf_name():
    name = "unittest_img_clf_delete"
    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    yield name
    if fo.dataset_exists(name):
        fo.delete_dataset(name)


def test_img_clf_dir(tmp_path, temp_dataset, img_clf_name):
    # test export
    tags = ["test", "validation"]
    export.to_clf_dir(
        temp_dataset.name,
        export_dir=str(tmp_path),
        tags=tags,
        label_field="resnet18-imagenet-torch",
        max_samples=10,
    )

    export_counts = {}
    for tag in tags:
        assert (tmp_path / tag).exists()
        assert len(list((tmp_path / tag).glob("*"))) >= 1
        n_files = list((tmp_path / tag).rglob("*.*"))
        assert len(n_files) >= 1
        export_counts.update({tag: len(n_files)})

    # test import
    import_counts = export.from_clf_dir(
        img_clf_name,
        str(tmp_path),
        label_field="label",
    )
    for tag in tags:
        assert tag in import_counts
        assert import_counts[tag] == export_counts[tag]
