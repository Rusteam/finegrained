from copy import deepcopy

import pytest
import fiftyone as fo

from finegrained.data import annotations
from .conftest import temp_dataset

@pytest.fixture()
def temp_dataset(temp_dataset, backend_config):
    new_name = anno_key = "temp_annotations_test"
    new = temp_dataset.clone(new_name)
    yield new, new_name, anno_key
    if anno_key in new.list_annotation_runs():
        creds = deepcopy(backend_config)
        creds.pop("backend")
        results = new.load_annotation_results(anno_key, **creds)
        results.cleanup()
        new.delete_annotation_run(anno_key)
    fo.delete_dataset(new_name)


@pytest.fixture()
def backend_config():
    return dict(
        backend="cvat",
        url="http://localhost:8080",
        username="cvat",
        password="cvat",
    )


def test_annotate_samples(temp_dataset, backend_config):
    dataset, name, anno_key = temp_dataset
    target_classes = ["foo", "bar", "baz"]
    label_field = "label_test"
    annotations.annotate(dataset=name, annotation_key=anno_key, label_field=label_field, backend=backend_config,
                         dataset_kwargs={"max_samples": 5}, label_type="classification", classes=target_classes)

    results = dataset.get_annotation_info(anno_key)
    assert results.key == anno_key
    assert results.config.label_schema[label_field]['classes'] == target_classes

    new_field = "new_field"
    annotations.load(name, annotation_key=anno_key, backend=backend_config, dest_field=new_field)

    assert dataset.has_sample_field(new_field)
    assert len(dataset.exists(new_field)) == 0
