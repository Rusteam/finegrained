from copy import deepcopy

import fiftyone as fo
import pytest

from finegrained.data import annotations


@pytest.fixture()
def temp_dataset_anno(temp_dataset, backend_config):
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


def test_annotate_new_field(temp_dataset_anno, backend_config):
    dataset, name, anno_key = temp_dataset_anno
    target_classes = ["foo", "bar", "baz"]
    label_field = "label_test"
    annotations.annotate(
        dataset=name,
        annotation_key=anno_key,
        label_field=label_field,
        label_type="classification",
        backend=backend_config,
        classes=target_classes,
        task_name="unit-test",
        segment_size=None,
        **{"max_samples": 5}
    )

    results = dataset.get_annotation_info(anno_key)
    assert results.key == anno_key
    assert results.config.label_schema[label_field]["classes"] == target_classes

    new_field = "new_field"
    annotations.load(
        name, annotation_key=anno_key, backend=backend_config, dest_field=new_field
    )

    assert dataset.has_sample_field(new_field)
    assert len(dataset.exists(new_field)) == 0
