import pytest

import finegrained.data
from finegrained.data import display


def test_print_labels(temp_dataset, capsys):
    display.print_labels(temp_dataset.name, "predictions")
    captured = capsys.readouterr()
    labels = captured.out.strip().splitlines()
    assert "airplane" in labels
    assert len(labels) == 77


@pytest.mark.parametrize(
    "pred_field,gt_field,check_fields",
    [
        (
            "resnet18-imagenet-torch",
            "resnet18-imagenet-torch",
            "precision,recall,accuracy".split(","),
        ),
        ("predictions", "ground_truth", "f1-score,micro avg".split(",")),
    ],
)
def test_eval_report(temp_dataset, capsys, pred_field, gt_field, check_fields):
    display.eval_report(
        temp_dataset.name, pred_field, gt_field, eval_kwargs=dict(compute_mAP=True)
    )
    captured = capsys.readouterr()

    for field in check_fields:
        assert field in captured.out


@pytest.mark.parametrize("avg_size", [False, True])
def test_compute_area(temp_dataset, avg_size):
    field = temp_dataset.make_unique_field_name()
    min_area, max_area = finegrained.data.display.compute_area(
        dataset=temp_dataset.name, field=field, average_size=avg_size
    )

    assert max_area > min_area

    if avg_size:
        assert max_area < 1000
    else:
        assert max_area > 1000
