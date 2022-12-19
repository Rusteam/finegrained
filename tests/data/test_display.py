from finegrained.data import display


def test_print_labels(temp_dataset, capsys):
    display.print_labels(temp_dataset.name, "predictions")
    captured = capsys.readouterr()
    labels = captured.out.strip().splitlines()
    assert 'airplane' in labels
    assert len(labels) == 77


def test_classification_report(temp_dataset, capsys):
    display.classification_report(temp_dataset.name, "resnet18-imagenet-torch", "resnet18-imagenet-torch")
    captured = capsys.readouterr()
    assert 'precision' in captured.out
    assert 'recall' in captured.out
    assert 'accuracy' in captured.out
