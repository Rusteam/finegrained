from finegrained.data import display


def test_print_labels(temp_dataset, capsys):
    display.print_labels(temp_dataset.name, "predictions")
    captured = capsys.readouterr()
    labels = captured.out.strip().splitlines()
    assert 'airplane' in labels
    assert len(labels) == 77
