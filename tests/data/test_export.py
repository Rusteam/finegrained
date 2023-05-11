from pathlib import Path

import pandas as pd

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
