"""Utils to work with structured data.
"""
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


def load_data(
    path: str, key: Optional[str] = None, **kwargs
) -> Union[pd.DataFrame, np.ndarray, list]:
    """Load data from csv/json file."""
    if isinstance(path, list):
        data = pd.DataFrame(path)
    else:
        path = Path(path)
        if path.suffix == ".csv":
            data = pd.read_csv(path, **kwargs)
        elif path.suffix == ".csv":
            data = pd.read_json(path, **kwargs)
        elif path.suffix == ".npy":
            data = np.load(path, allow_pickle=False)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

    if key:
        data = data[key].tolist()

    return data


def write_data(path: str, data: list):
    """Write data to csv/json file."""
    path = Path(path)
    data = pd.DataFrame(data)
    if path.suffix == ".csv":
        data.to_csv(path, index=False)
    elif path.suffix == ".json":
        data.to_json(path, index=False)
    elif path.suffix == ".npy":
        with open(path, "wb") as f:
            np.save(f, data, allow_pickle=False)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
