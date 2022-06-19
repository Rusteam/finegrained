"""Torch and torchvision utils.
"""
from typing import Optional

import torch
import torchvision.transforms as T

from finegrained.utils import types


def get_cuda_count() -> int:
    """Get a count of gpus if cuda is available."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def _parse_transform(
    name: str,
    args: Optional[types.LIST_TYPE] = (),
    kwargs: Optional[dict] = {},
) -> list:
    return getattr(T, name)(*args, **kwargs)


def parse_transforms(
    transforms: types.LIST_DICT, normalize: bool = True
) -> T.Compose:
    if transforms is None:
        return None
    transforms = [_parse_transform(**t) for t in transforms]
    if normalize:
        transforms.append(
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        )
    return T.Compose(transforms)
