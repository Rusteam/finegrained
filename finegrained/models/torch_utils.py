"""Torch and torchvision utils.
"""
from typing import Optional

import torch
import torchvision.transforms as T
from pytorch_lightning.accelerators import CUDAAccelerator, MPSAccelerator

from finegrained.utils import types

cuda_available = torch.cuda.is_available()


def get_device(device_type: Optional[str] = None) -> tuple[torch.device, int]:
    """Get torch.device and its count with cuda/mps if available, else cpu

    Args:
        device_type: preferred device type

    Returns:
        torch.device instance and its count
    """
    if (
        device_type is None and CUDAAccelerator.is_available()
    ) or device_type == "cuda":
        return torch.device("cuda"), CUDAAccelerator.auto_device_count()
    elif (
        device_type is None and MPSAccelerator.is_available()
    ) or device_type == "mps":
        return torch.device("mps"), MPSAccelerator.auto_device_count()
    else:
        if device_type is not None and device_type != "cpu":
            raise ValueError(f"Unknown device type {device_type}")
        return torch.device("cpu"), 1


def get_default_batch_size() -> int:
    return 16 if cuda_available else 4


def _parse_transform(
    name: str,
    args: Optional[types.LIST_TYPE] = (),
    kwargs: Optional[dict] = {},
) -> list:
    return getattr(T, name)(*args, **kwargs)


def parse_transforms(transforms: types.LIST_DICT, normalize: bool = True) -> T.Compose:
    if transforms is None:
        return None
    transforms = [_parse_transform(**t) for t in transforms]
    if normalize:
        transforms.append(T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    return T.Compose(transforms)
