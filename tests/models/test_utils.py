"""Test utils for training.
"""
import pytest
import torch
import torchvision.transforms as T
from PIL import Image

from finegrained.models import torch_utils


def test_parse_transforms():
    input_str = [
        dict(name="RandAugment", kwargs={"num_ops": 3, "magnitude": 7}),
        dict(name="ToTensor"),
        dict(name="Resize", args=[(224, 112)]),
        dict(name="RandomHorizontalFlip"),
    ]
    transforms = torch_utils.parse_transforms(input_str, normalize=True)

    image = torch.randint(0, 255, size=(600, 400, 3), dtype=torch.uint8)
    image = Image.fromarray(image.numpy())
    output = transforms(image)
    assert output.size() == torch.Size((3, 224, 112))
    assert output.dtype == torch.float
