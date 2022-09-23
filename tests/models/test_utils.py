"""Test utils for training.
"""
import pytest
import torch
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


@pytest.mark.parametrize("device_type", [None, "cpu"])
def test_get_device(device_type):
    device, count = torch_utils.get_device(device_type)

    if device_type is None:
        if torch.cuda.is_available():
            assert device.type == "cuda"
            assert count == torch.cuda.device_count()
        elif torch.backends.mps.is_available():
            assert device.type == "mps"
            assert count == 1
        else:
            assert device.type == "cpu"
            assert count == 1
    else:
        assert device.type == device_type
        assert count == 1
