"""Extend Flash modules and classes.
"""
from dataclasses import dataclass
from typing import Callable, Tuple, Union

import torch
import torchvision.transforms as T
from flash.core.data.io.input_transform import InputTransform
from flash.image import ImageClassificationInputTransform


@dataclass
class RandAugmentTransform(InputTransform):
    """Applies RandAugment transformation for input images."""

    image_size: Tuple[int, int] = (224, 224)
    mean: Union[float, Tuple[float, float, float]] = (0.485, 0.456, 0.406)
    std: Union[float, Tuple[float, float, float]] = (0.229, 0.224, 0.225)
    num_ops: int = 3
    magnitude: int = 9

    def input_per_sample_transform(self):
        return T.Compose(
            [
                T.ToTensor(),
                T.Resize(self.image_size),
                T.Normalize(self.mean, self.std),
            ]
        )

    def train_input_per_sample_transform(self):
        return T.Compose(
            [
                T.RandAugment(num_ops=self.num_ops, magnitude=self.magnitude),
                T.ToTensor(),
                T.Resize(self.image_size),
                T.RandomHorizontalFlip(),
                T.Normalize(self.mean, self.std),
            ]
        )

    def target_per_sample_transform(self) -> Callable:
        return torch.as_tensor


TRANSFORMS = dict(randaugment=RandAugmentTransform)


def get_transform(name: str) -> InputTransform:
    """Get image transform definition

    If given transform exists, then return it.
    Return basic image transforms otherwise.

    Args:
        name: name of transform definition

    Returns:
        Flash InputTransform object
    """
    if name is None:
        return ImageClassificationInputTransform
    return TRANSFORMS.get(name.lower(), ImageClassificationInputTransform)
