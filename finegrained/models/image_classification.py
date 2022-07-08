"""Train, evaluate and predict with image classification models.
"""
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from fiftyone import Dataset, types as fot
from flash import DataModule
from flash.core.data.utilities.classification import SingleLabelTargetFormatter
from flash.image import (
    ImageClassificationData,
    ImageClassifier,
)
from torchvision import transforms as T

from finegrained.data.dataset_utils import (
    load_fiftyone_dataset,
    get_unique_labels
)
from finegrained.models.flash_base import FlashFiftyOneTask
from finegrained.models.flash_transforms import get_transform
from finegrained.models.triton import init_model_repo
from finegrained.utils import types, os_utils


# TODO split before training

def parse_image_size(image_size: types.IMAGE_SIZE) -> Tuple[int, int]:
    """Return height and width from a tuple or int image size

    Args:
        image_size: a int size or a tuple of (height, width)

    Returns:
        a tuple of height and width
    """
    if isinstance(image_size, int):
        return image_size, image_size
    elif isinstance(image_size, (tuple, list)):
        assert len(image_size) == 2, \
            f"if tuple/list should be 2 elements, got {image_size=}"
        return image_size
    else:
        raise ValueError(f"{image_size=} of type {type(image_size)} not understood")


def _get_patch_array(patch_sample, patch_field) -> np.ndarray:
    img = np.array(Image.open(patch_sample.filepath))
    img_h, img_w = img.shape[:2]
    x, y, w, h = patch_sample[patch_field].bounding_box
    xmin = max(0, int(img_w * x))
    ymin = max(0, int(img_h * y))
    xmax = min(img_w, int(img_w * (x + w)))
    ymax = min(img_h, int(img_h * (y + h)))
    return img[ymin: ymax, xmin: xmax].transpose(2, 0, 1)


class ImageClassification(FlashFiftyOneTask):
    """Image classification task."""

    def _init_training_datamodule(
        self,
        dataset: str,
        label_field: str,
        tags: types.DICT_STR_STR = {"train": "train", "val": "val"},
        **kwargs
    ):
        dataset = load_fiftyone_dataset(dataset)
        dataset_tags = list(tags.values())
        self.labels = get_unique_labels(
            dataset.match_tags(dataset_tags), label_field
        )
        self.data = ImageClassificationData.from_fiftyone(
            train_dataset=dataset.match_tags(tags["train"]),
            val_dataset=dataset.match_tags(tags["val"]),
            label_field=label_field,
            batch_size=kwargs.get("batch_size", 16),
            target_formatter=SingleLabelTargetFormatter(
                labels=self.labels, num_classes=len(self.labels)
            ),
            train_transform=get_transform(kwargs.get("train_transform")),
            val_transform=get_transform(kwargs.get("val_transform")),
            transform_kwargs=kwargs.get("transform_kwargs"),
        )

    def _init_prediction_datamodule(
        self,
        dataset: str,
        image_size: Tuple[int, int],
        batch_size: int = 4,
        patch_field: str = None,
        **kwargs
    ) -> Tuple[DataModule, Dataset]:
        self.prediction_dataset = load_fiftyone_dataset(dataset, **kwargs)
        if bool(patch_field):
            assert self.prediction_dataset.has_sample_field(patch_field)
            self.patches = self.prediction_dataset.to_patches(patch_field)

            arrays = [_get_patch_array(smp, patch_field) for smp in self.patches]
            self.data = ImageClassificationData.from_numpy(
                predict_data=arrays,
                batch_size=batch_size,
                transform_kwargs=dict(image_size=image_size)
            )
        else:
            self.data = ImageClassificationData.from_fiftyone(
                predict_dataset=self.prediction_dataset,
                batch_size=batch_size,
                transform_kwargs=dict(image_size=image_size),
            )

    def _init_model(self, backbone: str, **kwargs) -> ImageClassifier:
        self.model = ImageClassifier(
            num_classes=len(self.labels),
            backbone=backbone,
            labels=self.labels,
            **kwargs,
        )

    def _load_pretrained_model(self, ckpt_path: str) -> ImageClassifier:
        self.model = ImageClassifier.load_from_checkpoint(ckpt_path)

    def _get_available_backbones(self):
        return ImageClassifier.available_backbones()

    def generate_dummy_inputs(self, image_size):
        h, w = parse_image_size(image_size)
        return torch.randn(2, 3, h, w),

    def _create_triton_config(self, image_size: types.IMAGE_SIZE):
        h, w = parse_image_size(image_size)
        return {
            "backend": "onnxruntime",
            "max_batch_size": 16,
            "input": [
                dict(
                    name=self.input_names[0],
                    data_type="TYPE_FP32",
                    dims=[3, h, w]
                )
            ],
            "output": [
                dict(
                    name=self.output_names[0],
                    data_type="TYPE_FP32",
                    dims=[self.model.num_classes],
                    label_filename="labels.txt"
                )
            ]
        }

    @property
    def has_triton_labels(self) -> bool:
        return True


class ImageTransform(torch.nn.Module):
    def __init__(self, image_size: int):
        super(ImageTransform, self).__init__()
        self._ = torch.nn.Sequential()
        self.transforms = T.Compose([
            T.Lambda(lambda x: x.permute(2, 0, 1) / 255.0),
            T.Resize(image_size),
            T.RandomCrop(image_size),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.image_size = image_size

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        transformed = self.transforms(samples.squeeze(0))
        batch = transformed.unsqueeze(0)
        return batch

    @staticmethod
    def generate_dummy_input():
        dummy = torch.randint(0, 255, size=(2, 256, 233))
        return dummy,

    def export_onnx(self, write_path: str):
        dummy = self.generate_dummy_input()
        torch.onnx.export(self,
                          dummy,
                          write_path,
                          opset_version=13,
                          input_names=["raw"],
                          output_names=["preprocessed"],
                          dynamic_axes=None)

    def export_triton(self, triton_repo: str, triton_model: str, version=1):
        model_version_dir = init_model_repo(triton_repo, triton_model, version)
        self.export_onnx(model_version_dir / "model.onnx")
