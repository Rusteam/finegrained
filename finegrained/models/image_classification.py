"""Train, evaluate and predict with image classification models.
"""
import tempfile
from pathlib import Path
from typing import Tuple

import cv2
import fiftyone as fo
import fiftyone.utils.patches as foup
import numpy as np
import torch
from flash.core.data.utilities.classification import SingleLabelTargetFormatter
from flash.image import ImageClassificationData, ImageClassifier
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from finegrained.models.flash_base import FlashFiftyOneTask
from finegrained.models.flash_transforms import get_transform
from finegrained.utils import types
from finegrained.utils.dataset import get_unique_labels, load_fiftyone_dataset
from finegrained.utils.triton import TritonExporter


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
        assert (
            len(image_size) == 2
        ), f"if tuple/list should be 2 elements, got {image_size=}"
        return image_size
    else:
        raise ValueError(f"{image_size=} of type {type(image_size)} not understood")


def _get_patch_array(patch_sample, patch_field, transpose=False) -> np.ndarray:
    img = np.array(Image.open(patch_sample.filepath))
    img_h, img_w = img.shape[:2]
    x, y, w, h = patch_sample[patch_field].bounding_box
    xmin = max(0, int(img_w * x))
    ymin = max(0, int(img_h * y))
    xmax = min(img_w, int(img_w * (x + w)))
    ymax = min(img_h, int(img_h * (y + h)))
    crop = img[ymin:ymax, xmin:xmax]
    if transpose:
        crop = crop.transpose(2, 0, 1)
    return crop


def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _to_patches(dataset: fo.Dataset, patch_field: str, write_dir: Path):
    if not write_dir.exists():
        write_dir.mkdir(parents=True)

    i = 0
    for smp in tqdm(dataset, "extracting patches"):
        img = read_rgb(smp.filepath)
        for det in smp[patch_field].detections:
            patch = foup.extract_patch(img, det)
            path = write_dir / f"{i:04d}.jpg"
            cv2.imwrite(str(path), patch)
            det["patch_filepath"] = str(path)
            i += 1
        smp.save()


class ImageClassification(FlashFiftyOneTask, TritonExporter):
    """Image classification task."""

    def _init_training_datamodule(
        self,
        dataset: str,
        label_field: str,
        tags: types.DICT_STR_STR = {
            "train": "train",
            "val": "val",
            "test": "test",
        },
        **kwargs,
    ):
        dataset = load_fiftyone_dataset(dataset)
        dataset_tags = list(tags.values())
        self.labels = get_unique_labels(dataset.match_tags(dataset_tags), label_field)
        self.data = ImageClassificationData.from_fiftyone(
            train_dataset=dataset.match_tags(tags["train"]),
            val_dataset=dataset.match_tags(tags["val"]),
            test_dataset=dataset.match_tags(tags["test"])
            if "test" in dataset_tags
            else None,
            label_field=label_field,
            batch_size=kwargs.get("batch_size", 16),
            target_formatter=SingleLabelTargetFormatter(
                labels=self.labels, num_classes=len(self.labels)
            ),
            transform=get_transform(kwargs.get("train_transform")),
            transform_kwargs=kwargs.get("transform_kwargs"),
        )

    def _init_prediction_datamodule(
        self,
        dataset: str,
        image_size: Tuple[int, int],
        batch_size: int = 4,
        patch_field: str = None,
        **kwargs,
    ):
        self.prediction_dataset = load_fiftyone_dataset(dataset, **kwargs)
        if bool(patch_field):
            assert self.prediction_dataset.has_sample_field(patch_field)

            export_dir = Path(tempfile.TemporaryDirectory().name).resolve() / "export"

            _to_patches(self.prediction_dataset, patch_field, export_dir)
            self.patch_dataset = fo.Dataset.from_images_dir(str(export_dir))
            self.patch_dataset.persistent = False

            self.data = ImageClassificationData.from_fiftyone(
                predict_dataset=self.patch_dataset,
                batch_size=batch_size,
                transform_kwargs=dict(image_size=image_size),
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

    def _load_model_torch(self, ckpt_path: str) -> torch.nn.Module:
        self._load_pretrained_model(ckpt_path)
        model = SoftmaxClassifier(self.model)
        return model

    def generate_dummy_inputs(self, image_size) -> Tuple[torch.Tensor]:
        h, w = parse_image_size(image_size)
        return (torch.randn(2, 3, h, w),)

    def _create_triton_config(self, image_size: types.IMAGE_SIZE, **kwargs) -> dict:
        h, w = parse_image_size(image_size)
        return {
            "backend": "onnxruntime",
            "max_batch_size": self.triton_batch_size,
            "input": [
                dict(
                    name=self.input_names[0],
                    data_type="TYPE_FP32",
                    dims=[3, h, w],
                )
            ],
            "output": [
                dict(
                    name=self.output_names[0],
                    data_type="TYPE_FP32",
                    dims=[self.model.num_classes],
                    label_filename=self.triton_labels_path,
                )
            ],
        }

    @property
    def triton_labels(self) -> types.LIST_STR:
        labels = self.model.hparams["labels"]
        return labels

    @property
    def triton_python_file(self) -> str:
        return (
            Path(__file__).parents[1]
            / "utils"
            / "triton_python"
            / "image_classification.py"
        )

    def _create_triton_python_backend(self):
        return {
            "backend": "python",
            "max_batch_size": self.triton_batch_size,
            "input": [dict(name="IMAGE", dims=[-1, -1, 3], data_type="TYPE_UINT8")],
            "output": [
                dict(
                    name="CLASS_PROBS",
                    dims=[-1],
                    data_type="TYPE_FP32",
                    label_filename=self.triton_labels_path,
                )
            ],
        }

    def _generate_triton_python_names(
        self, preprocessing_name: str, classifier_name: str, **kwargs
    ):
        return [
            f"preprocessing={preprocessing_name}",
            f"classifier={classifier_name}",
        ]

    def _create_triton_ensemble_config(
        self, preprocessing_name: str, classifier_name: str
    ) -> dict:
        return {
            "platform": "ensemble",
            "max_batch_size": 1,
            "input": [dict(name="IMAGE", dims=[-1, -1, 3], data_type="TYPE_UINT8")],
            "output": [
                dict(
                    name="CLASS_PROBS",
                    dims=[-1],
                    data_type="TYPE_FP32",
                )
            ],
            "ensemble_scheduling": dict(
                step=[
                    dict(
                        model_name=preprocessing_name,
                        model_version=-1,
                        input_map=dict(image="IMAGE"),
                        output_map=dict(output="PREPROCESSED"),
                    ),
                    dict(
                        model_name=classifier_name,
                        model_version=-1,
                        input_map=dict(image="PREPROCESSED"),
                        output_map=dict(output="CLASS_PROBS"),
                    ),
                ]
            ),
        }


class ImageTransform(torch.nn.Module, TritonExporter):
    def __init__(self, size: int):
        super(ImageTransform, self).__init__()
        self._ = torch.nn.Sequential()
        self.transforms = T.Compose(
            [
                T.Lambda(lambda x: x.permute(2, 0, 1) / 255.0),
                T.Resize(size),
                T.RandomCrop(size),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.image_size = size

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        transformed = self.transforms(samples.squeeze(0))
        batch = transformed.unsqueeze(0)
        return batch

    def generate_dummy_inputs(self, **kwargs) -> tuple[torch.Tensor]:
        h, w = parse_image_size(self.image_size)
        dummy = torch.randint(
            0, 255, size=(1, h * 2 + 1, w * 3 - 10, 3), dtype=torch.uint8
        )
        return (dummy,)

    @property
    def dynamic_axes(self):
        return {"image": [0, 1, 2], "output": [0]}

    def _load_model_torch(self, *args, **kwargs) -> torch.nn.Module:
        return self

    @property
    def triton_batch_size(self):
        return 1


class SoftmaxClassifier(torch.nn.Module):
    def __init__(self, model):
        super(SoftmaxClassifier, self).__init__()
        self.model = model
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.model(x)
        probs = self.softmax(logits)
        return probs
