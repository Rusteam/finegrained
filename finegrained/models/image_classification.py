"""Train, evaluate and predict with image classification models.
"""
from typing import Tuple

from fiftyone import Dataset
from flash import DataModule
from flash.core.data.utilities.classification import SingleLabelTargetFormatter
from flash.image import (
    ImageClassificationData,
    ImageClassifier,
)

from finegrained.data.dataset_utils import (
    load_fiftyone_dataset,
    get_unique_labels
)
from finegrained.models.flash_base import FlashFiftyOneTask
from finegrained.models.flash_transforms import get_transform
from finegrained.utils import types


# TODO split before training

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
        **kwargs
    ) -> Tuple[DataModule, Dataset]:
        self.prediction_dataset = load_fiftyone_dataset(dataset, **kwargs)
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
