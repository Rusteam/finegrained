"""Train, evaluate and predict with image meta-learning .
"""
from typing import Tuple

import fiftyone.core.labels as fol
import torch
from fiftyone import Dataset
from flash import DataModule
from flash.image import ImageClassificationData, ImageClassifier

from finegrained.models.flash_base import FlashFiftyOneTask
from finegrained.models.flash_transforms import get_transform
from finegrained.utils import types
from finegrained.utils.dataset import get_all_filepaths, load_fiftyone_dataset
from finegrained.utils.triton import TritonExporter

# TODO turn into a class
# TODO split before training


def _init_support_datamodule(
    dataset: str,
    label_field: str,
    image_size: Tuple[int, int],
    batch_size: int = 4,
    **kwargs,
) -> DataModule:
    dataset = load_fiftyone_dataset(dataset, **kwargs)

    data = ImageClassificationData.from_fiftyone(
        train_dataset=dataset,
        label_field=label_field,
        batch_size=batch_size,
        transform_kwargs=dict(image_size=image_size),
    )
    return data


def _init_query_datamodule(
    dataset: str, image_size: Tuple[int, int], batch_size: int = 4, **kwargs
) -> Tuple[DataModule, Dataset]:
    dataset = load_fiftyone_dataset(dataset, **kwargs)
    data = ImageClassificationData.from_fiftyone(
        predict_dataset=dataset,
        batch_size=batch_size,
        transform_kwargs=dict(image_size=image_size),
    )
    return data, dataset


class ImageMetalearn(FlashFiftyOneTask, TritonExporter):
    """Image meta learning."""

    @property
    def model_keys(self):
        return ["backbone", "training_strategy", "training_strategy_kwargs"]

    def _init_training_datamodule(self, dataset: str, label_field: str, **kwargs):
        dataset = load_fiftyone_dataset(dataset)
        self.labels = dataset.match_tags(["train_metalearn", "val_metalearn"]).distinct(
            f"{label_field}.label"
        )

        def get_targets(tag):
            targets = [
                self.labels.index(smp[label_field].label)
                for smp in dataset.match_tags(tag).select_fields(label_field)
            ]
            return targets

        train_files = get_all_filepaths(dataset.match_tags("train_metalearn"))
        train_targets = get_targets("train_metalearn")
        val_files = get_all_filepaths(dataset.match_tags("val_metalearn"))
        val_targets = get_targets("val_metalearn")

        # use from_files because of label mapping error with from_fiftyone
        self.data = ImageClassificationData.from_files(
            train_files=train_files,
            train_targets=train_targets,
            val_files=val_files,
            val_targets=val_targets,
            batch_size=kwargs.get("batch_size", 16),
            transform=get_transform(kwargs.get("transform")),
            transform_kwargs=kwargs.get("transform_kwargs"),
        )

    def _init_prediction_datamodule(
        self,
        support_dataset: str,
        support_label: str,
        query_dataset: str,
        image_size: types.IMAGE_SIZE,
        batch_size: int,
        support_kwargs={},
        query_kwargs={},
    ):
        self.support_data = _init_support_datamodule(
            support_dataset,
            support_label,
            image_size=image_size,
            batch_size=batch_size,
            **support_kwargs,
        )
        self.query_data, self.prediction_dataset = _init_query_datamodule(
            query_dataset,
            image_size=image_size,
            batch_size=batch_size,
            **query_kwargs,
        )

    def _init_model(
        self,
        backbone: str,
        training_strategy: str,
        training_strategy_kwargs: dict,
        **kwargs,
    ):
        self.model = ImageClassifier(
            num_classes=len(self.labels),
            backbone=backbone,
            labels=self.labels,
            training_strategy=training_strategy,
            training_strategy_kwargs=training_strategy_kwargs,
            **kwargs,
        )

    def _load_pretrained_model(self, ckpt_path: str):
        self.model = ImageClassifier.load_from_checkpoint(ckpt_path)

    def _predict(self):
        from learn2learn.nn import PrototypicalClassifier

        feature_extractor = self.model.backbone
        support_features, support_labels = self.calculate_features(
            feature_extractor, self.support_data.train_dataloader()
        )
        query_features, _ = self.calculate_features(
            feature_extractor, self.query_data.predict_dataloader()
        )

        clf = PrototypicalClassifier(support=support_features, labels=support_labels)
        dist = clf(query_features)
        class_probs = dist.softmax(dim=1)
        conf, index = class_probs.max(dim=1)

        def _get_class_label(index):
            return self.support_data.train_dataset.labels[index]

        predictions = [
            fol.Classification(
                label=_get_class_label(i.item()),
                confidence=c.item(),
                logits=l.cpu().numpy(),
            )
            for c, i, l in zip(conf, index, dist)
        ]

        return predictions

    def predict(
        self,
        support_dataset: str,
        support_label_field: str,
        query_dataset: str,
        query_label_field: str,
        ckpt_path: str,
        image_size: types.IMAGE_SIZE = (224, 224),
        batch_size: int = 4,
        support_kwargs={},
        query_kwargs={},
    ):
        """Classify samples from a dataset and assign values to label field

        Args:
            support_dataset: support fiftyone dataset
            support_label_field: label field for the support dataset
            query_dataset: query fiftyone dataset
            query_label_field: where to assign predictions on the query dataset
            ckpt_path: flash model checkpoint path
            image_size: Image size for inference
            batch_size: predictions batch size
            support_kwargs: support dataset filters
            query_kwargs: query dataset filters
        """
        self._init_prediction_datamodule(
            support_dataset=support_dataset,
            support_label=support_label_field,
            query_dataset=query_dataset,
            image_size=image_size,
            batch_size=batch_size,
            support_kwargs=support_kwargs,
            query_kwargs=query_kwargs,
        )
        self._load_pretrained_model(ckpt_path)
        predictions = self._predict()
        self.prediction_dataset.set_values(query_label_field, predictions)
        print(
            f"{len(predictions)} predictions saved to "
            f"{query_label_field} field in {query_dataset}"
        )

    def _get_available_backbones(self):
        return ImageClassifier.available_backbones()

    def _load_model_torch(self, ckpt_path) -> torch.nn.Module:
        self._load_pretrained_model(ckpt_path)
        return self.model.backbone

    def generate_dummy_inputs(
        self, image_size: int
    ) -> tuple[torch.Tensor,]:
        return (torch.rand(1, 3, image_size, image_size),)

    def _create_triton_ensemble_config(
        self, preprocessing_name: str, feature_extractor: str, embedding_size: int
    ) -> dict:
        return {
            "platform": "ensemble",
            "max_batch_size": 1,
            "input": [dict(name="IMAGE", dims=[-1, -1, 3], data_type="TYPE_UINT8")],
            "output": [
                dict(
                    name="FEATURES",
                    dims=[embedding_size],
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
                        model_name=feature_extractor,
                        model_version=-1,
                        input_map=dict(image="PREPROCESSED"),
                        output_map=dict(output="FEATURES"),
                    ),
                ]
            ),
        }
