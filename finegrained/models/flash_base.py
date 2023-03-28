"""Base logic to use Flash datamodules, models and and trainers.
"""
import itertools
from itertools import chain
from pathlib import Path
from typing import Tuple

import fiftyone as fo
import torch
from flash import Trainer
from flash.core.classification import FiftyOneLabelsOutput
from tqdm import tqdm

from finegrained.models.torch_utils import get_device
from finegrained.utils.dataset import load_fiftyone_dataset
from finegrained.utils.os_utils import read_yaml


def _map_classifications_to_detections(predictions, patches, patch_field) -> dict:
    paired = []
    for clf, smp in zip(predictions, patches):
        detection = fo.Detection(
            label=clf.label,
            bounding_box=smp[patch_field].bounding_box,
            confidence=clf.confidence,
        )
        paired.append((detection, smp.sample_id))

    detections = {}
    for key, group in itertools.groupby(paired, lambda x: x[1]):
        detections[key] = fo.Detections(detections=[d for d, _ in group])
    return detections


class FlashFiftyOneTask:
    """A base class to inherit from for task-specific learners."""

    _model = None
    _data = None
    _prediction_dataset = None
    _patch_dataset = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, val):
        self._model = val

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        self._data = val

    @property
    def train_dataset(self):
        return self._train_dataset

    @train_dataset.setter
    def train_dataset(self, val):
        self._train_dataset = val

    @property
    def prediction_dataset(self):
        return self._prediction_dataset

    @prediction_dataset.setter
    def prediction_dataset(self, val):
        self._prediction_dataset = val

    @property
    def patch_dataset(self):
        return self._patch_dataset

    @patch_dataset.setter
    def patch_dataset(self, val):
        self._patch_dataset = val

    @property
    def data_keys(self):
        return ["dataset", "label_field"]

    @property
    def model_keys(self):
        return ["backbone"]

    @property
    def trainer_keys(self):
        return ["epochs"]

    def _init_training_datamodule(self, *args, **kwargs):
        raise NotImplementedError("Subclass has to implement this method")

    def _init_prediction_datamodule(self, *args, **kwargs):
        raise NotImplementedError("Subclass has to implement this method")

    def _init_model(self, *args, **kwargs):
        raise NotImplementedError("Subclass has to implement this method")

    def _load_pretrained_model(self, *args, **kwargs):
        raise NotImplementedError("Subclass has to implement this method")

    def _load_fiftyone_dataset(self, name: str, **kwargs):
        dataset = load_fiftyone_dataset(name, **kwargs)
        return dataset

    @staticmethod
    def calculate_features(model, dataloader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract embeddings for all samples in a dataloader

        Args:
            model: flash model instance
            dataloader: torchvision dataloader instance

        Returns:
            a tuple with features and labels if present
        """
        feature_extractor = model.eval()
        features = []
        labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="extracting features"):
                features.append(feature_extractor(batch["input"]))
                if "target" in batch.keys():
                    labels.append(batch["target"].argmax(dim=1))

        features = torch.cat(features)
        labels = torch.cat(labels) if len(labels) > 0 else labels
        return features, labels

    def _validate_train_config(self, cfg: dict):
        self.cfg_keys = dict(
            data=self.data_keys,
            model=self.model_keys,
            trainer=self.trainer_keys,
        )

        for key in self.cfg_keys.keys():
            assert key in cfg, f"config file has to contain {key=}"
        for comp, comp_keys in self.cfg_keys.items():
            for key in comp_keys:
                assert key in cfg[comp], f"{comp} has to contain {key=}"

    def _get_available_backbones(self):
        raise NotImplementedError("Subclass has to implement this method")

    def _finetune(self, epochs: int, **kwargs):
        assert self.model is not None, "make sure a model is initialized"
        assert self.data is not None, "make sure a datamodule is initialized"

        device, device_count = get_device(kwargs.get("device"))
        trainer = Trainer(
            max_epochs=epochs,
            limit_train_batches=kwargs.get("limit_train_batches"),
            limit_val_batches=kwargs.get("limit_val_batches"),
            accelerator=device.type,
            devices=device_count,
        )

        strategy = kwargs.get("strategy", ("freeze_unfreeze", 1))
        if isinstance(strategy, list):
            strategy = tuple(strategy)

        trainer.finetune(
            self.model,
            datamodule=self.data,
            strategy=strategy,
        )

        trainer.save_checkpoint(kwargs.get("save_checkpoint", "model.pt"))

        if hasattr(self, "test_dataset") and (
            test_loader := self.data.test_dataloader()
        ):
            trainer.test(dataloaders=test_loader)

    def _predict(self):
        device, device_count = get_device()
        trainer = Trainer(accelerator=device.type, devices=device_count)
        predictions = trainer.predict(
            self.model,
            datamodule=self.data,
            output=FiftyOneLabelsOutput(self.model.labels, return_filepath=False),
        )
        predictions = list(chain.from_iterable(predictions))
        return predictions

    def finetune(self, cfg: str | Path):
        """Fine-tune model with given data

        Args:
            cfg: path to yaml file with data, model and trainer configs
        """
        cfg = read_yaml(cfg)
        self._validate_train_config(cfg)
        self._init_training_datamodule(**cfg["data"])
        self._init_model(**cfg["model"])
        self._finetune(**cfg["trainer"])

    def predict(
        self,
        dataset: str,
        label_field: str,
        ckpt_path: str,
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 4,
        patch_field: str = None,
        **kwargs,
    ):
        """Classify samples from a dataset and assign values to label field

        Args:
            dataset: which dataset to run samples on
            label_field: which field to assign predictions to
            ckpt_path: flash model checkpoint path
            image_size: Image size for inference
            batch_size: predictions batch size
            patch_field: run predictions on patches on this field
            **kwargs: dataset loading filters

        Returns:
            none
        """
        self._init_prediction_datamodule(
            dataset,
            image_size=image_size,
            batch_size=batch_size,
            patch_field=patch_field,
            **kwargs,
        )
        assert (
            self.prediction_dataset is not None
        ), "Make sure _init_prediction_datamodule() creates self.prediction_dataset"
        self._load_pretrained_model(ckpt_path)
        predictions = self._predict()
        if bool(patch_field):
            self.patch_dataset.set_values(label_field, predictions)

            for smp in self.prediction_dataset:
                detections = []
                for det in smp[patch_field].detections:
                    clf = self.patch_dataset[det.patch_filepath][label_field]
                    detections.append(
                        fo.Detection(
                            label=clf.label,
                            bounding_box=det.bounding_box,
                            confidence=clf.confidence,
                        )
                    )

                smp[label_field] = fo.Detections(detections=detections)
                smp.save()
        else:
            self.prediction_dataset.set_values(label_field, predictions)

    def list_backbones(self, prefix: str = None):
        backbones = self._get_available_backbones()
        if prefix:
            backbones = list(filter(lambda x: x.startswith(prefix), backbones))
        return backbones
