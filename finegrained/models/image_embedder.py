"""Train, eval and predict with image embedding model.
"""
from flash import Trainer
from flash.image import ImageClassificationData, ImageEmbedder

from finegrained.data.dataset_utils import load_fiftyone_dataset
from finegrained.models.torch_utils import get_cuda_count
from finegrained.models.utils import validate_train_config
from finegrained.utils.os_utils import read_yaml


def _init_datamodule(dataset: str, dataset_kwargs={}, **kwargs):
    dataset = load_fiftyone_dataset(dataset, **dataset_kwargs)
    data = ImageClassificationData.from_fiftyone(
        train_dataset=dataset,
        batch_size=kwargs.get("batch_size", 16)
    )
    return data


def _init_model(backbone: str,
                training_strategy: str = "dino",
                head: str = "dino_head",
                pretraining_transform: str = "dino_transform",
                **kwargs) -> ImageEmbedder:
    model = ImageEmbedder(
        training_strategy=training_strategy,
        head=head,
        pretraining_transform=pretraining_transform,
        backbone=backbone,
        pretrained=kwargs.get("pretrained", False),
        optimizer=kwargs.get("optimizer", "Adam"),
        learning_rate=kwargs.get("learning_rate", 0.001),
    )
    return model


def _finetune(model, data, epochs: int, **kwargs):
    trainer = Trainer(max_epochs=epochs,
                      gpus=get_cuda_count(),
                      limit_train_batches=kwargs.get("limit_train_batches"),
                      limit_val_batches=kwargs.get("limit_val_batches"),
                      )
    trainer.finetune(model,
                     datamodule=data,
                     strategy=kwargs.get("strategy", ("freeze_unfreeze", 1))
                     )
    trainer.save_checkpoint(kwargs.get("save_checkpoint", "model.pt"))


def finetune(cfg: str):
    cfg = read_yaml(cfg)
    validate_train_config(cfg, data_keys=["dataset"],
                          model_keys=["backbone"],
                          trainer_keys=["epochs"])
    data = _init_datamodule(**cfg["data"])
    model = _init_model(**cfg["model"])
    _finetune(model, data, **cfg["trainer"])
