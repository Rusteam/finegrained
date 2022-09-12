"""Train, eval and predict with image embedding model.
"""
from flash.image import ImageClassificationData, ImageEmbedder

from finegrained.utils.dataset import load_fiftyone_dataset
from finegrained.models.flash_base import FlashFiftyOneTask

# TODO finish here


class ImageSelfSupervised(FlashFiftyOneTask):
    """Use unsupervised learning in order to classify images"""

    @property
    def data_keys(self):
        return ["dataset"]

    def _init_training_datamodule(
        self, dataset: str, dataset_kwargs={}, **kwargs
    ):
        dataset = load_fiftyone_dataset(dataset, **dataset_kwargs)
        self.data = ImageClassificationData.from_fiftyone(
            train_dataset=dataset, batch_size=kwargs.get("batch_size", 16)
        )

    def _init_model(
        self,
        backbone: str,
        training_strategy: str = "simclr",
        head: str = "simclr_head",
        pretraining_transform: str = "simclr_transform",
        **kwargs
    ) -> ImageEmbedder:
        model = ImageEmbedder(
            training_strategy=training_strategy,
            head=head,
            pretraining_transform=pretraining_transform,
            backbone=backbone,
            pretrained=kwargs.get("pretrained", True),
            optimizer=kwargs.get("optimizer", "Adam"),
            learning_rate=kwargs.get("learning_rate", 0.001),
        )
        return model

    def _get_available_backbones(self):
        return ImageEmbedder.available_backbones()
