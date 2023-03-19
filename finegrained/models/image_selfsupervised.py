"""Train, eval and predict with image embedding model.
"""
from flash.image import ImageEmbedder

from finegrained.models import ImageClassification


class ImageSelfSupervised(ImageClassification):
    """Use unsupervised learning in order to classify images"""

    @property
    def data_keys(self):
        return ["dataset"]

    def _init_model(
        self,
        backbone: str,
        training_strategy: str = "simclr",
        head: str = "simclr_head",
        pretraining_transform: str = "simclr_transform",
        **kwargs
    ):
        self.model = ImageEmbedder(
            training_strategy=training_strategy,
            head=head,
            pretraining_transform=pretraining_transform,
            backbone=backbone,
            pretrained=kwargs.get("pretrained", True),
            optimizer=kwargs.get("optimizer", "Adam"),
            learning_rate=kwargs.get("learning_rate", 0.001),
        )

    def _get_available_backbones(self):
        return ImageEmbedder.available_backbones()
