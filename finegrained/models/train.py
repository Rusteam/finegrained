# from itertools import chain
# from typing import Any
#
# from flash import Trainer, DataModule
# from flash.image import ObjectDetectionData, ObjectDetector
# from flash.image.detection.output import FiftyOneDetectionLabelsOutput
#
# import fiftyone as fo
# import fiftyone.utils.splits as fous
# import fiftyone.zoo as foz
#
# from finegrained.data.utils import load_fiftyone_dataset, get_unique_labels
# from finegrained.utils.os_utils import parse_yaml
#
#
# def _load_datamodule(dataset: str, label_field: str, **kwargs) -> tuple[DataModule, int]:
#     if "fields_exist" not in kwargs:
#         kwargs.update({"fields_exist": label_field})
#     dataset = load_fiftyone_dataset(dataset, **kwargs)
#     splits = {"train": 0.8, "test": 0.1, "val": 0.1}
#     fous.random_split(dataset, splits)
#     train_dataset = dataset.match_tags("train")
#     test_dataset = dataset.match_tags("test")
#     val_dataset = dataset.match_tags("val")
#     if len(dataset.default_classes) > 0:
#         dataset.default_classes.pop(0)
#     datamodule = ObjectDetectionData.from_fiftyone(
#         train_dataset=train_dataset,
#         test_dataset=test_dataset,
#         val_dataset=val_dataset,
#         label_field=label_field,
#         transform_kwargs={"image_size": 512},
#         batch_size=4,
#     )
#     return datamodule, len(get_unique_labels(dataset, label_field))
#
#
# def _build_model(head: str, backbone: str, num_classes: int):
#     model = ObjectDetector(
#         backbone=backbone,
#         head=head,
#         num_classes=num_classes,
#         image_size=512,
#     )
#     return model
#
#
# def _finetune(model, datamodule, **kwargs):
#     trainer = Trainer(max_epochs=kwargs.get("max_epochs", 10),
#                       limit_train_batches=10)
#
#     trainer.finetune(model, datamodule=datamodule, strategy="freeze")
#
#     trainer.save_checkpoint(kwargs.get("save_checkpoint", "model.pt"))
#
#
# def object_detection(config: Any):
#     conf = parse_yaml(config)
#     data_conf = conf.get("data", AssertionError)
#     datamodule, n_classes = _load_datamodule(data_conf['dataset'],
#                                              data_conf['label_field'],
#                                              **data_conf['kwargs'])
#     model = _build_model(num_classes=n_classes, **conf.get("model", AssertionError))
#     _finetune(model, datamodule, **conf.get("train"))
#
