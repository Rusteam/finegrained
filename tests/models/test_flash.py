"""Test training models.
"""
from pathlib import Path

import fiftyone as fo
import fiftyone.utils.random as four
import fiftyone.zoo as foz
import pytest
import torch

from finegrained.data.brain import compute_hardness
from finegrained.data.tag import split_classes, split_dataset
from finegrained.models import (
    ImageClassification,
    ImageMetalearn,
    ImageSelfSupervised,
    ImageTransform,
)
from finegrained.utils.os_utils import read_yaml, write_yaml


@pytest.fixture(scope="session")
def clf_dataset():
    if fo.dataset_exists("train_test_temp"):
        fo.delete_dataset("train_test_temp")
    dataset = foz.load_zoo_dataset("quickstart").take(100).clone("train_test_temp")
    four.random_split(dataset, {"train": 0.7, "val": 0.1, "test": 0.2})
    yield dataset
    fo.delete_dataset(dataset.name)


@pytest.fixture
def clf_config(clf_dataset, tmp_path):
    cfg_path = Path(tmp_path) / "config.yaml"
    model_path = Path(tmp_path) / "model.pt"
    cfg = dict(
        data=dict(
            dataset=clf_dataset.name,
            label_field="ground_truth",
            batch_size=2,
            transform="randaugment",
            transform_kwargs=dict(image_size=(224, 112)),
        ),
        model=dict(
            backbone="efficientnet_b0",
        ),
        trainer=dict(
            epochs=2,
            limit_train_batches=3,
            limit_val_batches=2,
            save_checkpoint=str(model_path),
            strategy="freeze",
            device="cpu",
        ),
    )
    write_yaml(cfg, str(cfg_path))

    yield cfg_path, model_path, clf_dataset
    cfg_path.unlink(False)
    model_path.unlink(True)


def test_classification_finetune(clf_config, tmp_path):
    cfg, model_path, dataset = clf_config
    img_clf = ImageClassification()
    img_clf.finetune(str(cfg))
    assert model_path.exists()

    img_clf.predict(
        dataset.name,
        label_field="test_temp_predictions",
        ckpt_path=str(model_path),
        include_tags=["test", "val"],
        image_size=(224, 112),
    )
    assert dataset.has_sample_field("test_temp_predictions")

    _ = dataset.evaluate_classifications(
        "test_temp_predictions",
        gt_field="resnet18-imagenet-torch",
        eval_key="test_eval_temp",
    )

    img_clf.predict(
        dataset.name,
        label_field="test_temp_patches",
        ckpt_path=str(model_path),
        include_tags=["test", "val"],
        image_size=(224, 112),
        patch_field="predictions",
        include_labels={"predictions": ["carrot", "car", "kite"]},
        max_samples=5,
    )
    assert dataset.has_sample_field("test_temp_patches")


@pytest.fixture
def meta_learn_cfg(clf_dataset, tmp_path):
    cfg_path = Path(tmp_path) / "meta_learn.yaml"
    model_path = Path(tmp_path) / "meta_learn.pt"

    meta_learn_dataset = clf_dataset.clone(clf_dataset.name + "_meta_learn")
    label_field = "resnet18-imagenet-torch"
    split_classes(
        meta_learn_dataset.name,
        label_field,
        train_size=0.5,
        val_size=0.5,
        min_samples=3,
        overwrite=True,
        split_names=("train_metalearn", "val_metalearn"),
    )

    cfg = dict(
        data=dict(
            dataset=meta_learn_dataset.name,
            label_field=label_field,
            batch_size=2,
            transform_kwargs={"image_size": (224, 224)},
        ),
        model=dict(
            backbone="mobilenetv3_rw",
            training_strategy="prototypicalnetworks",
            training_strategy_kwargs=dict(
                ways=2,
                meta_batch_size=2,
                shots=2,
                queries=1,
                num_task=-1,
                epoch_length=2,
                test_ways=2,
                test_shots=2,
                test_queries=1,
            ),
            optimizer="AdamW",
            learning_rate=0.003,
        ),
        trainer=dict(
            epochs=2,
            save_checkpoint=str(model_path),
            strategy="freeze",
            device="cpu",
        ),
    )
    write_yaml(cfg, str(cfg_path))

    yield cfg_path, model_path, label_field
    cfg_path.unlink(False)
    model_path.unlink(True)
    if fo.dataset_exists(meta_learn_dataset.name):
        fo.delete_dataset(meta_learn_dataset.name)


def test_metalearning_finetune(meta_learn_cfg):
    import onnx

    cfg, model_path, label_field = meta_learn_cfg
    img_meta = ImageMetalearn()
    img_meta.finetune(cfg)
    assert model_path.exists()

    conf = read_yaml(cfg)
    split_dataset(conf["data"]["dataset"], splits=dict(support=0.6, query=0.4))
    img_meta.predict(
        support_dataset=conf["data"]["dataset"],
        support_label_field=label_field,
        query_dataset=conf["data"]["dataset"],
        query_label_field="new_label_prediction",
        ckpt_path=str(model_path),
        image_size=(280, 140),
        batch_size=4,
        support_kwargs=dict(include_tags=["support"]),
        query_kwargs=dict(include_tags=["query"]),
    )
    compute_hardness(
        conf["data"]["dataset"], "new_label_prediction", include_tags=["query"]
    )

    onnx_path = model_path.with_suffix(".onnx")
    img_meta.export_onnx(str(model_path), str(onnx_path), image_size=224)
    assert onnx_path.exists()
    onnx_model = onnx.load_model(str(onnx_path))
    onnx.checker.check_model(onnx_model)


def test_image_metalearn_triton_ensemble(tmp_path):
    img_meta = ImageMetalearn()
    img_meta.export_triton_ensemble(
        str(tmp_path / "triton"),
        "prototypical",
        version=1,
        preprocessing_name="image_transform",
        feature_extractor="feature_extractor",
        embedding_size=128,
    )
    assert (tmp_path / "triton" / "prototypical" / "1").exists()
    assert (tmp_path / "triton" / "prototypical" / "config.pbtxt").exists()


@pytest.fixture
def selfsupervised_config(clf_dataset, tmp_path):
    cfg_path = Path(tmp_path) / "embed_config.yaml"
    model_path = Path(tmp_path) / "embed_model.pt"
    cfg = dict(
        data=dict(
            dataset=clf_dataset.name,
            label_field="ground_truth",
            batch_size=2,
            transform_kwargs=dict(image_size=(224, 112)),
        ),
        model=dict(
            backbone="vision_transformer",
            pretrained=True,
            training_strategy="barlow_twins",
            head="barlow_twins_head",
            pretraining_transform="barlow_twins_transform",
        ),
        trainer=dict(
            epochs=2,
            save_checkpoint=str(model_path),
            limit_train_batches=2,
            strategy="freeze",
            device="cpu",
        ),
    )
    write_yaml(cfg, str(cfg_path))
    yield cfg_path, model_path
    cfg_path.unlink(False)
    model_path.unlink(True)


def test_selfsupervised_finetune(selfsupervised_config):
    cfg_path, model_path = selfsupervised_config
    img_emb = ImageSelfSupervised()
    img_emb.finetune(str(cfg_path))
    assert model_path.exists()


def test_image_transform(tmp_path):
    img_t = ImageTransform(size=112)
    img = torch.rand(224, 224, 3)
    out = img_t(img)

    assert out.size() == torch.Size([1, 3, 112, 112])
    assert out.dtype == torch.float32

    img_t.export_onnx(str(tmp_path / "img_t.onnx"))
    assert (tmp_path / "img_t.onnx").exists()
