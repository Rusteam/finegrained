"""Test ONNX and Triton export.
"""
from pathlib import Path
from typing import List

import onnx
import pytest
import fiftyone as fo
import fiftyone.zoo as foz
import torch
from flash.image import ImageClassifier
from flash import Trainer
from onnxruntime import InferenceSession
from transformers.utils import to_numpy

from finegrained.models import ImageClassification, ImageTransform
from finegrained.models.image_classification import SoftmaxClassifier


@pytest.fixture(scope="module")
def clf_dataset():
    if fo.dataset_exists("train_test_temp"):
        fo.delete_dataset("train_test_temp")
    dataset = (
        foz.load_zoo_dataset("quickstart").take(100).clone("train_test_temp")
    )
    yield dataset
    fo.delete_dataset(dataset.name)


@pytest.fixture(scope="module")
def triton_repo(tmp_path_factory):
    triton_repo = tmp_path_factory.mktemp("triton")
    return triton_repo


@pytest.fixture
def image_classifier_ckpt(clf_dataset, tmp_path):
    ckpt_path = tmp_path / "img_clf_ckpt.pt"
    model = ImageClassifier(labels=list("ABCDAEFG"), backbone="resnet18")
    trainer = Trainer(max_epochs=1, limit_train_batches=2)
    trainer.model = model
    trainer.save_checkpoint(ckpt_path)
    yield ckpt_path
    ckpt_path.unlink(missing_ok=True)


def test_image_clf(image_classifier_ckpt, triton_repo):
    image_size = (250, 125)

    img_clf = ImageClassification()
    img_clf.export_triton(
        ckpt_path=str(image_classifier_ckpt),
        triton_repo=str(triton_repo),
        triton_name="image_classifier",
        image_size=image_size,
        version=1,
    )

    triton_model_path = triton_repo / "image_classifier"
    onnx_path = triton_model_path / "1" / "model.onnx"
    _check_triton_onnx(triton_model_path)
    _check_triton_labels(triton_model_path)
    _check_onnx_model(onnx_path)

    dummy = img_clf.generate_dummy_inputs(image_size)
    torch_model = SoftmaxClassifier(img_clf.model)
    _verify_model_outputs(torch_model, str(onnx_path), dummy)


def test_image_clf_ensemble(triton_repo, tmp_path):

    img_clf = ImageClassification()
    img_clf.export_triton_ensemble(
        triton_repo=str(triton_repo),
        triton_name="img_clf_model",
        version=1,
        preprocessing_name="prep",
        classifier_name="clf",
    )

    triton_model_path = triton_repo / "img_clf_model"
    _check_triton_ensemble(triton_model_path)


def test_img_transforms(triton_repo):
    image_size = (200, 100)

    img_T = ImageTransform(image_size)
    img_T.export_triton(
        ckpt_path="",
        triton_repo=triton_repo,
        triton_name="image_prep",
        image_size=image_size,
    )

    triton_model_path = triton_repo / "image_prep"
    onnx_path = triton_model_path / "1" / "model.onnx"
    _check_triton_onnx(triton_model_path)
    _check_onnx_model(onnx_path)

    dummy = img_T.generate_dummy_inputs(image_size)
    _verify_model_outputs(img_T, str(onnx_path), dummy)


def _verify_model_outputs(
    model: torch.nn.Module, onnx_path: str, dummy: List[torch.Tensor]
):
    ort = InferenceSession(str(onnx_path))
    input_names = [inp.name for inp in ort.get_inputs()]
    output_names = [out.name for out in ort.get_outputs()]
    input_feed = {k: to_numpy(v) for k, v in zip(input_names, dummy)}
    ort_out = ort.run(output_names, input_feed)

    model.eval()
    with torch.no_grad():
        model_out = model(*dummy)

    torch.testing.assert_allclose(
        model_out, torch.tensor(ort_out[0]), atol=1e-2, rtol=5e-2
    )


def _check_triton_onnx(model_dir):
    assert model_dir.exists()
    assert (model_dir / "config.pbtxt").exists()
    assert (model_dir / "1" / "model.onnx").exists()


def _check_triton_python(model_dir, version: int = 1, with_names=False):
    assert (model_dir / str(version) / "model.py").exists()
    assert (model_dir / "config.pbtxt").exists()

    if with_names:
        assert (model_dir / "names.txt").exists()


def _check_triton_ensemble(model_dir):
    assert (model_dir / "1").exists()
    assert (model_dir / "config.pbtxt").exists()


def _check_triton_labels(model_dir):
    assert (model_dir / "labels.txt").exists()


def _check_onnx_model(onnx_path):
    onnx_model = onnx.load_model(onnx_path)
    onnx.checker.check_model(onnx_model)
