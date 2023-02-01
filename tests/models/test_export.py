"""Test ONNX and Triton export.
"""
from typing import List

import fiftyone as fo
import fiftyone.zoo as foz
import onnx
import pytest
import torch
from flash import Trainer
from flash.image import ImageClassifier
from onnxruntime import InferenceSession
from transformers.utils import to_numpy

from finegrained.models import ImageClassification, ImageTransform
from finegrained.models.image_classification import SoftmaxClassifier
from finegrained.models.yolo import (
    YOLOv5Model,
    YOLOv5Postprocessing,
    YOLOv5Preprocessing,
)


@pytest.fixture(scope="module")
def clf_dataset():
    if fo.dataset_exists("train_test_temp"):
        fo.delete_dataset("train_test_temp")
    dataset = foz.load_zoo_dataset("quickstart").take(100).clone("train_test_temp")
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


def test_yolov5_model(triton_repo, tmp_path):
    src = tmp_path / "temp_file.onnx"
    open(src, "wb").close()

    model = YOLOv5Model()
    model.export_triton(
        ckpt_path=str(src),
        triton_repo=triton_repo,
        triton_name="v5_model",
        version=1,
    )

    _check_triton_onnx(triton_repo / "v5_model")

    model.export_triton_ensemble(
        triton_repo=str(triton_repo),
        triton_name="v5_ensemble",
        version=1,
        preprocessing_name="v5_prep",
        model_name="v5_model",
        postprocessing_name="v5_post",
    )

    _check_triton_ensemble(triton_repo / "v5_model")


def test_yolov5_prep(triton_repo):
    image_size = 320
    v5_prep = YOLOv5Preprocessing(image_size=image_size)
    v5_prep.export_triton(
        ckpt_path="",
        triton_repo=triton_repo,
        triton_name="v5_prep",
    )

    triton_model_path = triton_repo / "v5_prep"
    onnx_path = triton_model_path / "1" / "model.onnx"
    _check_triton_onnx(triton_model_path)
    _check_onnx_model(onnx_path)

    dummy = v5_prep.generate_dummy_inputs()
    _verify_model_outputs(v5_prep, str(onnx_path), dummy)


@pytest.mark.parametrize("torchscript", [False, True])
def test_yolov5_post(triton_repo, torchscript):
    v5_post = YOLOv5Postprocessing()
    v5_post.export_triton(
        ckpt_path="",
        triton_repo=triton_repo,
        triton_name="v5_post",
        torchscript=torchscript,
    )

    triton_model_path = triton_repo / "v5_post"
    if torchscript:
        _check_triton_torchscript(triton_model_path)
    else:
        _check_onnx_model(triton_model_path / "1" / "model.onnx")
        _check_triton_onnx(triton_model_path)

    dummy = v5_post.generate_dummy_inputs()
    ext = ".pt" if torchscript else ".onnx"
    _verify_model_outputs(v5_post, str(triton_model_path / "1" / f"model{ext}"), dummy)


def _verify_model_outputs(
    model: torch.nn.Module, exported_path: str, dummy: List[torch.Tensor]
):
    if exported_path.endswith(".onnx"):
        ort = InferenceSession(exported_path, providers=["CPUExecutionProvider"])
        input_names = [inp.name for inp in ort.get_inputs()]
        output_names = [out.name for out in ort.get_outputs()]
        input_feed = {k: to_numpy(v) for k, v in zip(input_names, dummy)}
        actual = ort.run(output_names, input_feed)
    elif exported_path.endswith(".pt"):
        jit_model = torch.jit.load(exported_path)
        actual = jit_model(*dummy)
    else:
        raise ValueError(f"{exported_path} extension not supported")

    model.eval()
    with torch.no_grad():
        model_out = model(*dummy)

    if isinstance(model_out, torch.Tensor):
        model_out = [model_out]

    for expected, actual in zip(model_out, actual):
        torch.testing.assert_allclose(
            expected, torch.tensor(actual), atol=1e-2, rtol=5e-2
        )


def _check_triton_onnx(model_dir):
    assert model_dir.exists()
    assert (model_dir / "config.pbtxt").exists()
    assert (model_dir / "1" / "model.onnx").exists()


def _check_triton_torchscript(model_dir):
    assert model_dir.exists()
    assert (model_dir / "config.pbtxt").exists()
    assert (model_dir / "1" / "model.pt").exists()


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
