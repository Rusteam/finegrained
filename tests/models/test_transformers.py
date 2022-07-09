"""Test ONNX and Triton export.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
from onnxruntime import InferenceSession
from transformers.utils import to_numpy

from finegrained.models import SentenceEmbeddings
from tests.models.test_export import _check_triton_onnx, _check_triton_python, \
    _check_triton_ensemble

SENTENCE_EMBEDDINGS = ["symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli"]


@pytest.mark.parametrize("model_name", SENTENCE_EMBEDDINGS)
def test_onnx(model_name, tmp_path):
    write_path = str(Path(tmp_path) / "model.onnx")
    sent_emb = SentenceEmbeddings(model_name=model_name, device="cpu")
    sent_emb.export_onnx(write_path)

    ort = InferenceSession(write_path)

    dummy = sent_emb.generate_dummy_input()
    with torch.no_grad():
        model_out = sent_emb._model(*dummy)

    input_feed = {
        k: to_numpy(v)
        for k, v in zip(sent_emb.input_names, dummy)
    }
    ort_out = ort.run(
        output_names=sent_emb.output_names, input_feed=input_feed
    )

    np.testing.assert_allclose(
        model_out[0].numpy(), ort_out[0], atol=1e-4, rtol=1e-2
    )


@pytest.mark.parametrize("model_name", SENTENCE_EMBEDDINGS)
def test_triton(model_name, tmp_path):
    repo_dir = str(Path(tmp_path) / "triton")
    sent_emb = SentenceEmbeddings(model_name=model_name, device="cpu")
    sent_emb.export_triton(repo_dir, "sent_emb", version=1)

    export_dir = Path(repo_dir) / "sent_emb"
    _check_triton_onnx(export_dir)


@pytest.mark.parametrize("model_name", SENTENCE_EMBEDDINGS)
def test_triton_python_backend(model_name, tmp_path):
    model = SentenceEmbeddings(model_name)
    export_dir = Path(tmp_path) / "triton"
    model.export_tokenizer(export_dir, "tokenizer", version=1)
    _check_triton_python(export_dir)


@pytest.mark.parametrize("model_name", SENTENCE_EMBEDDINGS)
def test_triton_ensemble(model_name, tmp_path):
    model = SentenceEmbeddings(model_name)
    export_dir = Path(tmp_path) / "triton"
    model.export_ensemble(export_dir, "ensemble", version=1)
    _check_triton_ensemble(export_dir)
