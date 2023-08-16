"""Utils to work with transformers package.
"""
import shutil
from pathlib import Path

import torch

from finegrained.models import torch_utils
from finegrained.utils import types
from finegrained.utils.triton import init_model_repo, save_triton_config


class MeanPooling(torch.nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        self._ = torch.nn.Identity()

    def forward(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        sizes = [
            token_embeddings[:, 0, 0].greater_equal(-1e6).sum(),
            token_embeddings[0, :, 0].greater_equal(-1e6).sum(),
            token_embeddings[0, 0, :].greater_equal(-1e6).sum(),
        ]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(sizes)
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )


class TransformersBase:
    def __init__(
        self,
        model_name: str,
        batch_size: int = torch_utils.get_default_batch_size(),
        device=torch_utils.get_device(),
    ):
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(device)
        self.embedding_dim = self._model.embeddings.token_type_embeddings.embedding_dim
        self.model_name = model_name
        self._batch_size = batch_size

    @property
    def batch_size(self):
        return self._batch_size

    def tokenizer(self, inputs: types.LIST_STR):
        tokens = self._tokenizer(
            inputs, padding=True, truncation=True, return_tensors="pt"
        )
        _ = [v.to(self.device) for v in tokens.values()]
        return tokens

    def model(self):
        raise NotImplementedError()


class SentenceEmbeddingsModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mean_pooling = MeanPooling()

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        embeddings = self.model(input_ids, attention_mask)
        embeddings = self.mean_pooling(embeddings, attention_mask)
        return embeddings


class SentenceEmbeddings(TransformersBase):
    """Extract features for text chunks."""

    @torch.no_grad()
    def model(self, tokens: dict) -> torch.Tensor:
        model_output = self._model(**tokens)
        embeddings = MeanPooling().forward(model_output, tokens["attention_mask"])
        return embeddings

    def predict(self, questions):
        if isinstance(questions, str):
            questions = [questions]

        # TODO add batch processing
        tokens = self.tokenizer(questions)
        embeddings = self.model(tokens)
        return embeddings

    def generate_dummy_input(self):
        dummy = ["This is sentence one", "And this is a second sentence"]
        tokens = self.tokenizer(dummy)
        return tokens.data["input_ids"], tokens.data["attention_mask"]

    @property
    def input_names(self):
        return ["input_ids", "attention_mask"]

    @property
    def output_names(self):
        return ["embeddings"]

    @property
    def dynamic_axes(self):
        return {
            "input_ids": {0: "batch_size", 1: "tokens"},
            "attention_mask": {0: "batch_size", 1: "tokens"},
            "embeddings": {0: "batch_size"},
        }

    def export_onnx(self, write_path: str) -> None:
        """Export model into onnx format

        Args:
            write_path:

        Returns:

        """
        # TODO finish with this
        # model = SentenceEmbeddingsModel(self._model.to("cpu"))
        dummy_tokens = self.generate_dummy_input()
        torch.onnx.export(
            self._model,
            dummy_tokens,
            write_path,
            export_params=True,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes,
            opset_version=13,
        )

    def export_triton(self, triton_repo: str, triton_name: str, version: int = 1):
        model_version_dir = init_model_repo(triton_repo, triton_name, version)
        self.export_onnx(model_version_dir / "model.onnx")

        config = self._create_triton_config()
        write_config = model_version_dir.parent / "config.pbtxt"
        save_triton_config(config, write_config)

    def _create_triton_config(self):
        return {
            "backend": "onnxruntime",
            "max_batch_size": self.batch_size,
        }

    def _create_ensemble_config(
        self,
    ) -> dict:
        return {
            "platform": "ensemble",
            "max_batch_size": self.batch_size,
            "input": [dict(name="TEXT", data_type="TYPE_STRING", dims=[-1])],
            "output": [
                dict(name="ATTENTION_MASK", data_type="TYPE_INT64", dims=[-1]),
                dict(
                    name="EMBEDDINGS",
                    data_type="TYPE_FP32",
                    dims=[-1, self.embedding_dim],
                ),
            ],
            "ensemble_scheduling": dict(
                step=[
                    dict(
                        model_name="sentence_tokenizer",
                        model_version=-1,
                        input_map=dict(text="TEXT"),
                        output_map=dict(
                            input_ids="INPUT_IDS",
                            attention_mask="ATTENTION_MASK",
                        ),
                    ),
                    dict(
                        model_name="sentence_embeddings",
                        model_version=-1,
                        input_map=dict(
                            input_ids="INPUT_IDS",
                            attention_mask="ATTENTION_MASK",
                        ),
                        output_map=dict(embeddings="EMBEDDINGS"),
                    ),
                ]
            ),
        }

    def export_ensemble(self, triton_repo, triton_name, version=1):
        model_version_dir = init_model_repo(triton_repo, triton_name, version)

        config = self._create_ensemble_config()
        write_file = model_version_dir.parent / "config.pbtxt"
        save_triton_config(config, write_file)

    def export_tokenizer(self, triton_repo, triton_name, version=1):
        model_version_dir = init_model_repo(triton_repo, triton_name, version)

        shutil.copy(
            Path(__file__).parents[1]
            / "utils"
            / "triton_python"
            / "transformers_tokenizer.py",
            model_version_dir / "model.py",
        )

        config = self._create_tokenizer_config()
        write_config = model_version_dir.parent / "config.pbtxt"
        save_triton_config(config, write_config)

    @staticmethod
    def _create_tokenizer_config():
        return {
            "backend": "python",
            "max_batch_size": 16,
            "input": [dict(name="text", data_type="TYPE_STRING", dims=[-1])],
            "output": [
                dict(name="input_ids", data_type="TYPE_INT64", dims=[-1]),
                dict(name="attention_mask", data_type="TYPE_INT64", dims=[-1]),
            ],
            "parameters": {
                "EXECUTION_ENV_PATH": {
                    "string_value": "/opt/tritonserver/backends/python/py39.tar.gz"
                }
            },
        }
