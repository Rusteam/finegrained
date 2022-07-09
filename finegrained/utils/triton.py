"""Triton inference server and client utils.
"""
import shutil
from pathlib import Path
from typing import List

import torch
from google.protobuf import json_format, text_format
from tritonclient.grpc import model_config_pb2

from finegrained.utils import types


def save_triton_config(config: dict, write_file: str) -> None:
    """Dump dict config into protobuf file

    Args:
        config: a dict triton model config
        write_file: a path to write config

    Returns:
        None
    """
    parsed = json_format.ParseDict(config, model_config_pb2.ModelConfig())
    parsed_bytes = text_format.MessageToBytes(parsed)
    with open(write_file, "wb") as f:
        f.write(parsed_bytes)


def init_model_repo(triton_repo: str, triton_name: str, version: int):
    model_version_dir = Path(triton_repo) / triton_name / str(version)
    model_version_dir.mkdir(parents=True, exist_ok=True)
    return model_version_dir


def _export_txt_file(labels: List[str], filepath: Path):
    filepath.write_text("\n".join(labels))


class TritonExporter:
    def non_implemented(self):
        raise NotImplementedError("Subclass has to implement this method")

    @property
    def input_names(self):
        return ["image"]

    @property
    def output_names(self):
        return ["output"]

    @property
    def triton_batch_size(self):
        return 16

    @property
    def dynamic_axes(self):
        return {"image": {0: "batch_size"}, "output": {0: "batch_size"}}

    @property
    def triton_labels(self) -> List[str]:
        return None

    @property
    def triton_labels_path(self) -> str:
        return "labels.txt"

    def _create_triton_config(self, *args, **kwargs) -> dict:
        return {
            "backend": "onnxruntime",
            "max_batch_size": self.triton_batch_size,
        }

    @staticmethod
    def triton_python_file(self) -> str:
        self.non_implemented()

    def generate_dummy_inputs(self, *args, **kwargs) -> List[torch.Tensor]:
        self.non_implemented()

    def _load_model_torch(self, *args, **kwargs) -> torch.nn.Module:
        self.non_implemented()

    def _create_triton_python_backend(self):
        self.non_implemented()

    def _generate_triton_python_names(self, *args, **kwargs):
        self.non_implemented()

    def _create_triton_ensemble_config(self, *args, **kwargs):
        self.non_implemented()

    def export_onnx(
        self, model_path: str, write_path: str, image_size: types.IMAGE_SIZE
    ):
        """Create an ONNX model from a torch model.

        Args:
            model_path: flash training checkpoint
            write_path: where to save onnx model (*.onnx extension)
            image_size: model input size
        """
        model = self._load_model_torch(model_path)
        dummy = self.generate_dummy_inputs(image_size=image_size)
        torch.onnx.export(
            model,
            dummy,
            write_path,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes,
        )

    def export_triton(
        self,
        ckpt_path: str,
        triton_repo: str,
        triton_name: str,
        image_size: types.IMAGE_SIZE,
        version: int = 1,
    ):
        """Create a Triton model from a torch model.

        Args:
            ckpt_path: load a trained model from flash this checkpoint
            triton_repo: triton model repository path
            triton_name: triton model name
            image_size: model input size
            version: triton model version
        """
        model_version_dir = init_model_repo(triton_repo, triton_name, version)

        if ckpt_path is not None:
            self.export_onnx(
                ckpt_path,
                model_version_dir / "model.onnx",
                image_size=image_size,
            )

        if config := self._create_triton_config(image_size):
            write_config = model_version_dir.parent / "config.pbtxt"
            save_triton_config(config, write_config)

        if labels := self.triton_labels:
            _export_txt_file(
                labels, model_version_dir.parent / self.triton_labels_path
            )

        print(
            f"Triton-onnx model has been exported to {str(model_version_dir.parent)}"
        )

    def export_triton_ensemble(self, triton_repo: str, triton_name: str, version: int = 1,
                               **kwargs):
        model_version_dir = init_model_repo(triton_repo, triton_name, version)

        if config := self._create_triton_ensemble_config(**kwargs):
            write_config = model_version_dir.parent / "config.pbtxt"
            save_triton_config(config, write_config)

        print(f"Triton-ensemble has been exported to {str(model_version_dir.parent)}")

    def export_triton_python(
        self, triton_repo: str, triton_name: str, version: int = 1, **kwargs
    ):
        """Create a triton model with python backend.

        Args:
            triton_repo: triton model repository path
            triton_name: triton model name
            version: triton model version
            **kwargs: extra params to pass to _generate_triton_python_names
        """
        model_version_dir = init_model_repo(triton_repo, triton_name, version)

        shutil.copy(
            self.triton_python_file,
            model_version_dir / "model.py",
        )

        if config := self._create_triton_python_backend():
            write_config = model_version_dir.parent / "config.pbtxt"
            save_triton_config(config, write_config)

        if names := self._generate_triton_python_names(**kwargs):
            assert bool(kwargs), "Provide kwargs with dependent names"
            _export_txt_file(names, model_version_dir.parent / "names.txt")

        if "labels" in kwargs:
            shutil.copy(kwargs["labels"],
                        model_version_dir.parent / self.triton_labels_path)

        print(
            f"Triton-python model has been exported to {str(model_version_dir.parent)}"
        )
