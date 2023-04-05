"""Triton inference server and client utils.
"""
import shutil
from pathlib import Path
from typing import List

import torch
from google.protobuf import json_format, text_format
from onnxruntime import InferenceSession


def save_triton_config(config: dict, write_file: str) -> None:
    """Dump dict config into protobuf file

    Args:
        config: a dict triton model config
        write_file: a path to write config

    Returns:
        None
    """
    from tritonclient.grpc import model_config_pb2

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
        return []

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

    def generate_dummy_inputs(self, *args, **kwargs) -> tuple[torch.Tensor]:
        self.non_implemented()

    def _load_model_torch(self, *args, **kwargs) -> torch.nn.Module:
        self.non_implemented()

    def _create_triton_python_backend(self):
        self.non_implemented()

    def _generate_triton_python_names(self, *args, **kwargs):
        self.non_implemented()

    def _create_triton_ensemble_config(self, *args, **kwargs) -> dict:
        self.non_implemented()

    def export_onnx(self, model_path: str, write_path: str = "auto", **kwargs):
        """Create an ONNX model from a torch model.

        Args:
            model_path: flash training checkpoint
            write_path: where to save onnx model (*.onnx extension),
                if default 'auto' then save to the same dir as model_path
                but with *.onnx extension
            kwargs: keyword arguments to pass to generating dummy inputs
        """
        # TODO create output dir if not exist
        if write_path == "auto":
            write_path = Path(model_path).with_suffix(".onnx")
            if write_path.exists():
                raise FileExistsError(f"{write_path=!r} already exists")

        model = self._load_model_torch(model_path)
        dummy = self.generate_dummy_inputs(**kwargs)
        torch.onnx.export(
            model,
            dummy,
            str(write_path),
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes,
        )

    def export_torchscirpt(self, write_path, **kwargs):
        model = self._load_model_torch(**kwargs)
        dummy = self.generate_dummy_inputs(**kwargs)
        scripted = torch.jit.script(model, example_inputs=dummy)
        torch.jit.save(scripted, write_path)

    def export_triton(
        self,
        ckpt_path: str,
        triton_repo: str,
        triton_name: str,
        version: int = 1,
        torchscript: bool = False,
        **kwargs,
    ):
        """Create a Triton model from a torch model.

        Args:
            ckpt_path: load a trained model from flash this checkpoint
            triton_repo: triton model repository path
            triton_name: triton model name
            version: triton model version
            torchscript: if True, then export in torchscript format rather
                than ONNX
        """
        model_version_dir = init_model_repo(triton_repo, triton_name, version)

        if ckpt_path is not None:
            if torchscript:
                self.export_torchscirpt(
                    write_path=model_version_dir / "model.pt", **kwargs
                )
            else:
                self.export_onnx(
                    ckpt_path, str(model_version_dir / "model.onnx"), **kwargs
                )

        if config := self._create_triton_config(torchscript=torchscript, **kwargs):
            write_config = model_version_dir.parent / "config.pbtxt"
            save_triton_config(config, str(write_config))

        if labels := self.triton_labels:
            _export_txt_file(labels, model_version_dir.parent / self.triton_labels_path)

        print(f"Triton-onnx model has been exported to {str(model_version_dir.parent)}")

    def export_triton_ensemble(
        self, triton_repo: str, triton_name: str, version: int = 1, **kwargs
    ):
        """Create a triton ensemble model.

        Args:
            triton_repo: triton model repository path
            triton_name: triton model name
            version: triton model version
            kwargs: extra params to pass to _create_triton_ensemble_config
        """
        model_version_dir = init_model_repo(triton_repo, triton_name, version)

        if config := self._create_triton_ensemble_config(**kwargs):
            write_config = model_version_dir.parent / "config.pbtxt"
            save_triton_config(config, str(write_config))

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
            shutil.copy(
                kwargs["labels"],
                model_version_dir.parent / self.triton_labels_path,
            )

        print(
            f"Triton-python model has been exported to {str(model_version_dir.parent)}"
        )

    @staticmethod
    def print_onnx(model_path: str):
        """Print ONNX model inputs and outputs.

        Args:
            model_path: path to the ONNX model
        """
        ort = InferenceSession(model_path)
        print("INPUTS:")
        for inp in ort.get_inputs():
            print(">>>", inp)
        print("\nOUTPUTS:")
        for out in ort.get_outputs():
            print(">>>", out)
