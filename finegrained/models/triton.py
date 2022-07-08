"""Triton inference server and client utils.
"""
from pathlib import Path

from google.protobuf import json_format, text_format
from tritonclient.grpc import model_config_pb2


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