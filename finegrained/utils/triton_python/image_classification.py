"""Transformers tokenizer in triton's python backend format
"""
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils


def _read_names(repo) -> dict:
    names = (Path(repo) / "names.txt").read_text().strip().split("\n")
    names = {one.split("=")[0]: one.split("=")[1] for one in names}
    return names


# TODO unable to send batched requests with varying sizes
class TritonPythonModel:
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        inputs = [dict(name="IMAGE", dims=[-1, -1, 3], data_type="TYPE_UINT8")]
        outputs = [dict(name="CLASS_PROBS", dims=[-1], data_type="TYPE_FP32")]

        config = auto_complete_model_config.as_dict()
        existing_inputs = [inp["name"] for inp in config["input"]]
        existing_outputs = [inp["name"] for inp in config["output"]]

        for inp in inputs:
            if inp["name"] not in existing_inputs:
                auto_complete_model_config.add_input(inp)

        for out in outputs:
            if out["name"] not in existing_outputs:
                auto_complete_model_config.add_output(out)

        return auto_complete_model_config

    def initialize(self, args):
        self._names = _read_names(args["model_repository"])

    def execute(self, requests):
        responses = [self._process_request(one) for one in requests]
        return responses

    def _process_request(self, request):
        images = pb_utils.get_input_tensor_by_name(request, "IMAGE")
        images = images.as_numpy()

        preprocessed = [self._preprocess(img) for img in images]
        preprocessed = np.vstack(preprocessed)
        class_probs = self._classify(preprocessed)

        output = pb_utils.Tensor("CLASS_PROBS", class_probs)
        resp = pb_utils.InferenceResponse(output_tensors=[output])
        return resp

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        images = image[np.newaxis, ...]
        request = pb_utils.InferenceRequest(
            model_name=self._names["preprocessing"],
            requested_output_names=["output"],
            inputs=[pb_utils.Tensor("image", images)],
        )
        resp = request.exec()
        self._raise_for_error(resp)

        output = pb_utils.get_output_tensor_by_name(resp, "output")
        return output.as_numpy().astype(np.float32)

    def _classify(self, images: np.ndarray) -> np.ndarray:
        request = pb_utils.InferenceRequest(
            model_name=self._names["classifier"],
            requested_output_names=["output"],
            inputs=[pb_utils.Tensor("image", images)],
        )
        resp = request.exec()
        self._raise_for_error(resp)

        output = pb_utils.get_output_tensor_by_name(resp, "output")
        logits = output.as_numpy()
        probs = np.exp(logits) / np.exp(logits).sum(1)
        return probs

    def _raise_for_error(self, inference_response):
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
