"""Transformers tokenizer in triton's python backend format
"""
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer

# TODO link file instead of downloading
# TODO load from a name
# TODO load names from config


class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli"
        )

    def execute(self, requests):
        responses = [self._process_request(one) for one in requests]
        return responses

    def _process_request(self, request):
        texts = pb_utils.get_input_tensor_by_name(request, "text")
        texts = texts.as_numpy().tolist()
        texts = [t[0].decode() for t in texts]

        out = self.tokenizer(texts, padding=True, return_tensors="pt")

        input_ids = pb_utils.Tensor("input_ids", out["input_ids"])
        attention_masks = pb_utils.Tensor("attention_mask", out["attention_mask"])
        resp = pb_utils.InferenceResponse(output_tensors=[input_ids, attention_masks])
        return resp
