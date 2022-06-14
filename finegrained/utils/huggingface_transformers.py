"""Utils to work with transformers package.
"""
import torch
from transformers import AutoTokenizer, AutoModel


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class SentenceEmbeddings:
    def __init__(self, model_name: str, batch_size: int = 8):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # TODO move model to device
        self.model_name = model_name
        self.batch_size = batch_size

    @torch.no_grad()
    def embed(self, questions):

        if isinstance(questions, str):
            questions = [questions]

        # TODO add batch processing

        tokens = self.tokenizer(
            questions, padding=True, truncation=True, return_tensors="pt"
        )
        model_output = self.model(**tokens)
        embeddings = mean_pooling(model_output, tokens["attention_mask"])
        return embeddings
