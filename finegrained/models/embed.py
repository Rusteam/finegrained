"""Generate embeddings for input data.
"""
from typing import Dict, List

from finegrained.models.transformers_base import SentenceEmbeddings
from finegrained.utils.data import load_data, write_data
from finegrained.utils.similarity import SimilaritySearch


def text(model_name: str, input_data: str, input_data_key: str, output: str):
    """Generate embeddings for input data."""
    data = load_data(input_data, key=input_data_key)
    model = SentenceEmbeddings(model_name)
    embeddings = model.embed(data).numpy()
    if output is not None:
        write_data(output, embeddings)
    else:
        return data, embeddings


def most_similar(
    query: str, model_name: str, data: str, embeddings: str, top_k: int = 5
) -> List[Dict]:
    """Find most similar items in the data for the query.

    Args:
        query: Query to find most similar items.
        model_name: which model to embed the query with.
        data: Data to return for most similar items.
        embeddings: Embeddings to use for similarity search.
        top_k: number of top similar to return.

    Returns:
        Most similar items in the data for the query.
    """
    data = load_data(data)
    embeddings = load_data(embeddings)
    sim = SimilaritySearch(data.to_dict("records"), embeddings)
    model = SentenceEmbeddings(model_name)
    top_sim = sim.embed_and_find_similar(query, model, top_k=top_k)
    return top_sim
