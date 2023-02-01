"""Utils for similarity search.
"""
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.spatial.distance import cdist

from finegrained.models.transformers_base import SentenceEmbeddings


@dataclass
class SimilaritySearch:
    data: List[Dict]
    embeddings: np.ndarray

    def find_similar(self, query_vector: np.ndarray, top_k=5):
        query_vector = np.atleast_2d(query_vector)
        # TODO normalized distances
        dist = cdist(query_vector, self.embeddings)[0]
        top_inds = np.argsort(dist)[:top_k]
        items = [self._get_item(i, dist[i]) for i in top_inds]
        return items

    def _get_item(self, index: int, distance: float):
        d = deepcopy(self.data[index])
        d.update(dict(distance=distance))
        return d

    def embed_and_find_similar(self, query: str, model: SentenceEmbeddings, **kwargs):
        query_embed = model.embed(query).numpy()
        return self.find_similar(query_embed, **kwargs)
