"""Test database indexing.
"""
import random

import numpy as np
import pytest

from finegrained.utils.milvus import MilvusClient


@pytest.fixture()
def setup(tmp_path):
    milvus = MilvusClient()
    name = "test_milvus"
    yield milvus, name
    milvus.drop_collection(name)


def test_index(setup):
    milvus, name = setup

    n_dim, n_samples = 16, 5
    fields = [
        dict(name="pk", dtype=int, is_primary=True, auto_id=True),
        dict(name="question", dtype=str),
        dict(name="answer", dtype=str),
        dict(name="embeddings", dtype=np.float32, dim=n_dim),
    ]
    milvus.create_collection(name, fields, "Test collection to be deleted")
    data = [
        [
            " ".join(random.sample("This is a question sample".split(), k=3))
            for _ in range(n_samples)
        ],
        [
            " ".join(random.sample("This will be an answer examples".split(), k=3))
            for _ in range(n_samples)
        ],
        [np.random.rand(n_dim) for _ in range(n_samples)],
    ]

    milvus.insert_vectors(data, name)
    index = dict(index_type="IVF_FLAT", metric_type="L2")
    milvus.build_index("embeddings", name, **index)
