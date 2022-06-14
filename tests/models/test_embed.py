from pathlib import Path

import numpy as np
import pytest

from finegrained.models import embed
from finegrained.utils.data import write_data, load_data


@pytest.fixture
def text_file(tmp_path):
    file = Path(tmp_path) / "test.csv"
    out_file = Path(tmp_path) / "embed.npy"
    write_data(file, [{"text": "foo foo foo", "label": "bar"},
                      {"text": "bar bad bulk", "label": "baz"},
                      {"text": "baz bazooka", "label": "foo"}])
    yield file, out_file, ("text", "label")
    file.unlink(missing_ok=True)
    out_file.unlink(missing_ok=True)


@pytest.fixture
def embeddings_file(tmp_path):
    file = Path(tmp_path) / "embeddings.npy"
    write_data(file, np.random.randn(3, 768))
    yield file
    file.unlink(missing_ok=True)


@pytest.mark.parametrize("model_name,embeddings_dim",
                         [("symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli",
                           768)])
def test_text_embeddings(text_file, model_name, embeddings_dim):
    file, write_file, keys = text_file
    embed.text(model_name, file, keys[0], write_file)
    embeddings = load_data(write_file)
    assert embeddings.shape == (3, embeddings_dim)


@pytest.mark.parametrize("query,model_name,top_k",
                         [("foo foo foo",
                           "symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli",
                           1),
                          ("foo foo foo",
                           "symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli",
                           3)
                          ])
def test_most_similar(query, model_name, top_k, text_file, embeddings_file):
    data_file, *_ = text_file
    sim = embed.most_similar(query, model_name, data=data_file,
                             embeddings=embeddings_file,
                             top_k=top_k)
    assert len(sim) == top_k
    for one in sim:
        assert "text" in one
        assert "label" in one
