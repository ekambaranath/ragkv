import pytest

from ragkv.data import reorder_chunks
from ragkv.data.synthetic import make_dataset
from ragkv.layout import build_layout
from ragkv.tiny import tiny_model


@pytest.fixture(scope="session")
def tiny():
    model, tok = tiny_model(seed=0)
    return model, tok


@pytest.fixture(scope="session")
def tiny_eager():
    model, tok = tiny_model(seed=0, attn="eager")
    return model, tok


@pytest.fixture(scope="session")
def examples():
    return make_dataset(6, k=5, filler=2, noise=1.0, seed=3)


@pytest.fixture
def layout(examples):
    _, tok = tiny_model(seed=0)
    ex = examples[0]
    chunks = reorder_chunks(ex["chunks"], "shuffle", seed=1)
    return build_layout(tok, ex["question"], chunks)
