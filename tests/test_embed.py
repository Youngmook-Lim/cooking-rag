import pytest
from rag.embed import FakeEmbedder


def test_fake_embedder_is_deterministic():
    emb = FakeEmbedder(dim=8)
    a = emb.embed(["hello world"])
    b = emb.embed(["hello world"])
    assert a == b
    assert len(a) == 1
    assert len(a[0]) == 8


def test_fake_embedder_differs_by_text():
    emb = FakeEmbedder(dim=8)
    [v1], [v2] = emb.embed(["mayonnaise"]), emb.embed(["bread"])
    assert v1 != v2


@pytest.mark.slow
def test_real_embedder_smoke():
    from rag.embed import SentenceTransformerEmbedder
    emb = SentenceTransformerEmbedder()
    vecs = emb.embed(["why did my mayonnaise break?"])
    assert len(vecs) == 1
    assert len(vecs[0]) == emb.dim
