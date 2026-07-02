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


def test_embed_query_defaults_to_no_prefix():
    from rag.embed import FakeEmbedder
    emb = FakeEmbedder(dim=16)
    assert emb.embed_query(["hello"]) == emb.embed(["hello"])


def test_embed_query_applies_query_prefix():
    from rag.embed import FakeEmbedder
    emb = FakeEmbedder(dim=16)
    emb.query_prefix = "Q: "
    assert emb.embed_query(["hello"]) == emb.embed(["Q: hello"])


@pytest.mark.slow
def test_real_embedder_smoke():
    from rag.embed import SentenceTransformerEmbedder
    emb = SentenceTransformerEmbedder()
    vecs = emb.embed(["why did my mayonnaise break?"])
    assert len(vecs) == 1
    assert len(vecs[0]) == emb.dim


@pytest.mark.slow
def test_sentence_transformer_query_prefix_changes_vector():
    from rag.embed import SentenceTransformerEmbedder, BGE_QUERY_PREFIX
    emb = SentenceTransformerEmbedder()
    assert emb.query_prefix == BGE_QUERY_PREFIX
    q = "why did my mayonnaise break"
    assert emb.embed_query([q]) != emb.embed([q])          # prefix changes the vector

    plain = SentenceTransformerEmbedder(query_prefix="")
    assert plain.embed_query([q]) == plain.embed([q])      # override disables it
