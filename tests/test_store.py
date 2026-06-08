from rag.models import Chunk
from rag.embed import FakeEmbedder
from rag.store import ChromaStore


def _store():
    return ChromaStore(collection="test", persist_dir=None)  # ephemeral


def test_add_and_count():
    store = _store()
    emb = FakeEmbedder(dim=16)
    chunks = [Chunk(text="mayonnaise emulsion", post_id=5570, question_id=5567)]
    store.add(chunks, emb.embed([c.text for c in chunks]))
    assert store.count() == 1


def test_query_returns_nearest_with_metadata():
    store = _store()
    emb = FakeEmbedder(dim=16)
    chunks = [
        Chunk(text="mayonnaise emulsion broke", post_id=5570, question_id=5567),
        Chunk(text="how to bake sourdough bread", post_id=9001, question_id=9000),
    ]
    store.add(chunks, emb.embed([c.text for c in chunks]))

    qvec = emb.embed(["mayonnaise emulsion broke"])[0]
    results = store.query(qvec, k=1)
    assert len(results) == 1
    assert results[0].chunk.post_id == 5570
    assert results[0].chunk.question_id == 5567
    assert isinstance(results[0].score, float)
