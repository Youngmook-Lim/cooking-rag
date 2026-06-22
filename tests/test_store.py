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
    assert 0.0 <= results[0].score <= 1.0


def test_add_splits_into_batches():
    store = ChromaStore(collection="batch", persist_dir=None, batch_size=5)
    real_add = store._col.add
    sizes = []

    def spy(**kwargs):
        sizes.append(len(kwargs["ids"]))
        return real_add(**kwargs)

    store._col.add = spy
    emb = FakeEmbedder(dim=16)
    chunks = [Chunk(text=f"t{i}", post_id=i, question_id=1) for i in range(12)]
    store.add(chunks, emb.embed([c.text for c in chunks]))

    assert sizes == [5, 5, 2]
    assert store.count() == 12


def test_query_round_trips_chunk_index():
    store = _store()
    emb = FakeEmbedder(dim=16)
    chunks = [
        Chunk(text="second window text", post_id=5570, question_id=5567,
              chunk_index=1),
    ]
    store.add(chunks, emb.embed([c.text for c in chunks]))

    qvec = emb.embed(["second window text"])[0]
    results = store.query(qvec, k=1)
    assert results[0].chunk.chunk_index == 1


def test_get_all_chunks_round_trips():
    emb = FakeEmbedder(dim=16)
    store = ChromaStore(collection="all", persist_dir=None)
    chunks = [
        Chunk(text="alpha", post_id=1, question_id=1, chunk_index=0),
        Chunk(text="beta", post_id=2, question_id=1, chunk_index=1),
    ]
    store.add(chunks, emb.embed([c.text for c in chunks]))
    got = sorted(store.get_all_chunks(), key=lambda c: (c.post_id, c.chunk_index))
    assert [(c.text, c.post_id, c.question_id, c.chunk_index) for c in got] == [
        ("alpha", 1, 1, 0),
        ("beta", 2, 1, 1),
    ]
