from rag.models import Chunk
from rag.embed import FakeEmbedder
from rag.store import ChromaStore
from rag.retrieve import DenseRetriever


def test_dense_retriever_finds_relevant_chunk():
    emb = FakeEmbedder(dim=16)
    store = ChromaStore(collection="ret", persist_dir=None)
    chunks = [
        Chunk(text="mayonnaise emulsion broke", post_id=5570, question_id=5567),
        Chunk(text="how to bake sourdough bread", post_id=9001, question_id=9000),
    ]
    store.add(chunks, emb.embed([c.text for c in chunks]))

    retriever = DenseRetriever(embedder=emb, store=store)
    results = retriever.retrieve("mayonnaise emulsion broke", k=1)
    assert results[0].chunk.post_id == 5570
