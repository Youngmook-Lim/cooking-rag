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


def test_bm25_retriever_ranks_by_keyword():
    from rag.retrieve import BM25Retriever
    chunks = [
        Chunk(text="mayonnaise emulsion broke", post_id=5570, question_id=5567),
        Chunk(text="how to fix a broken emulsion", post_id=8000, question_id=7000),
        Chunk(text="best pan for searing steak", post_id=9001, question_id=9000),
    ]
    retriever = BM25Retriever(chunks)
    results = retriever.retrieve("broken emulsion", k=2)
    assert results[0].chunk.post_id == 8000
    assert isinstance(results[0].score, float)
    assert all(r.chunk.post_id != 9001 for r in results)


def test_hybrid_rrf_fuses_two_rankings():
    from rag.retrieve import HybridRetriever, Retriever
    from rag.models import Retrieved

    def _r(post_id, score):
        return Retrieved(
            chunk=Chunk(text="x", post_id=post_id, question_id=1, chunk_index=0),
            score=score)

    class _Stub(Retriever):
        def __init__(self, results):
            self._results = results
        def retrieve(self, query, k):
            return self._results[:k]

    dense = _Stub([_r(1, .9), _r(2, .8), _r(3, .7)])
    sparse = _Stub([_r(3, 5.0), _r(1, 4.0), _r(9, 3.0)])
    hybrid = HybridRetriever([dense, sparse], k_rrf=60)
    results = hybrid.retrieve("q", k=4)
    # RRF: post1 ranks (1,2); post3 ranks (3,1); post2 (2,-); post9 (-,3)
    assert [r.chunk.post_id for r in results] == [1, 3, 2, 9]
    # an item ranked by both retrievers outranks items ranked by only one
    assert results[0].chunk.post_id == 1
