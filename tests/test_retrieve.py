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


def test_dense_retriever_embeds_query_via_embed_query():
    from rag.embed import FakeEmbedder
    from rag.store import ChromaStore
    from rag.retrieve import DenseRetriever

    class RecordingEmbedder(FakeEmbedder):
        def __init__(self):
            super().__init__(dim=16)
            self.query_prefix = "Q: "
            self.query_calls = []

        def embed_query(self, texts):
            self.query_calls.append(list(texts))
            return super().embed_query(texts)

    emb = RecordingEmbedder()
    store = ChromaStore(collection="prefix", persist_dir=None)
    chunks = [Chunk(text="mayonnaise emulsion broke", post_id=5570, question_id=5567)]
    store.add(chunks, emb.embed([c.text for c in chunks]))

    retriever = DenseRetriever(embedder=emb, store=store)
    retriever.retrieve("why did my mayo break", k=1)
    assert emb.query_calls == [["why did my mayo break"]]


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


def test_hybrid_weight_downweights_a_retriever():
    from rag.retrieve import HybridRetriever, Retriever
    from rag.models import Retrieved

    def _r(post_id):
        return Retrieved(
            chunk=Chunk(text="x", post_id=post_id, question_id=1, chunk_index=0),
            score=0.0)

    class _Stub(Retriever):
        def __init__(self, results):
            self._results = results
        def retrieve(self, query, k):
            return self._results[:k]

    dense = _Stub([_r(9), _r(1)])   # post 1 at dense rank 2
    sparse = _Stub([_r(2)])         # post 2 at sparse rank 1

    equal = HybridRetriever([dense, sparse], weights=[1.0, 1.0])
    eq_ids = [r.chunk.post_id for r in equal.retrieve("q", k=5)]
    assert eq_ids.index(2) < eq_ids.index(1)   # sparse-only post 2 outranks dense post 1

    weighted = HybridRetriever([dense, sparse], weights=[1.0, 0.3])
    w_ids = [r.chunk.post_id for r in weighted.retrieve("q", k=5)]
    assert w_ids.index(1) < w_ids.index(2)     # down-weighting sparse flips the order


def test_reranking_retriever_overfetches_and_reorders():
    from rag.retrieve import RerankingRetriever, Retriever
    from rag.rerank import FakeReranker
    from rag.models import Retrieved

    def _ret(text, post_id):
        return Retrieved(
            chunk=Chunk(text=text, post_id=post_id, question_id=1), score=0.0)

    class _Base(Retriever):
        def __init__(self):
            self.asked_k = None
        def retrieve(self, query, k):
            self.asked_k = k
            return [
                _ret("best pan for steak", 1),
                _ret("mayonnaise emulsions break when oil added fast", 2),
                _ret("how to store mayonnaise", 3),
            ]

    base = _Base()
    retriever = RerankingRetriever(base, FakeReranker(), pool_size=50)
    out = retriever.retrieve("why did my mayonnaise break", k=2)
    assert base.asked_k == 50            # over-fetched the pool
    assert len(out) == 2                 # returned k
    assert out[0].chunk.post_id == 2     # relevant chunk floated to #1
