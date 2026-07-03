from abc import ABC, abstractmethod
from rag.embed import Embedder
from rag.store import VectorStore
from rag.models import Chunk, Retrieved
from rag.bm25 import BM25Index


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        ...


class DenseRetriever(Retriever):
    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        qvec = self.embedder.embed_query([query])[0]
        return self.store.query(qvec, k=k)


class BM25Retriever(Retriever):
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.index = BM25Index([c.text for c in chunks])

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        return [
            Retrieved(chunk=self.chunks[doc_id], score=score)
            for doc_id, score in self.index.search(query, k)
        ]


class HybridRetriever(Retriever):
    def __init__(self, retrievers: list[Retriever], weights: list[float] | None = None,
                 k_rrf: int = 60):
        self.retrievers = retrievers
        self.weights = weights if weights is not None else [1.0] * len(retrievers)
        self.k_rrf = k_rrf

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        fused: dict[tuple[int, int], float] = {}
        chosen: dict[tuple[int, int], Retrieved] = {}
        for weight, retriever in zip(self.weights, self.retrievers):
            for rank, r in enumerate(retriever.retrieve(query, k), start=1):
                key = (r.chunk.post_id, r.chunk.chunk_index)
                fused[key] = fused.get(key, 0.0) + weight * 1.0 / (self.k_rrf + rank)
                chosen.setdefault(key, r)
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [Retrieved(chunk=chosen[key].chunk, score=score)
                for key, score in ranked]
