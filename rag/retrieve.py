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
        qvec = self.embedder.embed([query])[0]
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
