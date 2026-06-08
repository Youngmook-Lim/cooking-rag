from abc import ABC, abstractmethod
from rag.embed import Embedder
from rag.store import VectorStore
from rag.models import Retrieved


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
