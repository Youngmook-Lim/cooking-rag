from abc import ABC, abstractmethod
from rag.models import Retrieved


def _tokens(text: str) -> set[str]:
    return set(text.lower().split())


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, retrieved: list[Retrieved],
               k: int) -> list[Retrieved]:
        ...


class FakeReranker(Reranker):
    """Offline reranker scoring by query/chunk word overlap. Not semantic."""

    def rerank(self, query: str, retrieved: list[Retrieved],
               k: int) -> list[Retrieved]:
        q = _tokens(query)
        scored = [
            Retrieved(chunk=r.chunk, score=float(len(q & _tokens(r.chunk.text))))
            for r in retrieved
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, retrieved: list[Retrieved],
               k: int) -> list[Retrieved]:
        if not retrieved:
            return []
        scores = self._model.predict([(query, r.chunk.text) for r in retrieved])
        scored = [
            Retrieved(chunk=r.chunk, score=float(s))
            for r, s in zip(retrieved, scores)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]
