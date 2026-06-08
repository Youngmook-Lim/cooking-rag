import hashlib
import struct
from abc import ABC, abstractmethod


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class FakeEmbedder(Embedder):
    """Deterministic hash-based embeddings for tests. Not semantically meaningful."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec: list[float] = []
            seed = text.encode("utf-8")
            i = 0
            while len(vec) < self.dim:
                h = hashlib.sha256(seed + struct.pack("i", i)).digest()
                for j in range(0, len(h), 4):
                    if len(vec) >= self.dim:
                        break
                    n = struct.unpack("I", h[j:j + 4])[0]
                    vec.append((n % 1000) / 1000.0)
                i += 1
            out.append(vec)
        return out


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]
