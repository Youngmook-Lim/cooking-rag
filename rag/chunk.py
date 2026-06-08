from abc import ABC, abstractmethod
from rag.models import Document, Chunk


class Chunker(ABC):
    @abstractmethod
    def split(self, doc: Document) -> list[Chunk]:
        ...


class FixedSizeChunker(Chunker):
    def __init__(self, max_words: int = 256, overlap: int = 32):
        if overlap >= max_words:
            raise ValueError("overlap must be smaller than max_words")
        self.max_words = max_words
        self.overlap = overlap

    def split(self, doc: Document) -> list[Chunk]:
        words = doc.body.split()
        if not words:
            return []
        step = self.max_words - self.overlap
        chunks: list[Chunk] = []
        for start in range(0, len(words), step):
            window = words[start:start + self.max_words]
            chunks.append(
                Chunk(
                    text=" ".join(window),
                    post_id=doc.post_id,
                    question_id=doc.question_id,
                )
            )
            if start + self.max_words >= len(words):
                break
        return chunks
