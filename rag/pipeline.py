from rag.models import Document
from rag.chunk import Chunker
from rag.embed import Embedder
from rag.store import VectorStore
from rag.retrieve import Retriever
from rag.generate import Generator, Answer


class RagPipeline:
    def __init__(self, chunker: Chunker, embedder: Embedder, store: VectorStore,
                 retriever: Retriever, generator: Generator):
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.retriever = retriever
        self.generator = generator

    def index(self, docs: list[Document]) -> None:
        answers = [d for d in docs if d.question_id != d.post_id]
        chunks = []
        for doc in answers:
            chunks.extend(self.chunker.split(doc))
        if not chunks:
            return
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.add(chunks, vectors)

    def query(self, question: str, k: int = 5) -> Answer:
        retrieved = self.retriever.retrieve(question, k=k)
        return self.generator.answer(question, retrieved)
