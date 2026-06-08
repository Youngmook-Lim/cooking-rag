from rag.chunk import FixedSizeChunker
from rag.embed import SentenceTransformerEmbedder
from rag.store import ChromaStore
from rag.retrieve import DenseRetriever
from rag.generate import Generator, ClaudeLLM
from rag.pipeline import RagPipeline

PERSIST_DIR = ".chroma"
COLLECTION = "cooking_baseline"


def build_pipeline() -> RagPipeline:
    embedder = SentenceTransformerEmbedder()
    store = ChromaStore(collection=COLLECTION, persist_dir=PERSIST_DIR)
    return RagPipeline(
        chunker=FixedSizeChunker(max_words=256, overlap=32),
        embedder=embedder,
        store=store,
        retriever=DenseRetriever(embedder=embedder, store=store),
        generator=Generator(llm=ClaudeLLM(), min_score=0.2),
    )
