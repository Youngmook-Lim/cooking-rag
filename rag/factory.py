import os
from rag.chunk import FixedSizeChunker
from rag.embed import SentenceTransformerEmbedder
from rag.store import ChromaStore
from rag.retrieve import DenseRetriever, Retriever
from rag.generate import Generator, OpenRouterLLM
from rag.pipeline import RagPipeline

PERSIST_DIR = ".chroma"
COLLECTION = "cooking_baseline"
DEFAULT_MODEL = "anthropic/claude-sonnet-4"


def _model_name() -> str:
    return os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)


def _embedder_and_store():
    embedder = SentenceTransformerEmbedder()
    store = ChromaStore(collection=COLLECTION, persist_dir=PERSIST_DIR)
    return embedder, store


def build_retriever() -> Retriever:
    """Embedder + persisted store + dense retriever. No LLM (no API key needed)."""
    embedder, store = _embedder_and_store()
    return DenseRetriever(embedder=embedder, store=store)


def build_pipeline() -> RagPipeline:
    embedder, store = _embedder_and_store()
    return RagPipeline(
        chunker=FixedSizeChunker(max_words=256, overlap=32),
        embedder=embedder,
        store=store,
        retriever=DenseRetriever(embedder=embedder, store=store),
        generator=Generator(llm=OpenRouterLLM(model=_model_name()), min_score=0.2),
    )
