import os
from rag.chunk import FixedSizeChunker
from rag.embed import SentenceTransformerEmbedder
from rag.store import ChromaStore
from rag.retrieve import DenseRetriever
from rag.generate import Generator, OpenRouterLLM
from rag.pipeline import RagPipeline

PERSIST_DIR = ".chroma"
COLLECTION = "cooking_baseline"
DEFAULT_MODEL = "anthropic/claude-sonnet-4"


def _model_name() -> str:
    return os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)


def build_pipeline() -> RagPipeline:
    embedder = SentenceTransformerEmbedder()
    store = ChromaStore(collection=COLLECTION, persist_dir=PERSIST_DIR)
    return RagPipeline(
        chunker=FixedSizeChunker(max_words=256, overlap=32),
        embedder=embedder,
        store=store,
        retriever=DenseRetriever(embedder=embedder, store=store),
        generator=Generator(llm=OpenRouterLLM(model=_model_name()), min_score=0.2),
    )
