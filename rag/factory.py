import os
from rag.chunk import FixedSizeChunker
from rag.embed import SentenceTransformerEmbedder
from rag.store import ChromaStore
from rag.retrieve import DenseRetriever, Retriever, BM25Retriever, HybridRetriever
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


def build_retriever(kind: str = "dense") -> Retriever:
    """dense | bm25 | hybrid. No LLM (no API key needed)."""
    if kind not in ("dense", "bm25", "hybrid"):
        raise ValueError(f"unknown retriever kind: {kind}")
    embedder, store = _embedder_and_store()
    if kind == "dense":
        return DenseRetriever(embedder=embedder, store=store)
    if kind == "bm25":
        return BM25Retriever(store.get_all_chunks())
    return HybridRetriever([
        DenseRetriever(embedder=embedder, store=store),
        BM25Retriever(store.get_all_chunks()),
    ])


def build_all_retrievers() -> dict[str, Retriever]:
    """All three retrievers, sharing one embedder/store/chunks/BM25 build."""
    embedder, store = _embedder_and_store()
    chunks = store.get_all_chunks()
    dense = DenseRetriever(embedder=embedder, store=store)
    sparse = BM25Retriever(chunks)
    return {"dense": dense, "bm25": sparse, "hybrid": HybridRetriever([dense, sparse])}


def build_pipeline() -> RagPipeline:
    embedder, store = _embedder_and_store()
    return RagPipeline(
        chunker=FixedSizeChunker(max_words=256, overlap=32),
        embedder=embedder,
        store=store,
        retriever=DenseRetriever(embedder=embedder, store=store),
        generator=Generator(llm=OpenRouterLLM(model=_model_name()), min_score=0.2),
    )
