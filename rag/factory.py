import os
from rag.chunk import FixedSizeChunker
from rag.embed import SentenceTransformerEmbedder
from rag.store import ChromaStore
from rag.retrieve import (DenseRetriever, Retriever, BM25Retriever,
                         HybridRetriever, RerankingRetriever)
from rag.rerank import CrossEncoderReranker
from rag.generate import Generator, OpenRouterLLM
from rag.pipeline import RagPipeline

PERSIST_DIR = ".chroma"
COLLECTION = "cooking_baseline"
DEFAULT_MODEL = "anthropic/claude-sonnet-4"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"


def _model_name() -> str:
    return os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)


def _judge_model_name() -> str:
    return os.environ.get("OPENROUTER_JUDGE_MODEL", _model_name())


def build_llm(temperature: float | None = None) -> OpenRouterLLM:
    return OpenRouterLLM(model=_model_name(), temperature=temperature)


def build_judge_llm(temperature: float | None = 0.0) -> OpenRouterLLM:
    return OpenRouterLLM(model=_judge_model_name(), temperature=temperature)


def _embedder_and_store():
    embedder = SentenceTransformerEmbedder()
    store = ChromaStore(collection=COLLECTION, persist_dir=PERSIST_DIR)
    return embedder, store


def build_dense_and_bm25() -> tuple[Retriever, Retriever]:
    """Shared dense + BM25 build (embedder, store, chunks, BM25 index built once)."""
    embedder, store = _embedder_and_store()
    chunks = store.get_all_chunks()
    return DenseRetriever(embedder=embedder, store=store), BM25Retriever(chunks)


def build_retriever(kind: str = "dense") -> Retriever:
    """dense | bm25 | hybrid | rerank. No LLM (no API key needed)."""
    if kind not in ("dense", "bm25", "hybrid", "rerank"):
        raise ValueError(f"unknown retriever kind: {kind}")
    embedder, store = _embedder_and_store()
    if kind == "dense":
        return DenseRetriever(embedder=embedder, store=store)
    if kind == "bm25":
        return BM25Retriever(store.get_all_chunks())
    if kind == "rerank":
        return RerankingRetriever(
            DenseRetriever(embedder=embedder, store=store),
            CrossEncoderReranker(DEFAULT_RERANK_MODEL),
        )
    return HybridRetriever([
        DenseRetriever(embedder=embedder, store=store),
        BM25Retriever(store.get_all_chunks()),
    ])


def build_all_retrievers() -> dict[str, Retriever]:
    """All retrievers, sharing one embedder/store/chunks/BM25 build."""
    dense, sparse = build_dense_and_bm25()
    return {
        "dense": dense,
        "bm25": sparse,
        "hybrid": HybridRetriever([dense, sparse]),
        "rerank": RerankingRetriever(dense, CrossEncoderReranker(DEFAULT_RERANK_MODEL)),
    }


def build_pipeline() -> RagPipeline:
    embedder, store = _embedder_and_store()
    return RagPipeline(
        chunker=FixedSizeChunker(max_words=256, overlap=32),
        embedder=embedder,
        store=store,
        retriever=DenseRetriever(embedder=embedder, store=store),
        generator=Generator(llm=build_llm(), min_score=0.2),
    )
