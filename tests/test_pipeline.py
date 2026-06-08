from rag.models import Document
from rag.chunk import FixedSizeChunker
from rag.embed import FakeEmbedder
from rag.store import ChromaStore
from rag.retrieve import DenseRetriever
from rag.generate import Generator, Answer
from rag.pipeline import RagPipeline


class FakeLLM:
    def complete(self, prompt: str) -> str:
        return "Oil was added too fast [5570]."


def _pipeline():
    emb = FakeEmbedder(dim=16)
    store = ChromaStore(collection="pipe", persist_dir=None)
    return RagPipeline(
        chunker=FixedSizeChunker(max_words=50, overlap=0),
        embedder=emb,
        store=store,
        retriever=DenseRetriever(embedder=emb, store=store),
        generator=Generator(llm=FakeLLM(), min_score=0.0),
    )


def _docs():
    return [
        Document(post_id=5567, question_id=5567, title="Why did my mayo break?",
                 body="my mayo separated why", is_accepted=False),   # question
        Document(post_id=5570, question_id=5567, title="",
                 body="emulsions break when oil is added too fast", is_accepted=True),
        Document(post_id=9001, question_id=9000, title="",
                 body="bake sourdough at high hydration", is_accepted=True),
    ]


def test_index_only_stores_answers():
    pipe = _pipeline()
    pipe.index(_docs())
    # 2 answers indexed, the question is skipped
    assert pipe.store.count() == 2


def test_query_returns_grounded_answer():
    pipe = _pipeline()
    pipe.index(_docs())
    ans = pipe.query("why did my mayonnaise break?", k=2)
    assert isinstance(ans, Answer)
    assert 5570 in ans.cited_post_ids
