from click.testing import CliRunner
from rag.models import Document
from rag.chunk import FixedSizeChunker
from rag.embed import FakeEmbedder
from rag.store import ChromaStore
from rag.retrieve import DenseRetriever
from rag.generate import Generator
from rag.pipeline import RagPipeline
import rag.cli as cli_mod


class FakeLLM:
    def complete(self, prompt: str) -> str:
        return "Oil added too fast [5570]."


def _fake_pipeline():
    emb = FakeEmbedder(dim=16)
    store = ChromaStore(collection="cli", persist_dir=None)
    pipe = RagPipeline(
        chunker=FixedSizeChunker(max_words=50, overlap=0),
        embedder=emb, store=store,
        retriever=DenseRetriever(embedder=emb, store=store),
        generator=Generator(llm=FakeLLM(), min_score=0.0),
    )
    pipe.index([
        Document(post_id=5570, question_id=5567, title="",
                 body="emulsions break when oil is added too fast",
                 is_accepted=True),
    ])
    return pipe


def test_ask_prints_answer_and_citations(monkeypatch):
    monkeypatch.setattr(cli_mod, "build_pipeline", lambda: _fake_pipeline())
    result = CliRunner().invoke(cli_mod.cli, ["ask", "why did my mayo break?"])
    assert result.exit_code == 0
    assert "5570" in result.output
    assert "oil" in result.output.lower()
