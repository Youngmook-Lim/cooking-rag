from click.testing import CliRunner
from rag.models import Document, Chunk, Retrieved
from rag.eval.models import TestCase
import rag.cli as cli_mod


def test_eval_build_writes_testset(tmp_path, monkeypatch):
    docs = [
        Document(post_id=1, question_id=1, title="q one", body="?"),
        Document(post_id=11, question_id=1, title="", body="ans", is_accepted=True),
    ]
    monkeypatch.setattr(cli_mod, "parse_posts", lambda p: docs)
    posts = tmp_path / "Posts.xml"
    posts.write_text("<x/>")
    out = tmp_path / "testset.json"
    result = CliRunner().invoke(
        cli_mod.cli,
        ["eval-build", str(posts), "--out", str(out), "--n", "5"],
    )
    assert result.exit_code == 0
    assert out.exists()
    assert "1 test case" in result.output


def test_eval_prints_table(monkeypatch):
    cases = [TestCase(question_id=1, query_text="q", accepted_post_id=11)]
    monkeypatch.setattr(cli_mod, "load_testset", lambda p: cases)

    class FakeRetriever:
        def retrieve(self, query, k):
            return [Retrieved(
                chunk=Chunk(text="a", post_id=11, question_id=1), score=0.9)]

    monkeypatch.setattr(cli_mod, "build_retriever", lambda: FakeRetriever())
    result = CliRunner().invoke(cli_mod.cli, ["eval", "--testset", "x.json"])
    assert result.exit_code == 0
    assert "Hit@1" in result.output
    assert "1.00" in result.output
