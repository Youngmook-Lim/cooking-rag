import json
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
    assert "Wrote 1 test case to" in result.output


def test_eval_prints_table(tmp_path, monkeypatch):
    cases = [TestCase(question_id=1, query_text="q", accepted_post_id=11)]
    monkeypatch.setattr(cli_mod, "load_testset", lambda p: cases)

    class FakeRetriever:
        def retrieve(self, query, k):
            return [Retrieved(
                chunk=Chunk(text="a", post_id=11, question_id=1), score=0.9)]

    monkeypatch.setattr(cli_mod, "build_retriever", lambda kind="dense": FakeRetriever())
    testset = tmp_path / "testset.json"
    testset.write_text("[]")
    result = CliRunner().invoke(cli_mod.cli, ["eval", "--testset", str(testset)])
    assert result.exit_code == 0
    assert "Hit@1" in result.output
    assert "1.00" in result.output


def test_eval_retriever_option_dispatches(tmp_path, monkeypatch):
    cases = [TestCase(question_id=1, query_text="q", accepted_post_id=11)]
    monkeypatch.setattr(cli_mod, "load_testset", lambda p: cases)
    seen = {}

    class FakeRetriever:
        def retrieve(self, query, k):
            return [Retrieved(
                chunk=Chunk(text="a", post_id=11, question_id=1), score=0.9)]

    def fake_build(kind):
        seen["kind"] = kind
        return FakeRetriever()

    monkeypatch.setattr(cli_mod, "build_retriever", fake_build)
    testset = tmp_path / "t.json"
    testset.write_text("[]")
    result = CliRunner().invoke(
        cli_mod.cli, ["eval", "--testset", str(testset), "--retriever", "bm25"])
    assert result.exit_code == 0
    assert seen["kind"] == "bm25"
    assert "retriever=bm25" in result.output


def test_eval_compare_prints_row_per_retriever(tmp_path, monkeypatch):
    cases = [TestCase(question_id=1, query_text="q", accepted_post_id=11)]
    monkeypatch.setattr(cli_mod, "load_testset", lambda p: cases)

    class FakeRetriever:
        def __init__(self, pid):
            self.pid = pid
        def retrieve(self, query, k):
            return [Retrieved(
                chunk=Chunk(text="a", post_id=self.pid, question_id=1), score=0.9)]

    monkeypatch.setattr(cli_mod, "build_all_retrievers", lambda: {
        "dense": FakeRetriever(11), "bm25": FakeRetriever(99),
        "hybrid": FakeRetriever(11)})
    testset = tmp_path / "t.json"
    testset.write_text("[]")
    result = CliRunner().invoke(cli_mod.cli, ["eval-compare", "--testset", str(testset)])
    assert result.exit_code == 0
    for name in ("dense", "bm25", "hybrid"):
        assert name in result.output


def test_eval_sweep_prints_row_per_weight(tmp_path, monkeypatch):
    cases = [TestCase(question_id=1, query_text="q", accepted_post_id=11)]
    monkeypatch.setattr(cli_mod, "load_testset", lambda p: cases)

    class FakeRetriever:
        def retrieve(self, query, k):
            return [Retrieved(
                chunk=Chunk(text="a", post_id=11, question_id=1), score=0.9)]

    monkeypatch.setattr(cli_mod, "build_dense_and_bm25",
                        lambda: (FakeRetriever(), FakeRetriever()))
    testset = tmp_path / "t.json"
    testset.write_text("[]")
    result = CliRunner().invoke(
        cli_mod.cli,
        ["eval-sweep", "--testset", str(testset), "--weights", "0,0.5,1.0"])
    assert result.exit_code == 0
    assert "bm25_weight" in result.output
    for w in ("0.0", "0.5", "1.0"):
        assert w in result.output


def test_eval_accepts_rerank_retriever(tmp_path, monkeypatch):
    cases = [TestCase(question_id=1, query_text="q", accepted_post_id=11)]
    monkeypatch.setattr(cli_mod, "load_testset", lambda p: cases)
    seen = {}

    class FakeRetriever:
        def retrieve(self, query, k):
            return [Retrieved(
                chunk=Chunk(text="a", post_id=11, question_id=1), score=0.9)]

    def fake_build(kind):
        seen["kind"] = kind
        return FakeRetriever()

    monkeypatch.setattr(cli_mod, "build_retriever", fake_build)
    testset = tmp_path / "t.json"
    testset.write_text("[]")
    result = CliRunner().invoke(
        cli_mod.cli, ["eval", "--testset", str(testset), "--retriever", "rerank"])
    assert result.exit_code == 0
    assert seen["kind"] == "rerank"


def test_eval_quality_writes_artifact_and_prints_summary(tmp_path, monkeypatch):
    cases = [TestCase(question_id=1, query_text="why did mayo break?",
                      accepted_post_id=11)]
    monkeypatch.setattr(cli_mod, "load_testset", lambda p: cases)
    docs = [Document(post_id=11, question_id=1, title="", body="oil added too fast")]
    monkeypatch.setattr(cli_mod, "parse_posts", lambda p: docs)

    class FakeRetriever:
        def retrieve(self, query, k):
            return [Retrieved(chunk=Chunk(text="oil added too fast", post_id=11,
                                          question_id=1), score=0.9)]

    monkeypatch.setattr(cli_mod, "build_retriever", lambda kind="dense": FakeRetriever())

    class FakeLLM:
        def complete(self, prompt):
            return '{"correctness": 4, "faithfulness": 5, "rationale": "ok"}'

    monkeypatch.setattr(cli_mod, "build_llm", lambda *a, **k: FakeLLM())

    testset = tmp_path / "t.json"; testset.write_text("[]")
    posts = tmp_path / "Posts.xml"; posts.write_text("<x/>")
    out = tmp_path / "q.json"
    result = CliRunner().invoke(cli_mod.cli, [
        "eval-quality", "--testset", str(testset), "--posts", str(posts),
        "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["summary"]["mean_correctness"] == 4.0
    assert data["cases"][0]["faithfulness"] == 5
    assert "correctness" in result.output
    assert "retriever=dense" in result.output


def test_eval_quality_skips_missing_reference(tmp_path, monkeypatch):
    cases = [TestCase(question_id=7, query_text="q", accepted_post_id=999)]
    monkeypatch.setattr(cli_mod, "load_testset", lambda p: cases)
    docs = [Document(post_id=11, question_id=1, title="", body="unrelated")]
    monkeypatch.setattr(cli_mod, "parse_posts", lambda p: docs)

    class FakeRetriever:
        def retrieve(self, query, k):
            return [Retrieved(chunk=Chunk(text="x", post_id=11, question_id=1),
                              score=0.9)]

    monkeypatch.setattr(cli_mod, "build_retriever", lambda kind="dense": FakeRetriever())

    class FakeLLM:
        def complete(self, prompt):
            return '{"correctness": 3, "faithfulness": 3, "rationale": "x"}'

    monkeypatch.setattr(cli_mod, "build_llm", lambda *a, **k: FakeLLM())

    testset = tmp_path / "t.json"; testset.write_text("[]")
    posts = tmp_path / "Posts.xml"; posts.write_text("<x/>")
    out = tmp_path / "q.json"
    result = CliRunner().invoke(cli_mod.cli, [
        "eval-quality", "--testset", str(testset), "--posts", str(posts),
        "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "999" in result.output or "not found" in result.output.lower()
    data = json.loads(out.read_text())
    assert data["summary"]["n_judged"] == 0
