from rag.models import Chunk, Retrieved
from rag.eval.quality import (
    QualityResult, QualityReport, JudgeParseError,
    build_judge_prompt, render_sources,
)


def _r(text, post_id, score=0.9):
    return Retrieved(chunk=Chunk(text=text, post_id=post_id, question_id=1), score=score)


def test_build_judge_prompt_includes_all_inputs_and_alternatives_note():
    prompt = build_judge_prompt(
        question="why did my mayo break?",
        answer="oil added too fast",
        sources="[5570] add oil slowly",
        reference="mayo breaks when oil is added too fast",
    )
    assert "why did my mayo break?" in prompt
    assert "oil added too fast" in prompt
    assert "[5570] add oil slowly" in prompt
    assert "mayo breaks when oil is added too fast" in prompt
    assert "other correct answers are also valid" in prompt.lower()
    assert "correctness" in prompt and "faithfulness" in prompt


def test_render_sources_uses_post_id_bracket_format():
    out = render_sources([_r("add oil slowly", 5570), _r("whisk hard", 42)])
    assert "[5570] add oil slowly" in out
    assert "[42] whisk hard" in out


def test_quality_result_and_report_construct():
    r = QualityResult(question_id=1, query="q", answer="a", cited_post_ids=[5570],
                      correctness=4, faithfulness=5, rationale="ok")
    assert r.refused is False and r.judge_error is False
    rep = QualityReport(n_cases=1, n_judged=1, mean_correctness=4.0,
                        mean_faithfulness=5.0, refusal_rate=0.0, n_judge_errors=0)
    assert rep.mean_correctness == 4.0


def test_judge_parse_error_is_value_error():
    assert issubclass(JudgeParseError, ValueError)


import pytest
from rag.eval.quality import Judge, FakeJudge, LLMJudge, parse_verdict


class _CannedLLM:
    def __init__(self, reply):
        self.reply = reply
        self.last_prompt = None

    def complete(self, prompt):
        self.last_prompt = prompt
        return self.reply


def test_parse_verdict_reads_scores_and_rationale():
    c, f, r = parse_verdict('{"correctness": 4, "faithfulness": 5, "rationale": "good"}')
    assert (c, f, r) == (4, 5, "good")


def test_parse_verdict_tolerates_surrounding_text():
    c, f, r = parse_verdict('Sure!\n{"correctness": 3, "faithfulness": 2, "rationale": "x"}\ndone')
    assert (c, f) == (3, 2)


def test_parse_verdict_rejects_non_json():
    with pytest.raises(JudgeParseError):
        parse_verdict("no json here")


def test_parse_verdict_rejects_out_of_range():
    with pytest.raises(JudgeParseError):
        parse_verdict('{"correctness": 9, "faithfulness": 5, "rationale": "x"}')


def test_llm_judge_builds_prompt_and_parses():
    llm = _CannedLLM('{"correctness": 5, "faithfulness": 4, "rationale": "ok"}')
    judge = LLMJudge(llm)
    c, f, r = judge.score("q", "a", "[1] src", "ref")
    assert (c, f, r) == (5, 4, "ok")
    assert "q" in llm.last_prompt and "ref" in llm.last_prompt


def test_llm_judge_raises_on_malformed():
    judge = LLMJudge(_CannedLLM("garbage"))
    with pytest.raises(JudgeParseError):
        judge.score("q", "a", "s", "ref")


def test_fake_judge_is_deterministic_and_in_range():
    judge = FakeJudge()
    c1, f1, _ = judge.score("q", "oil added too fast", "[1] add oil slowly", "oil too fast")
    c2, f2, _ = judge.score("q", "oil added too fast", "[1] add oil slowly", "oil too fast")
    assert (c1, f1) == (c2, f2)
    assert 1 <= c1 <= 5 and 1 <= f1 <= 5
    assert isinstance(judge, Judge)


from rag.eval.models import TestCase
from rag.eval.quality import run_quality_eval, aggregate_quality


class _FakeRetriever:
    def __init__(self, post_id=11):
        self.post_id = post_id

    def retrieve(self, query, k):
        return [Retrieved(
            chunk=Chunk(text="oil added too fast", post_id=self.post_id,
                        question_id=1), score=0.9)]


class _StubAnswer:
    def __init__(self, text, cited_post_ids, refused):
        self.text = text
        self.cited_post_ids = cited_post_ids
        self.refused = refused


class _FakeGenerator:
    def __init__(self, refused=False):
        self.refused = refused

    def answer(self, question, retrieved):
        if self.refused:
            return _StubAnswer("I couldn't find anything.", [], True)
        return _StubAnswer("oil added too fast [11]", [11], False)


class _ScriptedJudge(Judge):
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)

    def score(self, question, answer, sources, reference):
        v = self.verdicts.pop(0)
        if isinstance(v, Exception):
            raise v
        return v


def _case(qid=1, accepted=11):
    return TestCase(question_id=qid, query_text="why did mayo break?",
                    accepted_post_id=accepted)


def test_run_quality_eval_scores_and_aggregates():
    report, results = run_quality_eval(
        [_case()], _FakeRetriever(), _FakeGenerator(),
        _ScriptedJudge([(4, 5, "ok")]), {11: "oil too fast"}, k=5)
    assert report.n_cases == 1 and report.n_judged == 1
    assert report.mean_correctness == 4.0 and report.mean_faithfulness == 5.0
    assert report.refusal_rate == 0.0 and report.n_judge_errors == 0
    assert results[0].correctness == 4 and results[0].cited_post_ids == [11]


def test_run_quality_eval_excludes_refusals_from_means():
    report, results = run_quality_eval(
        [_case(1), _case(2)], _FakeRetriever(), _FakeGenerator(refused=True),
        _ScriptedJudge([]), {11: "ref"}, k=5)
    assert report.n_cases == 2 and report.n_judged == 0
    assert report.refusal_rate == 1.0
    assert report.mean_correctness == 0.0
    assert all(r.refused for r in results)


def test_run_quality_eval_counts_judge_errors():
    report, results = run_quality_eval(
        [_case()], _FakeRetriever(), _FakeGenerator(),
        _ScriptedJudge([JudgeParseError("bad")]), {11: "ref"}, k=5)
    assert report.n_judged == 0 and report.n_judge_errors == 1
    assert results[0].judge_error is True


def test_aggregate_quality_empty_is_zero():
    report = aggregate_quality([])
    assert report.n_cases == 0 and report.n_judged == 0
    assert report.mean_correctness == 0.0 and report.refusal_rate == 0.0
