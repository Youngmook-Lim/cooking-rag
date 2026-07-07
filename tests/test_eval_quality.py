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
