import pytest
from rag.models import Chunk, Retrieved
from rag.rerank import FakeReranker


def _ret(text, post_id):
    return Retrieved(chunk=Chunk(text=text, post_id=post_id, question_id=1), score=0.0)


def test_fake_reranker_orders_by_word_overlap():
    candidates = [
        _ret("best pan for searing steak", 1),                     # overlap 0
        _ret("how to store mayonnaise properly", 2),               # overlap 1
        _ret("mayonnaise emulsions break when oil added fast", 3), # overlap 2
    ]
    out = FakeReranker().rerank("why did my mayonnaise break", candidates, k=3)
    assert [r.chunk.post_id for r in out] == [3, 2, 1]


def test_fake_reranker_respects_k():
    candidates = [
        _ret("mayonnaise break", 3),
        _ret("mayonnaise", 2),
        _ret("steak", 1),
    ]
    out = FakeReranker().rerank("mayonnaise break", candidates, k=2)
    assert [r.chunk.post_id for r in out] == [3, 2]


def test_fake_reranker_empty_input():
    assert FakeReranker().rerank("q", [], k=5) == []


@pytest.mark.slow
def test_cross_encoder_reranker_promotes_relevant():
    from rag.rerank import CrossEncoderReranker
    candidates = [
        _ret("The best pan for searing a steak is cast iron.", 1),
        _ret("Mayonnaise breaks when the oil is added too quickly for the yolk.", 2),
    ]
    out = CrossEncoderReranker().rerank("why did my mayonnaise break?", candidates, k=2)
    assert out[0].chunk.post_id == 2
