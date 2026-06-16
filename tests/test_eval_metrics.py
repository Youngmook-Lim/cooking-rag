import pytest
from rag.models import Chunk, Retrieved
from rag.eval.metrics import distinct_answers, rank_of, reciprocal_rank, aggregate


def _r(post_id, score):
    return Retrieved(chunk=Chunk(text="x", post_id=post_id, question_id=1),
                     score=score)


def test_distinct_answers_dedupes_keeping_best_rank():
    retrieved = [_r(8001, .9), _r(5570, .8), _r(5570, .7), _r(9100, .6), _r(5570, .5)]
    assert distinct_answers(retrieved) == [8001, 5570, 9100]


def test_rank_of_found_and_missing():
    assert rank_of(5570, [8001, 5570, 9100]) == 2
    assert rank_of(1234, [8001, 5570, 9100]) is None


def test_reciprocal_rank():
    assert reciprocal_rank(2) == 0.5
    assert reciprocal_rank(None) == 0.0


def test_aggregate_hits_and_mrr():
    ranks = [1, 3, None, 2]   # 4 cases
    report = aggregate(ranks, ks=(1, 5, 10))
    assert report.n_cases == 4
    assert report.hit_at[1] == 0.25
    assert report.hit_at[5] == 0.75
    assert report.mrr == pytest.approx((1.0 + 1/3 + 0.0 + 0.5) / 4)
