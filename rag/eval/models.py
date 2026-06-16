from dataclasses import dataclass


@dataclass
class TestCase:
    """One retrieval probe: a question and its ground-truth accepted answer."""
    __test__ = False   # not a pytest test class despite the name
    question_id: int
    query_text: str          # the question title used as the search query
    accepted_post_id: int    # ground-truth answer's post_id


@dataclass
class EvalReport:
    """Aggregated retrieval metrics over a test set."""
    n_cases: int
    hit_at: dict[int, float]   # {1: .., 5: .., 10: ..}
    mrr: float
