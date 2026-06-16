from dataclasses import dataclass


@dataclass
class TestCase:
    question_id: int
    query_text: str          # the question title used as the search query
    accepted_post_id: int    # ground-truth answer's post_id


@dataclass
class EvalReport:
    n_cases: int
    hit_at: dict[int, float]   # {1: .., 5: .., 10: ..}
    mrr: float
