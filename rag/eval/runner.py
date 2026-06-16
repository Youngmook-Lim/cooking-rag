from rag.retrieve import Retriever
from rag.eval.models import TestCase, EvalReport
from rag.eval.metrics import distinct_answers, rank_of, aggregate


def run_eval(cases: list[TestCase], retriever: Retriever,
             ks: tuple[int, ...] = (1, 5, 10),
             fetch_depth: int = 50) -> EvalReport:
    if fetch_depth <= max(ks):
        raise ValueError(f"fetch_depth ({fetch_depth}) must exceed max(ks) ({max(ks)})")
    ranks: list[int | None] = []
    for case in cases:
        retrieved = retriever.retrieve(case.query_text, k=fetch_depth)
        ordered = distinct_answers(retrieved)
        ranks.append(rank_of(case.accepted_post_id, ordered))
    return aggregate(ranks, ks)
