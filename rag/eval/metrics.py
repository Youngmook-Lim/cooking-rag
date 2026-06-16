from rag.models import Retrieved
from rag.eval.models import EvalReport


def distinct_answers(retrieved: list[Retrieved]) -> list[int]:
    """post_ids in rank order, first occurrence wins (best rank kept)."""
    seen: set[int] = set()
    ordered: list[int] = []
    for r in retrieved:
        pid = r.chunk.post_id
        if pid not in seen:
            seen.add(pid)
            ordered.append(pid)
    return ordered


def rank_of(answer_id: int, ordered_ids: list[int]) -> int | None:
    for i, pid in enumerate(ordered_ids, start=1):
        if pid == answer_id:
            return i
    return None


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def aggregate(ranks: list[int | None], ks: tuple[int, ...]) -> EvalReport:
    n = len(ranks)
    hit_at = {
        k: (sum(1 for r in ranks if r is not None and r <= k) / n if n else 0.0)
        for k in ks
    }
    mrr = sum(reciprocal_rank(r) for r in ranks) / n if n else 0.0
    return EvalReport(n_cases=n, hit_at=hit_at, mrr=mrr)
