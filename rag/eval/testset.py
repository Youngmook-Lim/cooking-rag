import json
import random
from pathlib import Path
from rag.models import Document
from rag.eval.models import TestCase


def build_testset(docs: list[Document], n: int, seed: int) -> list[TestCase]:
    # question_id -> accepted answer post_id (answer, accepted, non-empty body)
    accepted = {
        d.question_id: d.post_id
        for d in docs
        if d.question_id != d.post_id and d.is_accepted and d.body.strip()
    }
    titles = {d.post_id: d.title for d in docs if d.question_id == d.post_id}
    eligible = sorted(qid for qid in accepted if qid in titles)
    chosen = random.Random(seed).sample(eligible, min(n, len(eligible)))
    return [
        TestCase(question_id=qid, query_text=titles[qid],
                 accepted_post_id=accepted[qid])
        for qid in chosen
    ]


def save_testset(cases: list[TestCase], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([vars(c) for c in cases], indent=2))


def load_testset(path: str) -> list[TestCase]:
    data = json.loads(Path(path).read_text())
    return [TestCase(**d) for d in data]
