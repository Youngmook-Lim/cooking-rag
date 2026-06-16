from rag.models import Document
from rag.eval.testset import build_testset, save_testset, load_testset


def _q(qid, title):
    return Document(post_id=qid, question_id=qid, title=title, body="?")


def _a(post_id, question_id, body="answer body", is_accepted=False):
    return Document(post_id=post_id, question_id=question_id, title="",
                    body=body, is_accepted=is_accepted)


def test_eligibility_requires_accepted_nonempty_answer():
    docs = [
        _q(1, "q one"), _a(11, 1, is_accepted=True),               # eligible
        _q(2, "q two"), _a(21, 2, is_accepted=False),             # no accepted
        _q(3, "q three"), _a(31, 3, body="  ", is_accepted=True),  # empty body
        _q(4, "   "), _a(41, 4, is_accepted=True),                 # blank title
    ]
    cases = build_testset(docs, n=10, seed=0)
    assert [c.question_id for c in cases] == [1]
    assert cases[0].query_text == "q one"
    assert cases[0].accepted_post_id == 11


def test_sample_is_deterministic_with_seed():
    docs = []
    for qid in range(1, 51):
        docs += [_q(qid, f"q{qid}"), _a(qid * 100, qid, is_accepted=True)]
    a = build_testset(docs, n=10, seed=42)
    b = build_testset(docs, n=10, seed=42)
    assert [c.question_id for c in a] == [c.question_id for c in b]
    assert len(a) == 10


def test_save_and_load_round_trips(tmp_path):
    docs = [_q(1, "q one"), _a(11, 1, is_accepted=True)]
    cases = build_testset(docs, n=10, seed=0)
    path = tmp_path / "eval" / "testset.json"
    save_testset(cases, str(path))
    assert load_testset(str(path)) == cases
