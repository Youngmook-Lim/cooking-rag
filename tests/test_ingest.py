from pathlib import Path
from rag.ingest import parse_posts

FIXTURE = Path(__file__).parent / "fixtures" / "sample_posts.xml"


def test_parses_questions_and_answers():
    docs = parse_posts(FIXTURE)
    assert len(docs) == 3


def test_html_is_stripped_to_text():
    docs = {d.post_id: d for d in parse_posts(FIXTURE)}
    assert docs[5570].body == "Emulsions break when oil is added too fast."


def test_question_carries_tags_and_title():
    docs = {d.post_id: d for d in parse_posts(FIXTURE)}
    q = docs[5567]
    assert q.title == "Why did my mayonnaise break?"
    assert q.tags == ["emulsion", "sauce"]
    assert q.question_id == 5567


def test_accepted_answer_is_flagged():
    docs = {d.post_id: d for d in parse_posts(FIXTURE)}
    assert docs[5570].is_accepted is True
    assert docs[5571].is_accepted is False
    assert docs[5570].question_id == 5567
