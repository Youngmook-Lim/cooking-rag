from rag.models import Document
from rag.chunk import FixedSizeChunker


def _answer(body: str, post_id: int = 5570) -> Document:
    return Document(post_id=post_id, question_id=5567, title="", body=body,
                    is_accepted=True)


def test_short_answer_becomes_one_chunk():
    chunker = FixedSizeChunker(max_words=10, overlap=0)
    chunks = chunker.split(_answer("emulsions break when oil is added too fast"))
    assert len(chunks) == 1
    assert chunks[0].post_id == 5570
    assert chunks[0].question_id == 5567


def test_long_answer_splits_into_windows():
    body = " ".join(str(i) for i in range(25))   # 25 words
    chunker = FixedSizeChunker(max_words=10, overlap=0)
    chunks = chunker.split(_answer(body))
    assert len(chunks) == 3                       # 10 + 10 + 5
    assert chunks[0].text.split()[0] == "0"
    assert chunks[1].text.split()[0] == "10"


def test_overlap_repeats_tail_words():
    body = " ".join(str(i) for i in range(20))
    chunker = FixedSizeChunker(max_words=10, overlap=2)
    chunks = chunker.split(_answer(body))
    # second window starts 2 words before the 10-word boundary
    assert chunks[1].text.split()[0] == "8"
