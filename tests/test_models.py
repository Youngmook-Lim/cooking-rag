from rag.models import Document, Chunk, Retrieved


def test_document_holds_thread_fields():
    doc = Document(
        post_id=5570,
        question_id=5567,
        title="Why did my mayonnaise break?",
        body="Emulsions break when oil is added too fast...",
        tags=["emulsion", "sauce"],
        score=42,
        is_accepted=True,
    )
    assert doc.post_id == 5570
    assert doc.is_accepted is True
    assert "emulsion" in doc.tags


def test_chunk_carries_source_metadata():
    chunk = Chunk(text="Emulsions break...", post_id=5570, question_id=5567)
    assert chunk.post_id == 5570


def test_retrieved_pairs_chunk_with_score():
    chunk = Chunk(text="x", post_id=1, question_id=1)
    r = Retrieved(chunk=chunk, score=0.87)
    assert r.score == 0.87
    assert r.chunk.post_id == 1
