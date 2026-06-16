from rag.models import Chunk
from rag.embed import FakeEmbedder
from rag.store import ChromaStore
from rag.retrieve import DenseRetriever
from rag.eval.models import TestCase
from rag.eval.runner import run_eval


def _seeded_retriever():
    emb = FakeEmbedder(dim=16)
    store = ChromaStore(collection="evalrun", persist_dir=None)
    chunks = [
        Chunk(text="mayonnaise emulsion broke", post_id=5570, question_id=5567),
        Chunk(text="how to bake sourdough bread", post_id=9001, question_id=9000),
    ]
    store.add(chunks, emb.embed([c.text for c in chunks]))
    return DenseRetriever(embedder=emb, store=store)


def test_hit_when_accepted_answer_indexed():
    retriever = _seeded_retriever()
    cases = [TestCase(question_id=5567, query_text="mayonnaise emulsion broke",
                      accepted_post_id=5570)]
    report = run_eval(cases, retriever, ks=(1, 5), fetch_depth=10)
    assert report.hit_at[1] == 1.0
    assert report.mrr == 1.0


def test_miss_when_answer_not_in_corpus():
    retriever = _seeded_retriever()
    cases = [TestCase(question_id=1, query_text="mayonnaise emulsion broke",
                      accepted_post_id=12345)]   # not in store
    report = run_eval(cases, retriever, ks=(1, 5), fetch_depth=10)
    assert report.hit_at[1] == 0.0
    assert report.mrr == 0.0


def test_empty_cases_is_all_zero():
    retriever = _seeded_retriever()
    report = run_eval([], retriever, ks=(1, 5), fetch_depth=10)
    assert report.n_cases == 0
    assert report.hit_at == {1: 0.0, 5: 0.0}
    assert report.mrr == 0.0


def test_multichunk_answer_dedupes_to_single_rank():
    emb = FakeEmbedder(dim=16)
    store = ChromaStore(collection="evalmulti", persist_dir=None)
    chunks = [
        Chunk(text="emulsion broke part one", post_id=5570, question_id=5567),
        Chunk(text="emulsion broke part two", post_id=5570, question_id=5567,
              chunk_index=1),
        Chunk(text="emulsion broke part three", post_id=5570, question_id=5567,
              chunk_index=2),
        Chunk(text="sourdough starter tips", post_id=9001, question_id=9000),
        Chunk(text="sharpening a chef knife", post_id=7200, question_id=7100),
    ]
    store.add(chunks, emb.embed([c.text for c in chunks]))
    retriever = DenseRetriever(embedder=emb, store=store)
    cases = [TestCase(question_id=5567, query_text="emulsion broke part two",
                      accepted_post_id=5570)]
    report = run_eval(cases, retriever, ks=(1, 5), fetch_depth=10)
    assert report.hit_at[1] == 1.0
    assert report.mrr == 1.0
