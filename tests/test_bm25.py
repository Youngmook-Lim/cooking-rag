from rag.bm25 import tokenize, BM25Index


def test_tokenize_lowercases_and_splits():
    assert tokenize("How To Fix, a Broken-Emulsion!") == \
        ["how", "to", "fix", "a", "broken", "emulsion"]


def _worked_example():
    return BM25Index([
        "mayonnaise emulsion broke",      # doc 0
        "how to fix a broken emulsion",   # doc 1
        "best pan for searing steak",     # doc 2
    ])


def test_ranks_doc_matching_more_query_terms_first():
    ranked = _worked_example().search("broken emulsion", k=3)
    ids = [doc_id for doc_id, _ in ranked]
    assert ids[0] == 1          # matches both "broken" and "emulsion"
    assert ids[1] == 0          # matches "emulsion" only
    assert 2 not in ids         # matches nothing -> not returned


def test_rare_term_outweighs_common_term():
    idx = BM25Index([
        "emulsion emulsion emulsion xanthan",  # doc 0: has the rare term
        "emulsion emulsion emulsion",          # doc 1: common term only
        "emulsion cooking",                    # doc 2
        "emulsion baking",                     # doc 3
    ])
    ranked = idx.search("xanthan emulsion", k=4)
    assert ranked[0][0] == 0    # rare "xanthan" lifts doc 0 to the top


def test_unknown_term_and_empty_query_return_no_results():
    idx = _worked_example()
    assert idx.search("zzz nonexistent", k=3) == []
    assert idx.search("", k=3) == []
