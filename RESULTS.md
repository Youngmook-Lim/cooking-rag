# Results

Retrieval evaluation leaderboard. Each row is one configuration scored by
`rag eval` / `rag eval-compare`. Higher is better for every column.

**Methodology (Setup 1):** `n=500` held-out questions (seed 42), query = question
title, `fetch_depth=50`, answer-level dedupe (distinct answers by `post_id`). Metrics
are deterministic for a fixed index + test set. See the README for what Hit@k and MRR
mean.

| Date | Embedder | Chunker | Retriever | Hit@1 | Hit@5 | Hit@10 | MRR | Notes |
|------|----------|---------|-----------|-------|-------|--------|-----|-------|
| 2026-06-16 | bge-small-en-v1.5 | fixed 256/32 | **dense** | 0.23 | 0.48 | 0.55 | 0.34 | baseline |
| 2026-06-16 | bge-small-en-v1.5 | fixed 256/32 | bm25 | 0.15 | 0.29 | 0.37 | 0.22 | hand-built BM25 |
| 2026-06-16 | bge-small-en-v1.5 | fixed 256/32 | hybrid (RRF k=60) | 0.21 | 0.45 | 0.54 | 0.31 | unweighted RRF; below dense |

## Backlog (planned experiments)

Each should add a row above when run. See the project notes for cost/effort.

- bge query-instruction prefix (query-side only; no re-index)
- weighted RRF (down-weight BM25)
- OpenAI embeddings (`text-embedding-3-small`/`-large`; rebuild index)
- chunking sweep (`max_words`/`overlap`; sentence/semantic chunking)
- cross-encoder reranker over top-k
- query preprocessing (expansion / HyDE / decomposition)

## Notes

- "Embedder/Chunker/Retriever" together define a config; change one at a time so a
  metric move is attributable.
- For configs that change the index (embedder, chunker), rebuild `.chroma/` before
  scoring.
- Hit@k/MRR are a deterministic **lower bound** on real quality — they credit only the
  question's own accepted answer, so a correct answer from a *different* post scores as
  a miss. An LLM-judged answer-correctness metric (planned) measures what this undercounts.
