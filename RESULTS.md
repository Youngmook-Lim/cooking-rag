# Results

Retrieval evaluation leaderboard. Each row is one configuration scored by
`rag eval` / `rag eval-compare`. Higher is better for every column.

**Methodology (Setup 1):** `n=500` held-out questions (seed 42), query = question
title, `fetch_depth=50`, answer-level dedupe (distinct answers by `post_id`). Metrics
are deterministic for a fixed index + test set. See the README for what Hit@k and MRR
mean.

| Date | Embedder | Chunker | Retriever | Hit@1 | Hit@5 | Hit@10 | MRR | Notes |
|------|----------|---------|-----------|-------|-------|--------|-----|-------|
| 2026-06-16 | bge-small-en-v1.5 | fixed 256/32 | dense | 0.23 | 0.48 | 0.55 | 0.34 | baseline |
| 2026-06-16 | bge-small-en-v1.5 | fixed 256/32 | bm25 | 0.15 | 0.29 | 0.37 | 0.22 | hand-built BM25 |
| 2026-06-16 | bge-small-en-v1.5 | fixed 256/32 | hybrid (RRF k=60) | 0.21 | 0.45 | 0.54 | 0.31 | unweighted RRF; below dense |
| 2026-07-02 | bge-small-en-v1.5 | fixed 256/32 | **dense** | 0.25 | 0.51 | 0.59 | 0.36 | + bge query prefix (best) |
| 2026-07-03 | bge-small-en-v1.5 | fixed 256/32 | dense + rerank (bge-reranker-base) | 0.19 | 0.42 | 0.52 | 0.29 | ↓ on this metric — see note below |

## Cross-encoder reranker (2026-07-03)

Reranked dense's top-50 with `bge-reranker-base` (`dense + rerank` row above). It scores
**lower on accepted-ID Hit@k**, but investigation shows this is the **metric, not the
reranker**: it correctly promotes genuinely-relevant chunks that simply aren't the one
*accepted* post. Concrete case — Q "Do some fish smell fishier? Why?": the reranker put
*"the fishy smell arises from the production of…"* (score 0.99) and *"fish only smell
fishy when no longer fresh"* (0.93) **above** the accepted answer about "super smellers"
(0.88) — arguably better answers to the literal question, but scored as misses. The
reranker also helps sharply when the accepted answer *is* the best match (e.g. yogurt
rank 12→1, sourdough 46→20).

**Conclusion:** reranking can't be fairly judged by accepted-ID Hit@k (it optimizes
topical relevance, not "is this the accepted post"). Not wired into `ask`; **the
LLM-judge answer-quality metric is the prerequisite to evaluate it properly.** The
`Reranker` / `RerankingRetriever` / `rag eval --retriever rerank` machinery remains.

## Weighted-RRF sweep (2026-07-03)

Swept the hybrid's BM25 fusion weight (dense fixed at 1.0), all with the bge query
prefix, via `rag eval-sweep`:

| bm25_weight | Hit@1 | Hit@5 | Hit@10 | MRR |
|-------------|-------|-------|--------|-----|
| 0.00 (dense-only)        | 0.25 | 0.51 | 0.59 | 0.36 |
| 0.25                     | 0.23 | 0.48 | 0.59 | 0.34 |
| 0.50                     | 0.22 | 0.46 | 0.56 | 0.33 |
| 1.00 (unweighted hybrid) | 0.22 | 0.44 | 0.53 | 0.32 |

**Negative result:** every nonzero BM25 weight hurts, monotonically — dense-only (0.0)
wins. BM25 adds no value on this corpus (short title queries vs answer vocabulary), so
weighted RRF is **explored and rejected**; dense + bge prefix remains the best retriever.
The weighted `HybridRetriever` and the `eval-sweep` tool remain for future use.

## Answer-quality metric methodology

- Generation and judging use the same model by default; set `OPENROUTER_JUDGE_MODEL` to
  judge with an independent model and avoid self-preference bias. Each run's artifact
  records both `model` and `judge_model`.
- Faithfulness is scored only against the chunks the answer was actually built from
  (`score >= min_score`), not the full retrieved set.

## Answer quality (LLM-judged) — dense vs rerank (2026-07-08)

First run of the answer-quality metric: 50 held-out questions (seed 42, same testset),
generation `anthropic/claude-sonnet-4` @ temp 0, judge `openai/gpt-4o` (independent
family — avoids self-preference bias). Scores 1–5, higher is better.

| Retriever | n_judged | Correctness | Faithfulness | Refusals |
|-----------|----------|-------------|--------------|----------|
| dense (bge prefix)                 | 50 | 4.00 | 4.48 | 0 |
| dense + rerank (bge-reranker-base) | 48 | 3.46 | 4.25 | 2 |

Paired on the 48 questions **both** answered: dense correctness 4.06 vs rerank 3.46
(−0.60), faithfulness 4.48 vs 4.25 (−0.23); per question, rerank is better on 7, worse
on 17, tied on 24.

**Result: the reranker loses on answer quality too.** This metric was built on the
hypothesis that accepted-ID Hit@k was *unfairly* harsh on the reranker (it promotes
topically-relevant non-accepted chunks — see the cross-encoder note above). An
independent judge does **not** vindicate it: dense produces better answers, not just
better accepted-ID hits. Dense + bge prefix remains the best config on both metric
families.

**Caveat (confounds the rerank number — follow-up):** the generator's refusal gate
`min_score = 0.2` and the faithfulness "used-chunks" filter are applied to the
retriever's raw scores — cosine similarities for dense (0–1) but cross-encoder logits
for rerank (a different scale). This produces the 2 rerank-only refusals and means the
reranked generator is gated by a cosine-tuned threshold on logit-scale scores. The
paired gap (−0.60) is large enough that fixing this is unlikely to reverse the verdict,
but treat the rerank scores as approximate until the threshold is made retriever-aware
(normalize reranker scores, or scale/skip the gate for rerank).

## Backlog (planned experiments)

Each should add a row above when run. See the project notes for cost/effort.

- **Retriever-aware refusal/faithfulness threshold** — `min_score = 0.2` is cosine-tuned
  but is applied to cross-encoder logits for rerank (see caveat above); normalize
  reranker scores or scale/skip the gate so answer-quality comparisons are apples-to-apples
- OpenAI embeddings (`text-embedding-3-small`/`-large`; rebuild index)
- chunking sweep (`max_words`/`overlap`; sentence/semantic chunking)
- query preprocessing (expansion / HyDE / decomposition)

## Notes

- "Embedder/Chunker/Retriever" together define a config; change one at a time so a
  metric move is attributable.
- For configs that change the index (embedder, chunker), rebuild `.chroma/` before
  scoring.
- Hit@k/MRR are a deterministic **lower bound** on real quality — they credit only the
  question's own accepted answer, so a correct answer from a *different* post scores as
  a miss. An LLM-judged answer-correctness metric (planned) measures what this undercounts.
