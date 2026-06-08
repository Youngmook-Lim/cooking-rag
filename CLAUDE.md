# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

A **from-scratch** (no LangChain/LlamaIndex in the core) Python RAG pipeline over the
**Cooking / Seasoned Advice** Stack Exchange corpus. Three goals, in priority order:
1. Learn RAG internals by building each component.
2. A portfolio piece — whose differentiator is a rigorous **evaluation harness**.
3. A "beats Google on long-tail cooking questions" demo.

Note: the author does not cook — personal daily utility is explicitly *not* a goal. The
cooking corpus was chosen for the vivid demo contrast and because Stack Exchange's
accepted answers give *free, self-labeling ground truth* for evaluation.

## Corpus

Cooking Stack Exchange data dump (`cooking.stackexchange.com.7z`) from
`archive.org/details/stackexchange`. Extract `Posts.xml` into `data/` (gitignored).
Only **answers** are indexed (the searchable haystack); **questions** are used as
queries/probes. An accepted answer is the ground-truth target for its question.

## Status

**Phases 0–1 (baseline RAG + CLI) are DONE and merged to `main`.**

Working end-to-end: parse `Posts.xml` → chunk → embed → store → dense retrieve →
generate a grounded, cited answer, driven by a `rag` CLI. Verified with real
embedder + real Chroma + retrieval; only the live Claude call is unverified (needs an
API key + the downloaded corpus).

## Architecture

The `rag/` package — small modules, each one responsibility, swappable behind an
interface so the eval harness and retrieval upgrades can A/B implementations:

- `models.py` — `Document`, `Chunk`, `Retrieved`, `Answer` dataclasses.
- `ingest.py` — `parse_posts(path)`: `Posts.xml` → `Document`s (HTML stripped, tags
  parsed, accepted-answer flagged).
- `chunk.py` — `Chunker` ABC + `FixedSizeChunker` (word-window).
- `embed.py` — `Embedder` ABC + `SentenceTransformerEmbedder` (`bge-small-en-v1.5`,
  default) + `FakeEmbedder` (deterministic, offline, for tests).
- `store.py` — `VectorStore` ABC + `ChromaStore` (cosine; uuid ids; metadata
  round-trips post_id/question_id).
- `retrieve.py` — `Retriever` ABC + `DenseRetriever`.
- `generate.py` — `build_prompt`, `Generator` (citations + refusal below `min_score`),
  `LLM` ABC + `ClaudeLLM` (`claude-opus-4-8`; reused as eval judge later).
- `pipeline.py` — `RagPipeline.index(docs)` / `.query(question, k)`. **Enforces the
  key rule:** only answers (`question_id != post_id`) are indexed.
- `factory.py` — `build_pipeline()` composition root wiring the real components.
- `cli.py` — `rag index <Posts.xml>` and `rag ask "<question>" [-k N]`.

## Toolchain & commands

Uses **`uv`**. Tests are **pytest**; the one test that loads the real embedding model
is gated behind `-m slow` so the default suite is fast and offline.

```bash
uv sync --extra dev                 # install
uv run pytest -m "not slow" -q      # fast suite (offline; uses FakeEmbedder/FakeLLM)
uv run pytest -m slow               # real sentence-transformers smoke test
export ANTHROPIC_API_KEY=sk-ant-... # needed for `rag ask` (live Claude)
uv run rag index data/Posts.xml     # build the index
uv run rag ask "why did my mayonnaise break?"
```

## Conventions

- Surgical changes only; match existing style; TDD (failing test → implement → pass →
  commit), frequent small commits.
- Commit messages: no AI/Claude attribution.
- `docs/` is **gitignored** — specs and plans live there but are NOT committed.

## What's next

- **Phase 2 — the evaluation harness** (the centerpiece): build the held-out test set
  from accepted answers; retrieval metrics (Hit@k, MRR) + LLM-judged answer quality
  (faithfulness, correctness); the config-comparison report. Setup 1 (index
  everything, target = a question's own accepted answer) then Setup 2 (hold out a
  question's own answer, force synthesis). Needs its own spec → plan.
- **Phase 3 — retrieval upgrades** (BM25, hybrid/RRF, reranking, better chunking),
  each measured against the baseline.
- Two follow-ups flagged in baseline review: (1) carry `is_accepted`/`score` onto
  `Chunk` so retrieval can rank by answer quality; (2) add a slow end-to-end test
  proving the `0.2` refusal threshold separates on- vs off-corpus questions with the
  real embedder.

Full design and plan (local only, gitignored):
`docs/superpowers/specs/2026-06-04-cooking-rag-pipeline-design.md`,
`docs/superpowers/plans/2026-06-04-cooking-rag-baseline.md`.
