# cooking-rag

A **from-scratch** Retrieval-Augmented Generation (RAG) pipeline over the
[Cooking / Seasoned Advice Stack Exchange](https://cooking.stackexchange.com) corpus —
built without LangChain or LlamaIndex in the core, so every component (chunking,
embedding, vector store, retrieval, generation) is hand-written and swappable.

Ask a long-tail cooking question and get a grounded, **cited** answer synthesized from
real Stack Exchange answers:

```
$ rag ask "why did my mayonnaise break?"
Emulsions break when the oil is added too quickly for the egg yolk to absorb it [5570].
To rescue it, whisk a fresh yolk in a clean bowl and slowly drizzle the broken
mixture back in [5570][8210].

Sources:
  https://cooking.stackexchange.com/a/5570
  https://cooking.stackexchange.com/a/8210
```

## Why this project exists

Three goals, in priority order:

1. **Learn RAG internals** by building each component from scratch.
2. **A portfolio piece** whose differentiator is a rigorous **evaluation harness** — not
   just "it works," but *measured* retrieval quality with a reproducible metrics table.
3. A **"beats Google on long-tail cooking questions"** demo.

The cooking corpus was chosen for the vivid demo contrast and because Stack Exchange's
**accepted answers give free, self-labeling ground truth**: for any question, the answer
the asker marked as correct is the retrieval target — no manual labeling needed. That's
what makes the eval harness possible.

> Only **answers** are indexed (the searchable haystack); **questions** are used as
> queries and as eval probes.

## How it works

```
INGEST:  Posts.xml → parse → keep answers only → chunk → embed ─┐
                                                                ▼
                                                          [ ChromaDB ]
                                                                ▲
SEARCH:  question → embed ──── retrieve top-k ──────────────────┘
                                     │
                                     ▼
                       refusal gate → prompt → LLM → cited answer
```

The `rag/` package, one responsibility per module:

| Module | Responsibility |
|---|---|
| `models.py` | `Document`, `Chunk`, `Retrieved`, `Answer` dataclasses |
| `ingest.py` | Parse `Posts.xml` → `Document`s (HTML stripped, accepted-answer flagged) |
| `chunk.py` | `FixedSizeChunker` — sliding word-window with overlap |
| `embed.py` | `SentenceTransformerEmbedder` (`bge-small-en-v1.5`) + offline `FakeEmbedder` |
| `store.py` | `ChromaStore` — cosine vector store (batched, paginated) |
| `retrieve.py` | `DenseRetriever`, `BM25Retriever`, `HybridRetriever` (RRF) |
| `bm25.py` | Hand-built BM25 keyword-search engine (inverted index) |
| `generate.py` | Prompt building, citation extraction, refusal, `OpenRouterLLM` |
| `pipeline.py` | `RagPipeline.index()` / `.query()` — enforces "answers only" |
| `eval/` | The evaluation harness (test-set builder, metrics, runner) |
| `factory.py` | Composition root wiring the real components |
| `cli.py` | The `rag` command-line interface |

Everything sits behind small interfaces (ABCs) so the eval harness can A/B different
implementations — that's how dense vs. BM25 vs. hybrid get compared.

## Requirements

- **Python ≥ 3.11**
- [**uv**](https://docs.astral.sh/uv/) (package/environment manager)
- An [**OpenRouter**](https://openrouter.ai) API key (for answer generation; the model is
  configurable and defaults to `anthropic/claude-sonnet-4`)
- ~**7z** to extract the corpus dump (`brew install p7zip` on macOS)

## Setup

### 1. Install dependencies

```bash
uv sync --extra dev
```

This creates a virtualenv and installs everything (including PyTorch via
`sentence-transformers`, so the first run is a sizable download). It also registers the
`rag` command.

### 2. Configure API access

```bash
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY=sk-or-...
# OPENROUTER_MODEL is optional (defaults to anthropic/claude-sonnet-4)
```

Config is read from `.env` automatically. Any OpenRouter-hosted model works — change
`OPENROUTER_MODEL` to pick a different one.

### 3. Download the corpus

Grab the Cooking Stack Exchange data dump and extract `Posts.xml` into `data/`:

```bash
mkdir -p data
curl -L -o cooking.7z https://archive.org/download/stackexchange/cooking.stackexchange.com.7z
7z e cooking.7z Posts.xml -odata/
```

(`data/` is gitignored.)

### 4. Build the index

```bash
uv run rag index data/Posts.xml
```

This parses the posts, keeps answers only, chunks them, embeds each chunk with
`bge-small-en-v1.5` (downloads on first run), and persists a Chroma vector store to
`.chroma/`. Expect a few minutes; it prints something like:

```
Indexed 73181 chunks from 64239 answers (91564 posts parsed).
```

## Usage

### Ask a question

```bash
uv run rag ask "how do I keep guacamole from browning?"
uv run rag ask "why did my bread not rise?" -k 8     # retrieve more passages
```

The answer is generated **only** from retrieved sources and cites the post IDs it used.
If nothing in the corpus is relevant enough, it refuses rather than hallucinating.

### Evaluate retrieval quality (the centerpiece)

The eval harness measures whether retrieval surfaces each question's *accepted answer*,
using **Hit@k** (was the right answer in the top-k?) and **MRR** (how high up?). It needs
no API key — it never calls the LLM.

```bash
# 1. Build a held-out test set (sampled questions + their accepted-answer targets)
uv run rag eval-build data/Posts.xml --n 500

# 2. Score a retriever
uv run rag eval --retriever dense      # or: bm25, hybrid
```

```
Setup 1 — retrieval (n=500, retriever=dense, fetch=50)
Hit@1   Hit@5   Hit@10   MRR
0.23    0.48    0.55     0.34
```

### Compare all retrievers at once

```bash
uv run rag eval-compare
```

```
Setup 1 — retrieval comparison (n=500, fetch=50)
retriever   Hit@1   Hit@5   Hit@10   MRR
dense       0.23    0.48    0.55     0.34
bm25        0.15    0.29    0.37     0.22
hybrid      0.21    0.45    0.54     0.31
```

**On this corpus, dense retrieval wins.** BM25 (keyword matching) underperforms because
the queries are short question *titles* while the answers rarely repeat that vocabulary,
and the naive equal-weight hybrid gets dragged down by the weaker BM25 signal. This is
exactly the kind of finding the harness exists to surface — it caught that the "hybrid
upgrade" was actually a regression. (Weighted RRF is a planned next experiment.)

## Development

```bash
uv run python -m pytest -m "not slow" -q   # fast suite (offline; fakes, no network)
uv run python -m pytest -m slow            # real sentence-transformers smoke test
```

> Use `uv run python -m pytest`, not `uv run pytest` — the latter can resolve to the
> wrong interpreter.

The project follows TDD with small, surgical commits. Tests use a deterministic
`FakeEmbedder` and an ephemeral Chroma store, so the default suite is fast and fully
offline.

## Command reference

| Command | Description |
|---|---|
| `rag index <Posts.xml>` | Parse the corpus and build the vector index |
| `rag ask "<question>" [-k N]` | Get a grounded, cited answer (`-k` = passages to retrieve) |
| `rag eval-build <Posts.xml> [--n N] [--seed S] [--out PATH]` | Build the held-out test set |
| `rag eval [--retriever dense\|bm25\|hybrid] [--fetch N]` | Score one retriever |
| `rag eval-compare [--fetch N]` | Score all three retrievers in one table |

## Configuration

Read from `.env`:

- `OPENROUTER_API_KEY` — required for `rag ask` (and to build the pipeline for `rag index`).
- `OPENROUTER_MODEL` — optional; defaults to `anthropic/claude-sonnet-4`.

## Roadmap

- **Done:** baseline RAG + CLI; Setup-1 retrieval eval harness (Hit@k / MRR); BM25 +
  hybrid (RRF) retrieval with a comparison report.
- **Next:** weighted RRF; LLM-judged answer quality (faithfulness, correctness); Setup 2
  (hold out each question's own answer to force synthesis); reranking and better chunking.
