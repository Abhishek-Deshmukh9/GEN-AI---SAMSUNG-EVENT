# Agentic Code Intelligence — Code Retrieval on CPU

## Overview
This repository contains the Phase 1 prototype for the **Agentic Code Intelligence** project. The solution is engineered for high-accuracy code retrieval over large, evolving codebases under strict minimal GPU resource constraints (100% CPU-optimized execution).

## Key Deliverables Completed

1. **Role 1: Query Understanding (`src/preprocessor.py`)**
   - **Rule-Based Intent Classifier**: Utilizes specialized regex patterns to classify user queries into three primary intents:
     - `definition-lookup`: Identifying function, class, interface, and struct definitions, signatures, and declarations.
     - `usage-lookup`: Identifying examples, function calls, imports, and invocation patterns.
     - `behavior-lookup`: Identifying processing workflows, runtime behaviors, logic flow, and algorithms.
   - **Identifier Expander**: Converts natural language concepts and code terms into:
     - `camelCase` (e.g., `preProcessInput`)
     - `PascalCase` (e.g., `PreProcessInput`)
     - `snake_case` (e.g., `pre_process_input`)
   - **StructuredQuery Schema**: Strict Pydantic model (`src.preprocessor.StructuredQuery`) providing typed contract for downstream sparse and dense retrieval roles.
   - **Zero Generative LLMs**: Purely deterministic, regex-based, and rule-based processing ensuring sub-millisecond query parsing with zero GPU or LLM generation overhead.

2. **Role 2: Corpus Processing & Chunking (`src/chunker.py`)**
   - **StructuralExtractor**: `ast`-based static analysis of every corpus document, recovering import statements verbatim, top-level function / async-function / class definitions with their signatures and docstrings, and a boilerplate-stripped core logic region.
   - **Boilerplate Stripping**: `if __name__ == "__main__":` guards, `argparse` construction / `add_argument` / `parse_args` / `set_defaults`, interactive `input()` prompts, and the `if` statements that test those scaffold values are removed from the core logic region.
   - **SnippetSchema**: Strict Pydantic model (`src.chunker.SnippetSchema`) carrying `id`, `doc_id`, `summary_view`, `full_view`, and metadata (`chunk_index`, `num_chunks`, `has_imports`, plus name / kind / signature / docstring / `parse_failed` / estimated tokens).
   - **LongDocChunker**: Recursive structural chunking for documents above a configurable token budget (default 256), splitting on function/class definitions first and falling back to blank-line paragraphs, physical lines, and word-level hard splits.
   - **ChunkCache**: SQLite-backed memoization of chunking results so re-running over an unchanged corpus skips re-processing.
   - **aggregate_scores**: Max-pooling of per-chunk similarity scores back to one score per document, ready to be called from `encoder.py` before MTEB ranks documents.
   - **Zero Generative LLMs / Zero GPU**: Pure `ast` static analysis, hashing, and string assembly. Standard library plus Pydantic only — no tokenizer, no network, no GPU, no LLM API calls.

3. **MTEB Retrieval Pipeline (`src/encoder.py`)**
   - **PrePostPipelineEncoder**: Subclasses `mteb.models.abs_encoder.AbsEncoder`.
   - **Model**: `nomic-ai/CodeRankEmbed` (137M, code-specific) on CPU, inputs capped at 512 tokens.
   - **Query instruction**: queries get the model's required prefix (`Represent this query for searching relevant code: `).
   - **Query Preprocessing Integration**: `QueryParser` categorizes every query; its enriched text (`enrich_queries=True`) is off by default because its intent prefix conflicts with the model's query instruction on long APPS problem statements.
   - Directly conforms to MTEB 2.x interface and passes all test assertions.

4. **MTEB AppsRetrieval Evaluation (`evaluate.py`)**
   - Evaluated on the **CoIR `AppsRetrieval`** test split (8,765 corpus documents, 3,765 queries).
   - Generates the official `appsretrieval_results.json` artifact for submission.

---

## Techniques Added (Role 2)

| Technique | Type | Purpose |
| :--- | :--- | :--- |
| AST structural extraction | Static analysis | Recover definition boundaries, signatures, and docstrings instead of guessing from text layout. |
| Boilerplate / scaffolding stripping | Static analysis + AST predicate matching | Drop CLI scaffolding (`__main__` guards, argparse, input prompts) that dilutes the embedded signal. |
| Structural recursive chunking | Hierarchical segmentation | Split on function/class boundaries first, then blank-line paragraphs, then lines, then words. |
| Context header enrichment | Local context injection | Prepend the parent signature + docstring to fragment summaries so mid-function chunks stay self-describing. |
| Dual-view schema (`summary_view` / `full_view`) | Multi-view representation | Embed a compact view now; retain the complete chunk text unembedded for a future reranker. |
| Tokenizer-free length estimation | Heuristic proxy | `whitespace split x 1.3` instead of a tokenizer, keeping the corpus stage dependency-free. |
| Content-addressed identity | Normalized hashing | `sha256(ast.dump(..., include_attributes=False))` — invariant to whitespace, line numbers, and comments. |
| Exact memoization keys | Whitespace-normalized hashing | Cache lookups keyed on a cheap, collision-free content hash instead of a second AST pass. |
| Max-pool score aggregation | Late-interaction reduction | Collapse multi-chunk documents to a single score by their best-matching chunk. |
| Graceful degradation | Robustness | Unparseable, empty, or binary-ish documents still produce exactly one snippet; cache/SQLite failures fall back to recomputation. |

### Design Decisions and Trade-offs

- **Core logic excludes definitions.** `core_logic` holds module-level statements only; definitions live in `units`, so no text is duplicated between the two views.
- **Chunks respect the budget.** The context header's token cost is deducted from the body budget, so every parseable chunk lands at or below `max_tokens`.
- **Unparseable documents stay intact.** The raw text is preserved in `full_view` and flagged with `parse_failed`, so no corpus document is ever dropped; `summary_view` is truncated to keep the embedded string bounded.
- **Cache key rationale.** Keying on the AST hash cost more than chunking itself, so `ChunkCache` keys on a whitespace-normalized hash: identical keys always imply identical output, hits skip parsing entirely, and comment-only edits are simply re-chunked. `SnippetSchema.id` remains the AST hash.
- **Not yet wired.** Role 2 is not connected to `encoder.py` / `evaluate.py` yet, so the baseline scores below are unchanged.

---

## Results (CoIR AppsRetrieval, test split)

_`appsretrieval_results.json` is still the MiniLM baseline (also in `results/baseline_minilm_results.json`). Re-run `python scripts/prepare.py`, `python scripts/tune_dev.py`, then `python evaluate.py` to produce the hybrid CodeRankEmbed + BM25 result._

### Baseline: all-MiniLM-L6-v2

| Metric | Score |
| :--- | :--- |
| **NDCG@10** | **0.05780** |
| **MRR@10** | **0.04829** |
| **MAP@10** | **0.04829** |
| **Recall@10** | **0.08871** |

---

## Repository Structure

```text
├── .gitignore
├── README.md
├── requirements.txt
├── evaluate.py
├── appsretrieval_results.json
├── src/
│   ├── __init__.py
│   ├── preprocessor.py   # Role 1: QueryParser, IdentifierExpander, StructuredQuery
│   ├── chunker.py        # Role 2: StructuralExtractor, SnippetSchema, LongDocChunker, ChunkCache
│   └── encoder.py        # MTEB PrePostPipelineEncoder (AbsEncoder)
├── tests/
│   ├── test_preprocessor.py
│   ├── test_chunker.py
│   └── test_encoder.py
└── data/
```

> Note: `src/__init__.py` imports the encoder (and therefore `mteb`, `torch`, and
> `sentence-transformers`) defensively, so the CPU-only analysis roles stay
> importable without the retrieval stack installed.

---

## Quickstart

### 1. Environment Setup
```bash
# Using uv (recommended) or standard venv
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

### 2. Run Unit Tests
```bash
python tests/test_preprocessor.py
python tests/test_chunker.py
python tests/test_encoder.py
```

### 3. Run MTEB Evaluation
```bash
python evaluate.py
```
This produces `appsretrieval_results.json` adhering to the MTEB evaluation format
(the same file is attached to the GitHub release).

### 4. Run the Retrieval Demo
```bash
python demo.py --limit 500                                  # quick interactive demo
python demo.py --query "count pairs whose sum is divisible by k" -k 5
python demo.py                                              # full 8.7k-snippet corpus
```
The demo prints the query intent, the top-k code snippets, and retrieval latency.

**Versioning (P1):** document embeddings are cached under `.cache/` keyed by a SHA-256
content hash. Re-indexing a new corpus version re-embeds only snippets whose content
changed; the demo reports how many were embedded vs. reused.

---

## Corpus Stage Performance Notes

Measured on CPU with synthetic 175 KB Python documents (40 chunks each):

| Operation | Throughput |
| :--- | :--- |
| Structural extraction + chunking | ~210k estimated tokens/s |
| Chunk cache hit (no re-parse) | ~9x cheaper than chunking |
| AST dump removal (`ast.get_source_segment` → line slicing) | 27.2s → 4.2s per 200 docs |

The chunk cache stores complete `full_view` text, so cache size scales with corpus
size; a reranker that only needs summaries could store `summary_view` instead.
