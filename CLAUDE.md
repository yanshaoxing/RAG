# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chinese-language RAG question-answering system over the novel《遥远的救世主》, built on LlamaIndex. Combines hybrid retrieval (vector + BM25), a hierarchical summary tree, and a Kuzu knowledge graph. All code comments, log messages, and prompts are in Chinese — keep that convention.

Dependencies are pinned in `requirements.txt`; install into a venv (`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`). The LLM/embedding/reranker endpoints live on the Lenovo intranet — outside it, queries and index builds will fail at the network step, but all module imports, chunking, and offline logic still work.

## Commands

```bash
# CLI query (run as a module from the repo root so `rag` is importable)
python -m app.cli "你的问题"        # 单次查询
python -m app.cli                  # 交互模式：索引加载一次，循环提问（exit/quit/空行退出）

# Streamlit web UI (run from repo root)
streamlit run app/ui.py

# Build the full knowledge graph (handles sys.path itself; resumable)
python scripts/build_full_graph.py           # auto-resume from cache
python scripts/build_full_graph.py --force   # delete cache, rebuild from scratch
python scripts/build_full_graph.py 31        # resume from chunk 31

# Visualize the graph (reads data/graph_cache/graph_db.db SQLite cache, outputs pyvis HTML)
python scripts/visualize_graph.py            # top-80 entities by degree
python scripts/visualize_graph.py 150        # top-150

# Offline unit tests (no intranet/LLM needed; run before committing pipeline changes)
.venv/bin/python -m pytest tests/ -q
```

### Rebuilding indexes

Indexing is staged; each stage persists to its own folder under `data/` and writes a `_DONE.json` completion marker when finished. Stage-completion detection uses the marker (a directory without one is treated as an interrupted build and gets rebuilt). To rebuild a stage, **delete its folder** and re-run an entry point — later stages that depend on it rebuild too, untouched earlier stages are loaded from disk:

| Stage | Folder | Contents |
|---|---|---|
| 1. Chunking | `data/chunks/` | serialized TextNodes |
| 2. Summary tree | `data/summary_tree/` | summary nodes + meta map |
| 3. BM25 | `data/bm25/` | jieba-tokenized BM25 index |
| 4. Vector | `data/vector/` (+ `vector/faiss/`) | FAISS HNSW index |
| 5. Graph | `data/graph_db/` | Kuzu DB (also `data/graph_cache/` SQLite cache) |

Source corpus lives in `data/raw/` (plain text / docx; chapter-aware splitting). All JSON persistence is atomic (tmp + `os.replace`).

Stage 4 embedding is checkpointed (`rag/indexing/embedding_checkpoint.py`): embeddings are computed in segments persisted to `data/embed_cache/` (independent of `data/vector/`, which gets wiped on stage rebuild), keyed by a corpus/model fingerprint; an interrupted embedding run resumes from the last completed segment, and the cache is deleted once the vector index persists successfully.

## Architecture

### Configuration: `rag/config.py` + prompts: `rag/prompts.py`

`rag/config.py` is parameters only: model endpoints, tuning parameters, feature toggles, query-time concurrency limits (`QUERY_REWRITE_MAX_CONCURRENCY`, `SUBQUERY_MAX_CONCURRENCY` — default 2 because the Davy endpoint 429s above 2 concurrent). Intranet endpoints and the API key can be overridden via env vars (`RAG_EMBED_OLLAMA_BASE_URL`, `RAG_DAVY_BASE_URL`, `RAG_DAVY_API_KEY`, `RAG_RERANK_BASE_URL`, `RAG_GRAPH_VALIDATE_LLM_BASE_URL`, `RAG_DEBUG`); hardcoded values remain the defaults. Every optional feature has an independent toggle: `REWRITE_ENABLED`, `DECOMPOSE_ENABLED`, `RERANK_ENABLED`, `SUMMARY_TREE_ENABLED`, `GRAPH_ENABLED`, `GRAPH_VALIDATE_ENABLED`, `DEBUG` (defaults to False). When changing behavior, look for a config knob before touching code.

**All** LLM prompt templates live in `rag/prompts.py` (rewrite/HyDE/keywords, decomposition, summarization, graph extract/validate/canonicalize/merge, query entity extraction, final QA template) — never put prompt text in config.py or inline in modules. Runtime placeholders use single braces (`{query}`), filled by the caller's `.format()`; literal JSON braces are `{{ }}`-escaped. The novel-context block `NOVEL_CONTEXT` is injected at import time via `str.replace` (not `.format`), so templates need no double-escaping. `tests/test_prompts.py` guards every template's placeholders.

### LLM providers: `rag/llm/factory.py`

Two providers, switchable per role via config: `"ollama"` (local/remote Ollama) and `"davy"` (Lenovo internal OpenAI-compatible cloud endpoint; `DavyLLM` implements llama_index `CustomLLM`, uses the CA cert in `assets/`, strips `<thinking>`/`<think>` blocks from responses — incrementally via `ThinkStreamFilter` in `stream_chat`, which is true token streaming (per-SSE-chunk yield) — and retries 429/5xx/network errors with exponential backoff honoring `Retry-After` — `DAVY_MAX_RETRIES`). Answer synthesis streams end-to-end when `ANSWER_STREAM_ENABLED=True`: `query()` returns a `StreamingResponse` and both entry points render `response_gen` incrementally. Four factory functions create role-specific LLMs: answer, rewrite, summary, and graph-validation (the validator deliberately uses a *different* model than extraction for cross-checking). Embeddings always come from remote Ollama (`qwen3-embedding:8b`, 4096-dim).

### Entry-point assembly: `rag/engine/bootstrap.py`

Both entry points (`app/cli.py`, `app/ui.py`) are thin display shells over `bootstrap.init_settings()` + `bootstrap.build_query_engine()` + `bootstrap.format_source_nodes()`. Never duplicate assembly logic in the entry points — change `bootstrap.py` instead.

### Indexing pipeline: `rag/indexing/staged_indexer.py`

`get_or_build_index()` is the single entry point. It runs/loads the 5 stages above and returns `(vector_index, bm25_retriever, summary_meta_map, graph_index)`. Summary-tree documents are **mixed into** both the BM25 and vector indexes alongside raw chunks. BM25 nodes store jieba-tokenized text with the original text kept in `metadata["original_text"]` (for summary nodes this is the *source* text they cover, set by the summary tree — never overwrite it).

- `rag/ingestion/preprocessor.py` — chapter/section-aware splitting of Chinese text (regex patterns for `一、`, `第X章/回/节/篇`, `　　1` subsection markers, with auto-detection of heading style; text between a chapter title and the first subsection marker is preserved as a `subsection=""` block), then chunking pipeline (`CHUNK_SIZE=1024`).
- `rag/summarization/summary_tree.py` — 4-level tree: L1 per-chunk one-liners → L2 subsection → L3 chapter → L4 whole-book summaries. `summary_meta_map` maps summary node → covered chunk range (closed interval), used at query time for redundancy filtering. Summaries that fall back to truncated source text (LLM failure) carry `metadata["summary_fallback"]=True` and are counted/warned at build time.

### Query pipeline (per query)

`rag/engine/query_engine.py` guarantees both entry points share prompt/synthesizer config.

1. **Decompose** (`rag/retrieval/query_decomposer.py`) — LLM classifies complex queries (prefix-matched 是/否 answer) and splits into sub-queries, each run through the full pipeline and merged.
2. **3-way rewrite** (`rag/retrieval/query_rewriter.py`) — NL rewrite + HyDE (→ vector retrieval) and keyword expansion (→ BM25), all grounded in the novel context block `NOVEL_CONTEXT` in `rag/prompts.py`. Terminology mapping (`assets/terminology.json`) runs even when `REWRITE_ENABLED=False`.
3. **Per-route filtering** (`rag/retrieval/hybrid_retriever.py`) — summary-redundancy filter (drop summary nodes whose covered chunks were already retrieved, `SUMMARY_REDUNDANCY_THRESHOLD`), then gap detection (adjacent score-drop ratio) and per-route min-score floors.
4. **RRF fusion** of the three routes, then **rerank** via bge-reranker-v2-m3 on a vLLM `/v1/rerank` endpoint (`rag/retrieval/reranker.py`). On reranker failure the pipeline falls back to RRF ordering (`rerank()` returns `None`; scores are never zeroed).
5. **Graph augmentation** — `GraphAugmentedQueryEngine` (in `rag/engine/query_engine.py`) appends Kuzu graph triples as an extra context node scored at the current minimum (`rag/graph/graph_retriever.py`: LLM extracts entities from the query → parameterized Kuzu `CONTAINS` match → neighbor traversal).
6. **Answer synthesis** with `QA_TEMPLATE_STR` (compact mode; answers must cite sources or say the material can't answer).

### Graph subsystem: `rag/graph/`

Pipeline per section: `Extractor` (recall-first LLM extraction) → rule filtering (`rules.json`) → `Validator` (LLM cross-check with a different model, can correct triples; handles string indices from the LLM; validates even single relations) → `DescriptionMerger` → `Canonicalizer` (alias → canonical name) → Kuzu. Extraction is per-section (no separate graph chunking). `Schema` auto-promotes unknown entity/relation types after `GRAPH_SCHEMA_GROWTH_THRESHOLD` occurrences; learned types are deliberately **excluded** from the build fingerprint so schema growth doesn't invalidate the resume cache. `cache.py` (SQLite in `data/graph_cache/`) makes construction resumable per-chunk; the main loop has a per-chunk try/except so one bad chunk never aborts the build; entity-only chunks are saved and marked completed. `metrics.py` collects extraction stats.

### Shared utilities: `rag/utils/`

- `json_parse.py` — `parse_json_obj` / `parse_json_list` / `coerce_index_set`: the single implementation of "json_repair then regex" LLM-output parsing, with guaranteed return types. Use these instead of rolling new parsers.
- `files.py` — `atomic_write_json` / `mark_stage_done` / `stage_complete` for stage persistence.

### Logging convention

Pipeline components use standard `logging` (`logger = logging.getLogger(__name__)` — module names under `rag.` are what the capture hooks). Step-numbered Chinese messages (`步骤 3.1 …`) go through `logger.info`. Entry points wrap pipeline execution in `rag.logging_utils.capture_pipeline_logs()` (contextvars-based) and render `cap.drain()` afterwards; this is what makes per-query logs work under Streamlit's cached engine. Do **not** reintroduce list-passing (`log_list`) plumbing. `DEBUG=True` (or `RAG_DEBUG=1`) enables top-k detail output at each step. Note: `ThreadPoolExecutor` workers don't inherit the capture context — their logs go to standard logging only, so keep per-stage summary logs on the main thread.
