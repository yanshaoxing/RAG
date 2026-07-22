# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chinese-language RAG question-answering system over the novel《遥远的救世主》, built on LlamaIndex. Combines hybrid retrieval (vector + BM25), a hierarchical summary tree, and a Kuzu knowledge graph. All code comments, log messages, and prompts are in Chinese — keep that convention.

There is no requirements.txt, no test suite, and no git repo. Key dependencies: `llama-index` (+ `llama-index-retrievers-bm25`, `llama-index-vector-stores-faiss`, `llama-index-graph-stores-kuzu`, `llama-index-llms-ollama`, `llama-index-embeddings-ollama`), `faiss`, `kuzu`, `jieba`, `streamlit`, `docx2txt`, `json_repair`, `networkx`, `pyvis`. Note: the system Python on this machine does not have these installed.

## Commands

```bash
# CLI query (question is hardcoded in the run_query() call at the bottom of app/cli.py — edit it there)
# Must be run as a module from the repo root so that `indexer` and `rag` are importable:
python -m app.cli

# Streamlit web UI (run from repo root)
streamlit run app/ui.py

# Build the full knowledge graph (handles sys.path itself; resumable)
python scripts/build_full_graph.py           # auto-resume from cache
python scripts/build_full_graph.py --force   # delete cache, rebuild from scratch
python scripts/build_full_graph.py 31        # resume from chunk 31

# Visualize the graph (reads data/graph_triples_cache_graph_db.json, outputs pyvis HTML)
python scripts/visualize_graph.py
```

### Rebuilding indexes

Indexing is staged; each stage persists to its own folder under `data/`. To rebuild a stage, **delete its folder** and re-run an entry point — later stages that depend on it rebuild too, untouched earlier stages are loaded from disk:

| Stage | Folder | Contents |
|---|---|---|
| 1. Chunking | `data/chunks/` | serialized TextNodes |
| 2. Summary tree | `data/summary_tree/` | summary nodes + meta map |
| 3. BM25 | `data/bm25/` | jieba-tokenized BM25 index |
| 4. Vector | `data/vector/` (+ `vector/faiss/`) | FAISS HNSW index |
| 5. Graph | `data/graph_db/` | Kuzu DB (also `data/graph_cache/` SQLite cache + `data/graph_triples_cache_graph_db.json`) |

Source corpus lives in `data/raw/` (plain text / docx; chapter-aware splitting).

## Architecture

### Configuration: `rag/config.py`

Single source of truth for **everything**: model endpoints, all tuning parameters, and all LLM prompts (query rewrite, HyDE, keyword expansion, decomposition, summarization, graph extraction/validation, final QA template). Every optional feature has an independent toggle: `REWRITE_ENABLED`, `DECOMPOSE_ENABLED`, `RERANK_ENABLED`, `SUMMARY_TREE_ENABLED`, `GRAPH_ENABLED`, `GRAPH_VALIDATE_ENABLED`, `DEBUG`. When changing behavior, look for a config knob before touching code.

### LLM providers: `rag/llm/factory.py`

Two providers, switchable per role via config: `"ollama"` (local/remote Ollama) and `"davy"` (Lenovo internal OpenAI-compatible cloud endpoint; `DavyLLM` implements llama_index `CustomLLM`, uses the CA cert in `assets/`, strips `<thinking>` blocks from responses). Four factory functions create role-specific LLMs: answer, rewrite, summary, and graph-validation (the validator deliberately uses a *different* model than extraction for cross-checking). Embeddings always come from remote Ollama (`qwen3-embedding:8b`, 4096-dim).

### Indexing pipeline: `rag/indexing/staged_indexer.py`

`get_or_build_index()` is the single entry point (root-level `indexer.py` is a thin compat wrapper around it). It runs/loads the 5 stages above and returns `(vector_index, bm25_retriever, summary_meta_map, graph_index)`. Summary-tree documents are **mixed into** both the BM25 and vector indexes alongside raw chunks. BM25 nodes store jieba-tokenized text with the original text kept in `metadata["original_text"]`.

- `rag/ingestion/preprocessor.py` — chapter/section-aware splitting of Chinese text (regex patterns for `一、`, `第X章/回/节/篇`, `（一）` headings, with auto-detection of heading style), then chunking pipeline (`CHUNK_SIZE=1024`).
- `rag/summarization/summary_tree.py` — 4-level tree: L1 per-chunk one-liners → L2 subsection → L3 chapter → L4 whole-book summaries. `summary_meta_map` maps summary node → covered chunk range, used at query time for redundancy filtering.

### Query pipeline (per query)

Both entry points (`app/cli.py`, `app/ui.py`) assemble an identical pipeline; `rag/engine/query_engine.py` guarantees they share prompt/synthesizer config.

1. **Decompose** (`rag/retrieval/query_decomposer.py`) — LLM classifies complex queries and splits into sub-queries, each run through the full pipeline and merged.
2. **3-way rewrite** (`rag/retrieval/query_rewriter.py`) — NL rewrite + HyDE (→ vector retrieval) and keyword expansion (→ BM25), all grounded in the novel context block `_NOVEL_CONTEXT` in config.
3. **Per-route filtering** (`rag/retrieval/hybrid_retriever.py`) — summary-redundancy filter (drop summary nodes whose covered chunks were already retrieved, `SUMMARY_REDUNDANCY_THRESHOLD`), then gap detection (adjacent score-drop ratio) and per-route min-score floors.
4. **RRF fusion** of the three routes, then **rerank** via bge-reranker-v2-m3 on a vLLM `/v1/rerank` endpoint (`rag/retrieval/reranker.py`).
5. **Graph augmentation** — `GraphAugmentedQueryEngine` (in `rag/engine/query_engine.py`) appends Kuzu graph triples as an extra context node (`rag/graph/graph_retriever.py`: LLM extracts entities from the query → fuzzy-match Kuzu nodes → depth-2 traversal).
6. **Answer synthesis** with `QA_TEMPLATE_STR` (compact mode; answers must cite sources or say the material can't answer).

### Graph subsystem: `rag/graph/`

Pipeline per chunk: `Extractor` (recall-first LLM extraction) → rule filtering (`rules.json`) → `Validator` (LLM cross-check with a different model, can correct triples) → `DescriptionMerger` → `Canonicalizer` (alias → canonical name) → Kuzu. `Schema` auto-promotes unknown entity/relation types after `GRAPH_SCHEMA_GROWTH_THRESHOLD` occurrences. `cache.py` (SQLite in `data/graph_cache/`) makes construction resumable per-chunk; `metrics.py` collects extraction stats. Graph chunks are larger than retrieval chunks (`GRAPH_CHUNK_SIZE=1500`).

### Logging convention

Pipeline components don't print directly — they append step-numbered Chinese messages (`步骤 3.1 …`) to a shared `log_list` passed in at construction; entry points flush and print it after each phase. Follow this pattern when adding pipeline steps. `DEBUG=True` in config enables top-k detail output at each step.
