# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chinese-language multi-corpus RAG question-answering system over novels (default corpus:《遥远的救世主》), built on LlamaIndex. Combines hybrid retrieval (vector + BM25), a hierarchical summary tree, and a Kuzu knowledge graph. All code comments, log messages, and prompts are in Chinese — keep that convention.

**Multi-corpus layout**: each book lives in `corpora/<slug>/` — `corpus.json` (required: `title`, `context`; optional `author`/`description`/`chapter_pattern`/`subsection_pattern` — the two pattern fields are chapter/subsection title regexes, hand-written or auto-written-back by LLM structure detection), `raw/` (source text), `terminology.json` (optional alias→canonical map), `graph_rules.json` (optional graph-rule additions, merged onto `rag/graph/rules.json`: lists unioned, scalars overridden), and `data/` (all 5 index stages + `graph_cache/` + `embed_cache/`). Book-specific content (title, character bios) belongs **only** in corpus assets — never hardcode it in code or prompt templates. `WuLingChaShi` is a tiny built demo corpus used to exercise multi-book switching.

**Active corpus & multi-engine**: `rag/corpus.py` holds the process-wide active corpus (startup default from `RAG_CORPUS` env / `config.DEFAULT_CORPUS`; switch with `set_active_corpus()`, enumerate with `list_corpora()`). All corpus-dependent paths in `rag/config.py` (`CHUNKS_DIR`, `TERM_MAP_PATH`, …, listed in `_CORPUS_RELATIVE_PATHS`) are **dynamic module attributes** (PEP 562) computed from the active corpus at access time — code reading `config.X` follows a switch automatically. The concurrency contract: query-time corpus state is **bound at engine construction** (indexes, graph store, `QueryRewriter` fixes its 3 prompts + term map in `__init__`), so engines for different books coexist in one process; build-time components (summary/graph prompts, graph rules) read the active corpus, so `bootstrap.build_query_engine(corpus_slug)` switches + builds under a global `_BUILD_LOCK`. Entry points: `python -m app.cli --corpus <slug>` / `--list`; the Streamlit UI has a sidebar book selector with per-book engine cache and chat history.

Dependencies are pinned in `requirements.txt`; install into a venv (`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`). Default endpoints are **public Aliyun** (`provider="aliyun"`): chat, embedding and rerank all live on one cn-beijing MaaS workspace sharing a single key (the three `RAG_PUBLIC_*_API_KEY` vars are kept so roles can be split again later). The models and prices currently in use are documented in `LLM.txt` at the repo root — **when you change a model, update `LLM.txt` too**; `scripts/test_public_llm.py` reads endpoints/models straight from `rag/config.py` so it can never drift. API keys are read from `.env` at the repo root (gitignored — never hardcode them; `rag/config.py` loads `.env` at import). The old Lenovo-intranet configs (davy/ollama/vllm) are kept and switchable per role. **Do not point chat at `dashscope-us`** — that endpoint runs content inspection and 400-rejects the novel corpus (`data_inspection_failed`); the workspace deployment does not. **`ALIYUN_ENABLE_THINKING` defaults to `False`**: Qwen3-series chat models reason by default and reasoning tokens can be 99% of the billed output (measured: one summary call = 5019 output tokens / 45.7s with thinking, 24 tokens / 0.7s without, no visible quality difference) — reasoning never reaches `message.content`, so the `<think>` stripper cannot see it but you still pay for it. Aliyun embedding is 1024-dim vs intranet 4096-dim: `EMBED_VECTOR_DIM` switches with `EMBED_PROVIDER`. **Same dimension does not mean reusable** — a different embedding model means a different vector space, so `_stage_vector` records `embed_model` in the stage marker and `_load_vector` raises on mismatch; deleting `data/vector/` is the fix.

## Commands

```bash
# CLI query (run as a module from the repo root so `rag` is importable)
python -m app.cli "你的问题"        # 单次查询（默认语料）
python -m app.cli -c WuLingChaShi "你的问题"   # 指定语料（书）
python -m app.cli --list           # 列出全部可用语料
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

Indexing is staged; each stage persists to its own folder under the active corpus's `corpora/<slug>/data/` and writes a `_DONE.json` completion marker when finished. Stage-completion detection uses the marker (a directory without one is treated as an interrupted build and gets rebuilt). To rebuild a stage, **delete its folder** and re-run an entry point — later stages that depend on it rebuild too, untouched earlier stages are loaded from disk:

| Stage | Folder (under `corpora/<slug>/data/`) | Contents |
|---|---|---|
| 1. Chunking | `chunks/` | serialized TextNodes |
| 2. Summary tree | `summary_tree/` | summary nodes + meta map |
| 3. BM25 | `bm25/` | jieba-tokenized BM25 index |
| 4. Vector | `vector/` (+ `vector/faiss/`) | FAISS HNSW index |
| 5. Graph | `graph_db/` | Kuzu DB (also `graph_cache/` SQLite cache) |

Source corpus lives in `corpora/<slug>/raw/` (plain text / docx; chapter-aware splitting). All JSON persistence is atomic (tmp + `os.replace`).

Stage 4 embedding is checkpointed (`rag/indexing/embedding_checkpoint.py`): embeddings are computed in segments persisted to the corpus's `data/embed_cache/` (independent of `data/vector/`, which gets wiped on stage rebuild), keyed by a corpus/model fingerprint; an interrupted embedding run resumes from the last completed segment, and the cache is deleted once the vector index persists successfully.

## Architecture

### Configuration: `rag/config.py` + prompts: `rag/prompts.py`

`rag/config.py` is parameters only: model endpoints, tuning parameters, feature toggles, query-time concurrency limits (`QUERY_REWRITE_MAX_CONCURRENCY`, `SUBQUERY_MAX_CONCURRENCY` — default 2 because the Davy endpoint 429s above 2 concurrent). Intranet endpoints and the API key can be overridden via env vars (`RAG_EMBED_OLLAMA_BASE_URL`, `RAG_DAVY_BASE_URL`, `RAG_DAVY_API_KEY`, `RAG_RERANK_BASE_URL`, `RAG_GRAPH_VALIDATE_LLM_BASE_URL`, `RAG_DEBUG`); hardcoded values remain the defaults. Every optional feature has an independent toggle: `REWRITE_ENABLED`, `DECOMPOSE_ENABLED`, `RERANK_ENABLED`, `SUMMARY_TREE_ENABLED`, `GRAPH_ENABLED`, `GRAPH_VALIDATE_ENABLED`, `DEBUG` (defaults to False). When changing behavior, look for a config knob before touching code.

**All** LLM prompt templates live in `rag/prompts.py` (rewrite/HyDE/keywords, decomposition, summarization, graph extract/validate/canonicalize/merge, query entity extraction, final QA template) — never put prompt text in config.py or inline in modules. Runtime placeholders use single braces (`{query}`), filled by the caller's `.format()`; literal JSON braces are `{{ }}`-escaped. Corpus-dependent templates are stored as raw `_RAW_*` strings with `{book_title}` / `{corpus_context}` markers (registered in `_RAW_TEMPLATES`) and rendered lazily by the module's PEP 562 `__getattr__`, which injects the active corpus profile's title/context via `str.replace` (not `.format`, so templates need no double-escaping) and caches per (corpus, template). Templates must contain **no** hardcoded book titles or character names — that content lives in `corpora/<slug>/corpus.json`. `tests/test_prompts.py` guards every template's placeholders and the no-hardcoding rule.

### LLM providers: `rag/llm/factory.py`

Three providers, switchable per role via config: `"ollama"` (local/remote Ollama), `"davy"` (Lenovo internal OpenAI-compatible cloud endpoint), and `"aliyun"` (public Aliyun workspace, the current default — reuses the `DavyLLM` client with `cert_path=""` meaning system CA). `DavyLLM` implements llama_index `CustomLLM`, uses the CA cert in `assets/` for Davy (custom-CA verify), strips `<thinking>`/`<think>` blocks from responses — incrementally via `ThinkStreamFilter` in `stream_chat`, which is true token streaming (per-SSE-chunk yield) — and retries 429/5xx/network errors with exponential backoff honoring `Retry-After` — `DAVY_MAX_RETRIES`. Answer synthesis streams end-to-end when `ANSWER_STREAM_ENABLED=True`: `query()` returns a `StreamingResponse` and both entry points render `response_gen` incrementally. Four factory functions create role-specific LLMs: answer, rewrite, summary, and graph-validation (the validator deliberately uses a *different* model than extraction for cross-checking: `qwen-flash` vs `qwen3.5-flash`). Embeddings switch on `EMBED_PROVIDER`: aliyun `qwen3.7-text-embedding` (1024-dim, `OpenAILikeEmbedding`, batch ≤10 per DashScope limit) or remote Ollama `qwen3-embedding:8b` (4096-dim).

### Entry-point assembly: `rag/engine/bootstrap.py`

Both entry points (`app/cli.py`, `app/ui.py`) are thin display shells over `bootstrap.init_settings()` + `bootstrap.build_query_engine()` + `bootstrap.format_source_nodes()`. Never duplicate assembly logic in the entry points — change `bootstrap.py` instead.

### Indexing pipeline: `rag/indexing/staged_indexer.py`

`get_or_build_index()` is the single entry point. It runs/loads the 5 stages above and returns `(vector_index, bm25_retriever, summary_meta_map, graph_index)`. Summary-tree documents are **mixed into** both the BM25 and vector indexes alongside raw chunks. BM25 nodes store jieba-tokenized text with the original text kept in `metadata["original_text"]` (for summary nodes this is the *source* text they cover, set by the summary tree — never overwrite it).

**Metadata exclusion keys**: LlamaIndex prepends all metadata to node content for embed/LLM modes, so bulk/structural keys must be excluded — `SUMMARY_EXCLUDED_META_KEYS` (summary_tree.py: `original_text`, `summary_child_ids`, …), `CHUNK_EXCLUDED_META_KEYS` (preprocessor.py: `section_path`), plus `original_text` on BM25 node copies. JSON serialization does **not** persist exclusion keys, so the `staged_indexer` deserializers re-apply them on load — any new deserialization path must do the same.

- `rag/ingestion/preprocessor.py` — chapter/section-aware splitting of Chinese text (regex patterns for `一、`, `第X章/回/节/篇`, `　　1` subsection markers, with auto-detection of heading style; text between a chapter title and the first subsection marker is preserved as a `subsection=""` block), then chunking pipeline (`CHUNK_SIZE=1024`). Pattern resolution (`_resolve_structure_patterns`): the corpus profile's `chapter_pattern`/`subsection_pattern` fields win; otherwise built-in auto-detection; if the built-in regexes have **zero hits** in the whole text and `STRUCTURE_DETECT_ENABLED`, `rag/ingestion/structure_detector.py` samples the text (head + 25%/50%/75% slices), asks the summary LLM for regexes, **deterministically validates** them (compilable, sane section count/title length per `STRUCTURE_*` config) and persists them back into `corpus.json` — so detection runs at most once per book and stays offline afterwards; any failure falls back silently (whole book becomes one `概述` section).
- `rag/summarization/summary_tree.py` — 4-level tree: L1 per-chunk one-liners → L2 subsection → L3 chapter → L4 whole-book summaries. `summary_meta_map` maps summary node → covered chunk range (closed interval), used at query time for redundancy filtering. Summaries that fall back to truncated source text (LLM failure) carry `metadata["summary_fallback"]=True` and are counted/warned at build time.

### Query pipeline (per query)

`rag/engine/query_engine.py` guarantees both entry points share prompt/synthesizer config.

1. **Decompose** (`rag/retrieval/query_decomposer.py`) — LLM classifies complex queries (prefix-matched 是/否 answer) and splits into sub-queries, each run through the full pipeline; results are merged by **per-sub-query rank RRF** (never by raw score — rerank scores and RRF-fallback scores differ by an order of magnitude).
2. **3-way rewrite** (`rag/retrieval/query_rewriter.py`) — NL rewrite + HyDE (→ vector retrieval) and keyword expansion (→ BM25), all grounded in the active corpus's `context` block (from `corpus.json`, injected by `rag/prompts.py`). Terminology mapping (the corpus's `terminology.json`) runs even when `REWRITE_ENABLED=False`. The BM25 query string is always passed through `tokenize_for_bm25` (`rag/utils/text.py`) before retrieval — the corpus is jieba-tokenized, and an untokenized Chinese query scores zero.
3. **Per-route filtering** (`rag/retrieval/hybrid_retriever.py`) — summary-redundancy filter (drop summary nodes whose covered chunks were already retrieved, `SUMMARY_REDUNDANCY_THRESHOLD`), then gap detection (adjacent score-drop ratio) and per-route min-score floors.
4. **RRF fusion** of the three routes, then **rerank** (`rag/retrieval/reranker.py`; `RERANK_PROVIDER`: aliyun `qwen3-rerank` with Bearer auth, or intranet bge-reranker-v2-m3 on a vLLM `/v1/rerank` endpoint — both return the same `results[].index`+`relevance_score` shape). Transient failures (network/429/5xx) get `RERANK_MAX_RETRIES` quick retries; on final failure the pipeline falls back to RRF ordering (`rerank()` returns `None`; scores are never zeroed).
5. **Graph augmentation** — `GraphAugmentedQueryEngine` (in `rag/engine/query_engine.py`) appends Kuzu graph triples as an extra context node scored at the current minimum (`rag/graph/graph_retriever.py`: LLM extracts entities from the query → parameterized Kuzu `CONTAINS` match → neighbor traversal).
6. **Answer synthesis** with `QA_TEMPLATE_STR` (compact mode; answers must cite sources or say the material can't answer).

### Graph subsystem: `rag/graph/`

Pipeline per section: `Extractor` (recall-first LLM extraction) → rule filtering (base `rules.json` merged with the corpus's `graph_rules.json`) → `Validator` (LLM cross-check with a different model, can correct triples; handles string indices from the LLM; validates even single relations) → `DescriptionMerger` → `Canonicalizer` (alias → canonical name, deterministic fast rules before LLM) → Kuzu. Extraction is per-section (no separate graph chunking). The build runs as a **bounded pipeline** (`_run_pipelined`): worker threads do extract+validate concurrently (`GRAPH_EXTRACT_MAX_CONCURRENCY`; no shared state — `Schema.resolve_type` is the one locked exception), while the main thread does all bookkeeping (canonicalize/merge/SQLite) serially in submission order — keep cache writes on the main thread. Merge/canonicalize are deduplicated per unique entity per chunk. The build fingerprint uses the **actual model names** (`_resolve_graph_models`); learned types are deliberately **excluded** from it so schema growth doesn't invalidate the resume cache. `cache.py` (SQLite in `data/graph_cache/`) makes construction resumable per-chunk; per-chunk error handling means one bad chunk never aborts the build; entity-only chunks are saved and marked completed. `metrics.py` collects extraction stats.

### Shared utilities: `rag/utils/`

- `json_parse.py` — `parse_json_obj` / `parse_json_list` / `coerce_index_set`: the single implementation of "json_repair then regex" LLM-output parsing, with guaranteed return types. Use these instead of rolling new parsers.
- `files.py` — `atomic_write_json` / `mark_stage_done` / `stage_complete` for stage persistence.
- `text.py` — `tokenize_for_bm25`: the single jieba tokenizer shared by BM25 index build and query time (both sides must tokenize identically).

### Logging convention

Pipeline components use standard `logging` (`logger = logging.getLogger(__name__)` — module names under `rag.` are what the capture hooks). Step-numbered Chinese messages (`步骤 3.1 …`) go through `logger.info`. Entry points wrap pipeline execution in `rag.logging_utils.capture_pipeline_logs()` (contextvars-based) and render `cap.drain()` afterwards; this is what makes per-query logs work under Streamlit's cached engine. Do **not** reintroduce list-passing (`log_list`) plumbing. `DEBUG=True` (or `RAG_DEBUG=1`) enables top-k detail output at each step. Note: `ThreadPoolExecutor` workers don't inherit the capture context — their logs go to standard logging only, so keep per-stage summary logs on the main thread.
