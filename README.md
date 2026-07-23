# 《遥远的救世主》RAG 问答系统

基于 LlamaIndex 的中文小说知识库检索问答系统：混合检索（向量 + BM25）、层次化摘要树、Kuzu 知识图谱三路证据融合，经 rerank 后流式生成带引用的回答。

## 功能特性

- **三路查询改写**（并行）：自然语言改写 / HyDE → 向量检索，关键词扩展 → BM25 检索；术语映射（通俗名 → 原文术语）在改写关闭时也生效
- **混合检索**：三路召回 → 摘要冗余过滤 → gap 过滤 → RRF 融合 → bge-reranker-v2-m3 重排（失败自动降级 RRF 顺序，瞬时故障先快速重试）
- **查询分解**：复杂问题自动拆分为子查询并行检索，按名次 RRF 融合合并
- **层次化摘要树**：L1 逐块 → L2 小节 → L3 章节 → L4 全书四级摘要，混入主索引解决"宏观问题查不到"；LLM 失败自动降级并显式标记
- **知识图谱**：按节 LLM 抽取实体/关系 → 规则过滤 → 不同模型交叉校验 → 描述合并 → 别名归一化 → Kuzu 持久化；构建为有界流水线并发，SQLite 缓存支持断点续传
- **真流式输出**：SSE 逐 token 渲染，增量剥离 `<think>` 思考块；断流未输出正文时自动整流重试
- **工程化**：分阶段索引（`_DONE.json` 完成标记 + 原子写入）、embedding 分段断点续传、188 例离线单测（无需内网）

## 环境要求

- Python 3.12，依赖版本固定在 `requirements.txt`
- LLM / embedding / reranker 端点位于联想内网（Davy 云端 + 远程 Ollama + vLLM rerank）。**内网之外**：查询与索引构建会在网络步骤失败，但模块导入、分块与全部离线单测可正常运行
- 端点与密钥可用环境变量覆盖（未设置时用内置默认值）：
  `RAG_DAVY_BASE_URL` / `RAG_DAVY_API_KEY` / `RAG_EMBED_OLLAMA_BASE_URL` / `RAG_RERANK_BASE_URL` / `RAG_GRAPH_VALIDATE_LLM_BASE_URL` / `RAG_DEBUG`

## 安装

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 使用

```bash
# CLI（从仓库根目录以模块方式运行）
python -m app.cli "丁元英的私募基金为什么解散"   # 单次查询
python -m app.cli                              # 交互模式：索引加载一次，循环提问

# Streamlit Web UI
streamlit run app/ui.py

# 知识图谱构建（可断点续传）
python scripts/build_full_graph.py            # 自动续传
python scripts/build_full_graph.py --force    # 删缓存全量重建

# 图谱可视化（输出 pyvis 交互式 HTML）
python scripts/visualize_graph.py             # 度数 top-80 实体

# 离线单测（不依赖内网，改动管线代码前必跑）
.venv/bin/python -m pytest tests/ -q
```

首次运行会自动依次构建 5 个索引阶段（分块 → 摘要树 → BM25 → 向量 → 图谱），各阶段独立持久化到 `data/` 下并写入 `_DONE.json` 完成标记；之后启动直接从磁盘加载。

## 索引重建

删除对应阶段目录后重跑任一入口即可——依赖它的后续阶段会一并重建，之前的阶段从磁盘加载：

| 阶段 | 目录 |
|---|---|
| 1 分块 | `data/chunks/` |
| 2 摘要树 | `data/summary_tree/` |
| 3 BM25 | `data/bm25/` |
| 4 向量（FAISS HNSW） | `data/vector/` |
| 5 图谱（Kuzu） | `data/graph_db/`（抽取缓存在 `data/graph_cache/`，不随重建删除） |

向量阶段的 embedding 分段落盘到 `data/embed_cache/`，中断后续跑只补缺失段。

## 目录结构

```
app/            CLI 与 Streamlit 入口（展示层薄壳）
rag/
  config.py     全部调参/端点/开关（纯参数，支持环境变量覆盖）
  prompts.py    全部 LLM prompt 模板
  engine/       bootstrap 装配 + 查询引擎（图谱上下文注入）
  retrieval/    三路改写 / 混合检索 / 查询分解 / rerank
  indexing/     分阶段索引管线 + embedding 断点续传
  summarization/ 摘要树
  graph/        图谱子系统（抽取/校验/合并/归一化/缓存/检索）
  ingestion/    章节感知切分 + 语义边界分块
  llm/          Davy / Ollama LLM 工厂（重试、真流式）
  utils/        JSON 解析、原子写入、分词、并行工具
scripts/        图谱构建 / 可视化脚本
tests/          离线单测（188 例）
data/raw/       源语料（txt / docx，章节感知切分）
assets/         术语映射、CA 证书
```

更多架构细节与开发约定见 [CLAUDE.md](CLAUDE.md)，历次改进记录见[项目改进分析报告](项目改进分析报告.md)。
