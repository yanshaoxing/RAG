# 中文小说多书 RAG 问答系统

基于 LlamaIndex 的中文小说知识库检索问答系统（多书语料，默认语料《遥远的救世主》）：混合检索（向量 + BM25）、层次化摘要树、Kuzu 知识图谱三路证据融合，经 rerank 后流式生成带引用的回答。

## 文档导航

| 文档 | 内容 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 整体架构：索引构建 5 阶段、查询流程、并行点、全景流程图、关键设计与优化、目录结构 |
| [docs/operations.md](docs/operations.md) | 部署与使用：环境要求、安装、CLI/UI 用法、多书语料（切书/新增书）、索引重建、已知限制 |
| [docs/models.md](docs/models.md) | 模型配置：公网阿里云四端点、provider 切换、思考开关、换嵌入模型须知 |
| [CLAUDE.md](CLAUDE.md) | 架构细节与开发约定（面向改代码） |

## 功能特性

- **三路查询改写**（并行）：自然语言改写 / HyDE → 向量检索，关键词扩展 → BM25 检索；术语映射（通俗名 → 原文术语）在改写关闭时也生效
- **混合检索**：三路召回 → 摘要冗余过滤 → gap 过滤 → RRF 融合 → cross-encoder 重排（公网 qwen3-rerank / 内网 bge-reranker-v2-m3；失败自动降级 RRF 顺序，瞬时故障先快速重试）
- **查询分解**：复杂问题自动拆分为子查询并行检索，按名次 RRF 融合合并
- **层次化摘要树**：L1 逐块 → L2 小节 → L3 章节 → L4 全书四级摘要，混入主索引解决"宏观问题查不到"；LLM 失败自动降级并显式标记
- **知识图谱**：按节 LLM 抽取实体/关系 → 规则过滤 → 不同模型交叉校验 → 描述合并 → 别名归一化 → Kuzu 持久化；构建为有界流水线并发，SQLite 缓存支持断点续传
- **真流式输出**：SSE 逐 token 端到端流式，增量剥离 `<think>` 思考块，绕开 llama_index Refine 合成器直接 `stream_chat` 逐 token 产出
- **用量计量**：token 取自服务端 `usage`（计费同源，非本地估算），单列 reasoning token；CLI 与 UI 每次查询输出「耗时分解 + 各模型 token + 估算费用」
- **工程化**：分阶段索引（`_DONE.json` 完成标记 + 原子写入）、embedding 分段断点续传、离线单测（无需内网）、评测脚手架（`eval/`）

## 快速开始

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 配置公网 key（写入项目根目录 .env，已 gitignore）：见 docs/models.md
python -m app.cli "丁元英的私募基金为什么解散"   # 单次查询（默认书）
python -m app.cli --list                       # 列出全部可用书
python -m app.cli -c WuLingChaShi "陈伯是谁？"   # 指定书查询
streamlit run app/ui.py                        # Web UI（侧边栏随时切书）
```

首次运行会自动依次构建 5 个索引阶段（分块 → 摘要树 → BM25 → 向量 → 图谱），持久化到激活语料的 `corpora/<slug>/data/`；之后启动直接从磁盘加载。完整用法（多书、重建、图谱脚本、测试）见 [docs/operations.md](docs/operations.md)。
