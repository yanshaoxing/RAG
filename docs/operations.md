# 部署与使用

> 返回 [README](../README.md) · 相关文档：[架构](architecture.md) · [部署与使用](operations.md) · [模型配置](models.md)

## 环境要求

- Python 3.12，依赖版本固定在 `requirements.txt`
- 当前默认走**公网阿里云**端点（见上文「模型配置」），需要在 `.env` 中配置三个 `RAG_PUBLIC_*_API_KEY`；内网配置（Davy 云端 + 远程 Ollama + vLLM rerank）仍保留，可按 provider 切回
- 内网端点与密钥可用环境变量覆盖（未设置时用内置默认值）：
  `RAG_DAVY_BASE_URL` / `RAG_DAVY_API_KEY` / `RAG_EMBED_OLLAMA_BASE_URL` / `RAG_RERANK_BASE_URL` / `RAG_GRAPH_VALIDATE_LLM_BASE_URL` / `RAG_DEBUG`
- 所有可选功能均有独立开关（`rag/config.py`）：`REWRITE_ENABLED` / `DECOMPOSE_ENABLED` / `RERANK_ENABLED` / `SUMMARY_TREE_ENABLED` / `GRAPH_ENABLED` / `GRAPH_VALIDATE_ENABLED` / `DEBUG`

## 安装

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 使用

```bash
# CLI（从仓库根目录以模块方式运行）
python -m app.cli "丁元英的私募基金为什么解散"   # 单次查询（默认书）
python -m app.cli --list                       # 列出全部可用书
python -m app.cli -c WuLingChaShi "陈伯是谁？"   # 指定书查询
python -m app.cli                              # 交互模式：索引加载一次，循环提问
python -m app.cli -c WuLingChaShi              # 交互模式（指定书；整个会话绑定该书）

# Streamlit Web UI（左侧边栏下拉框随时切书）
streamlit run app/ui.py

# 知识图谱构建（可断点续传）
python scripts/build_full_graph.py            # 自动续传
python scripts/build_full_graph.py --force    # 删缓存全量重建
python scripts/build_full_graph.py 31         # 从第 31 个 chunk 续跑

# 图谱可视化（输出 pyvis 交互式 HTML）
python scripts/visualize_graph.py             # 度数 top-80 实体
python scripts/visualize_graph.py 150         # top-150

# 离线单测（不依赖内网，改动管线代码前必跑）
.venv/bin/python -m pytest tests/ -q
```

首次运行会自动依次构建 5 个索引阶段（分块 → 摘要树 → BM25 → 向量 → 图谱），各阶段独立持久化到激活语料的 `corpora/<slug>/data/` 下并写入 `_DONE.json` 完成标记；之后启动直接从磁盘加载。

## 多书语料

每本书一个目录 `corpora/<slug>/`。仓库自带两本书：`YaoYuanDeJiuShiZhu`（《遥远的救世主》，默认书）和微型演示语料 `WuLingChaShi`（《雾岭茶事》，已构建索引），可直接体验双书切换。

### 如何切书（三种方式）

```bash
# ① CLI：--corpus / -c 指定书（不带则用默认书；--list 先看有哪些书）
python -m app.cli --list
python -m app.cli -c WuLingChaShi "陈伯是谁？"
python -m app.cli -c WuLingChaShi              # 交互模式整个会话绑定这本书，换书需退出重开

# ② Web UI：运行中随时切
streamlit run app/ui.py
#    左侧边栏「📖 选择书目」下拉框切换；每本书首次选中时构建/加载引擎（之后缓存），
#    对话历史各书独立，来回切换不丢

# ③ 环境变量：改进程默认书（CLI 不带 -c 时的书 / UI 下拉框的初始选中项）
RAG_CORPUS=WuLingChaShi python -m app.cli "问题"
RAG_CORPUS=WuLingChaShi streamlit run app/ui.py
```

### 如何新增一本书

1. 新建 `corpora/<新slug>/`，放入 `corpus.json`（必填 `title` 书名 + `context` 原著背景块，背景块会注入改写/摘要/图谱等 prompt，写得越具体检索质量越好）
2. 原文（txt / docx）放入 `corpora/<新slug>/raw/`
3. 可选：`terminology.json`（通俗名→原文术语）、`graph_rules.json`（图谱规则补充，如男性角色名单）；章节标题格式特殊（非「一、」/「第X回」中文数字式）的书无需额外配置——首次构建会自动采样送 LLM 检测结构并写回 `corpus.json`，也可手工预填 `chapter_pattern` 正则
4. 用上面任一方式选中该书 —— 首次选中自动构建全部索引（构建走 LLM，公网模型下注意语料规模与费用）

### 目录约定

```
corpora/<slug>/
  corpus.json        必需：title（书名）+ context（注入 prompt 的原著背景块），可选 author/description，
                     可选 chapter_pattern/subsection_pattern（章节/小节正则，手工填或 LLM 结构检测自动写回）
  raw/               源语料（txt / docx）
  terminology.json   可选：通俗名 → 原文术语映射
  graph_rules.json   可选：图谱规则补充（与 rag/graph/rules.json 基础规则合并：列表并集、标量覆盖）
  data/              5 阶段索引产物（重建时删除对应子目录即可）
```

prompt 模板不含任何书名/人物硬编码：语料相关模板带 `{book_title}` / `{corpus_context}` 标记，访问时由 `rag/prompts.py` 的 `__getattr__` 注入激活语料档案（`rag/corpus.py`）。config 的语料相关路径为动态属性，实时跟随激活语料。并发约定：查询期语料状态在**引擎构建时绑定**（索引/图存储/改写 prompt+术语表），切书不影响已建引擎；构建在全局构建锁内进行。

## 索引重建

删除对应阶段目录后重跑任一入口即可——依赖它的后续阶段会一并重建，之前的阶段从磁盘加载（下表目录均在激活语料的 `corpora/<slug>/data/` 下）：

| 阶段 | 目录 |
|---|---|
| 1 分块 | `chunks/` |
| 2 摘要树 | `summary_tree/` |
| 3 BM25 | `bm25/` |
| 4 向量（FAISS HNSW） | `vector/` |
| 5 图谱（Kuzu） | `graph_db/`（抽取缓存在 `graph_cache/`，不随重建删除） |

向量阶段的 embedding 分段落盘到 `embed_cache/`，中断后续跑只补缺失段。

## 已知限制与取舍

- **全量校验开销**：图谱校验当前对每条关系都过 LLM（阈值 2.0）；重建耗时敏感可降到 0.7 只校验低置信度关系。
- **标题风格检测仅采样前 3000 字符**：长序言可能导致全书判型错误（当前语料无此问题，换语料需注意）；无法识别的标题统一归入"概述"。
- **嵌套并发上限**：子查询（2）× 三路改写（2）最坏并发 4 > Davy 的 429 阈值 2，靠重试退避兜底。
- **BM25 阶段约 2 倍内存**（全量 `model_copy` + `original_text` 元数据）：单本小说规模无碍，扩大语料前需改。
- **密钥明文默认值**：内网工具，可用环境变量覆盖，属有意选择。
- **切书粒度为引擎级**：多书并存靠"每书一个引擎"，无跨书混合检索/自动路由（有意取舍——跨书混检会互相污染 BM25 词频、rerank 与图谱实体归一化）。
