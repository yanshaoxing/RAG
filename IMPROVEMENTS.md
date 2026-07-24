# 改进清单

盘点日期：2026-07-23 · 基线 commit `c5220a9` · 离线单测 224 例全通过（2.7s）

按「先能量化、再谈优化」的顺序排列。P0 = 阻碍后续一切调优或有安全风险；P1 = 明确收益、成本可控；P2 = 锦上添花。

---

## 📍 下次继续（交接）

**当前进度**：批次 0、1、2 已完成；期间连带完成 P0-2、P1-4、P1-6、P1-7 及模型迁移。离线单测 **289** 例全过 + `ruff check .` 零告警。最近提交待记，工作区含批次 2 改动。

**已完成**（详见各条目内的 ✅ 标记）：

| 项 | 内容 | commit |
|---|---|---|
| 批次 0 | 密钥去硬编码 + provider/开关环境变量化 + CLI 显式传语料 | `6def980` |
| 模型迁移 | cn-beijing 单 workspace + 关闭思考（`ALIYUN_ENABLE_THINKING`）+ 嵌入模型一致性守卫（P2-2） | `14edb14` |
| 批次 1 | token/耗时计量（`rag/metering.py`，token 取服务端 usage 不本地估算） | `5911630` |
| P1-6 | 真流式端到端修复（绕开 llama_index Refine 合成器） | `a05d322` |
| P1-7 | 如实申报上下文窗口（1M），避免全书查询多轮 refine | `a05d322` |
| 批次 2（P1-4） | GitHub Actions（ruff + pytest）+ `pyproject.toml`（pytest/ruff 配置） | 待记 |

**批次 2 落地细节**：`pyproject.toml` 收编 pytest（`testpaths`/`pythonpath`）与 ruff（`select = E,F,I,UP`）；`.github/workflows/test.yml` 在 push/PR 上跑 `ruff check .` + `pytest`。**ruff `line-length` 取 120**（非计划里的 100）——本项目是中文代码库，ruff 按每汉字 1 宽计数，100 对应视觉约 160+ 列过紧，44 处 E501 里 6 处还在 `prompts.py` 提示词字符串内，改动风险高；120 只剩 2 处、且不动提示词内容（已征得用户同意）。`target-version = py312` 连带触发 UP047，已把 `run_parallel_captured` 改为 PEP 695 泛型语法。全量 autofix 涉及 44 文件（导入排序 + 类型现代化 `Optional[X]→X|None` 等，纯机械、已 import 冒烟 + 全测验证）。`ruff==0.14.4` 已加进 `requirements.txt`。

**下一步**：**批次 3（补 `staged_indexer` 阶段跳转矩阵 / `graph_retriever` / `cache` 测试，1 人日）** → 批次 4 → 批次 5（评测脚手架，仍不需要整书）。分界线之后（批次 7~9）才需要整书。

**挂起项（需用户操作 / 确认，不阻塞我继续）**：
- ⚠️ **Davy 旧 key 仍在 git 历史**（`c5220a9` 及之前 `rag/config.py:138`）——用户已知悉，暂缓轮换。
- 💰 **单价待真实账单校准**：`config.MODEL_PRICES` 取自 LLM.txt（dashscope 公共模型价），实际走 workspace 专属部署，计费口径可能不同。批次 1 的计量已就绪，跑一次真实查询对一次账单即可校准。
- 📄 **全书语料**：批次 7 才需要，用户决定最后再传。

---

## P0-1 没有评估集，所有检索调参都是盲调

**现状**：仓库里没有任何 QA 评测集、检索指标脚本或回归基线。224 例单测全部是"管线不崩"的结构性测试，没有一条断言"这个问题能召回正确的段落"。

**为什么卡住别的事**：
- `rag/config.py:190-192` 的注释自己写着 `BM25_MIN_SCORE` "此下限基本不起过滤作用；重建索引后需按实际分布重新标定" —— 没有评估集就无法标定。
- `GAP_THRESHOLD`、`SUMMARY_REDUNDANCY_THRESHOLD`、`RRF_K`、`RERANK_CANDIDATE_POOL_SIZE`、要不要开 HyDE、要不要开分解，全是可调项，目前只能凭感觉。
- 摘要树 / 图谱 / 分解每一项都显著增加 LLM 调用成本，但没有任何证据说明它们各自带来多少召回收益。

**建议**：
1. 建 `eval/<slug>/qa.jsonl`：每条 `{question, gold_chunk_ids | gold_answer_points, type}`，type 覆盖事实、跨章、宏观概述、人物关系四类。全本 30~50 条即可起步，人工标注约半天。
2. 写 `scripts/eval_retrieval.py`：跑检索管线，输出 Recall@k / MRR / 命中 gold 的平均排名，按 type 分组；支持 `--config-override KEY=VALUE` 做单参数扫描。
3. 写 `scripts/eval_answer.py`：LLM-as-judge 打分（忠实度 / 引用正确 / 完整度），复用 `create_validate_llm()` 这类"异模型"思路避免自己判自己。
4. 结果落 `eval/baseline.json`，作为消融实验（关摘要树 / 关图谱 / 关分解 / 关 rerank）的对照。

**工作量**：1~2 人日。**收益**：解锁下面所有调参项。

---

## P0-2 内网 API key 硬编码在代码里，且已进 git 历史

**现状**：`rag/config.py:138`

```python
DAVY_API_KEY = os.environ.get("RAG_DAVY_API_KEY", "ZmU0Nt7kbSGH8SEIwvFCSVYwGDiTSQNMezbfLZW_3Aw")
```

公网三把 key 都规规矩矩走 `.env`（已 gitignore），唯独 Davy 的 key 留了明文兜底值，而且这个值在 git 历史里。README 里写着"公网 key 不入 git"，这条与之矛盾。

**建议**：
1. 联系内网侧轮换这把 key（历史无法靠改代码抹掉）。
2. 默认值改为 `""`，并在 `create_*_llm()` 里对 `provider=="davy"` 且 key 为空时抛出带修复提示的异常。
3. 把 Davy key 挪进 `.env`，与公网 key 一致。

**工作量**：15 分钟（轮换除外）。

---

## P1-1 三路检索是串行的，白白多等一次 embedding 往返

**现状**：`rag/retrieval/hybrid_retriever.py:88-95`

```python
vec_nl   = self._vector_retriever.retrieve(QueryBundle(nl_query))     # 一次 embedding API
vec_hyde = self._vector_retriever.retrieve(QueryBundle(hyde_passage)) # 又一次 embedding API
bm25_kw  = self._bm25_retriever.retrieve(...)                          # 本地
```

两次向量检索各含一次公网 embedding 请求（`text-embedding-v4`，ap-southeast-1，单次通常 0.3~1s），完全独立却串着跑。项目里已经有现成的 `run_parallel_captured`（三路改写、图检索+主检索都用了它），这里漏了。

**建议**：用 `run_parallel_captured([...], max_workers=3)` 并发三路，日志顺序仍由该工具保序。注意 BM25 那路的 `original_text` 还原逻辑跟着一起搬进 task 里。

**工作量**：30 分钟。**收益**：单次查询省一次 embedding 往返；子查询分解场景（最多 5 个子查询）收益乘以子查询数。

---

## P1-2 查询期没有任何耗时 / 成本度量

**现状**：管线日志只有"第几步、召回几条"，没有每步耗时。一次查询实际的 LLM 调用是：复杂度分类（可能）→ 3 路改写 → 图谱实体抽取 → 最终回答，加上 2 次 embedding + 1 次 rerank，一共 6~8 次网络往返，但没人知道时间花在哪。

**建议**：
1. 在 `rag/logging_utils.py` 加一个 `step_timer(name)` 上下文管理器，各步骤包一层，收尾打一行"耗时汇总"。
2. `DavyLLM._call` / `stream_chat` 记录 usage（阿里云 OpenAI 兼容响应带 `usage`），按查询累计 prompt/completion tokens，日志末尾输出，便于估成本。
3. UI 的"运行流程"折叠面板顺带展示耗时条。

**工作量**：半天。**收益**：P0-1 的评估脚本可以顺带产出延迟/成本维度，做"质量-成本"权衡而不是只看质量。

---

## P1-3 全本语料还没跑通，当前所有产物都是 1 章 8 个 chunk

**现状**：`corpora/YaoYuanDeJiuShiZhu/raw/` 只有 `Chapter 1.txt`（6571 字，且只是首章的一部分），`data/chunks/chunks.json` 里 8 个节点、平均 833 字/块；演示语料 3 个节点。也就是说整条管线从没在真实规模上跑过。按实测的 ~820 字/chunk 推算，全书 35 万字约 400~450 个 chunk。

**规模化时会先暴露的点**：
- 摘要树：L1 ≈ chunk 数 / 5 ≈ 90 次调用，L2 ≈ 小节数，L3 ≈ 章节数，合计约 300 次 LLM 调用 × 并发 2。
- 图谱按节抽取 + 异模型校验：每节 2 次 LLM 调用，再加主线程的 merge/canonicalize，是全流程耗时大头（小时量级）；断点续传已具备但没在长跑下验证过。
- `BM25_MIN_SCORE` / `VECTOR_MIN_SCORE` 的分布会完全变样，必须重标（见 P0-1）。
- `data/` 目前进 git（56 个文件 / 12MB，`.git` 26MB）。全本的 `vector/` + `graph_db/` 会到百 MB 量级，继续直接进 git 会让仓库迅速膨胀。

**建议**：
1. 先在全本上只跑到阶段 1（分块），确认章节正则命中数、chunk 总数、预估 LLM 调用量，再决定摘要树/图谱是否分批开。
2. 全本入库前决定 `data/` 的存储策略：保留小语料进 git（演示用），全本索引走 git-lfs 或加进 `.gitignore` 并提供 `scripts/` 一键重建。

**工作量**：跑批时间为主，决策 1 小时。

---

## P1-4 缺 CI、缺 lint/格式化配置 —— ✅ 已完成（批次 2）

**已落地**：`.github/workflows/test.yml`（push/PR 触发，`pip install -r requirements.txt` → `ruff check .` → `pytest tests/ -q`）+ `pyproject.toml`（pytest 配置 + ruff `select = E,F,I,UP`，`line-length = 120`，见「下次继续」的落地细节）。未做 pre-commit（可选，暂缓）。

**原始现状**：无 `.github/workflows/`、无 `pyproject.toml` / `ruff.toml` / `pytest.ini` / pre-commit。测试 2.7 秒跑完且完全离线（不依赖内网和 LLM），是天生适合 CI 的。

**建议**：
1. 加 `.github/workflows/test.yml`：`pip install -r requirements.txt && pytest tests/ -q`。
2. 加 `pyproject.toml` 收编 pytest 配置 + ruff（line-length 100，选 E/F/I/UP 规则集即可，不必激进）。
3. 可选 pre-commit：ruff + 一条"禁止 `rag/prompts.py` 外出现 prompt 文本"的自定义检查，把 CLAUDE.md 里的约定变成机器可执行的。

**工作量**：1~2 小时。

---

## P1-5 核心装配路径与图谱运行时无单测覆盖

**现状**：以下模块没有对应测试文件：

```
rag/engine/bootstrap.py      rag/engine/query_engine.py    rag/graph/graph_retriever.py
rag/graph/cache.py           rag/graph/extractor.py        rag/graph/merger.py
rag/graph/schema.py          rag/indexing/staged_indexer.py rag/llm/factory.py（仅流式部分有）
rag/utils/text.py            rag/logging_utils.py
```

其中风险最高的三个：
- **`staged_indexer`**：阶段跳转矩阵（`start_from <= N` 的五处分支，`staged_indexer.py:388-430`）是全项目最容易改错的地方——删某一阶段目录后从哪开始、哪些前置产物需要加载，全靠这段 if 链，且改错时表现为"静默重建整本书"而不是报错。可用 fake 索引 + monkeypatch 各阶段函数，断言"删第 N 阶段 → 恰好调用第 N..4 阶段、加载第 0..N-1 阶段"。
- **`graph_retriever`**：LLM 抽实体 → Kuzu 参数化查询 → 邻居遍历，可 mock LLM + 内存 Kuzu 覆盖。
- **`cache.py`**：断点续传的正确性直接决定长跑图谱构建能不能恢复。

**工作量**：1 人日（三个高风险模块）。

---

## P2-1 CLI 的 `--corpus` 走的是隐式全局状态

**现状**：`app/cli.py:136-148` 先 `corpus.set_active_corpus(args.corpus)` 切全局激活语料，然后 `build_engine()` 不带参数调用；而 `build_query_engine(corpus_slug)` 本来就支持显式传 slug（`bootstrap.py:57`）。功能正确，但依赖"前面已经切过全局态"这一隐式前提，与 UI 的显式传参路径（`ui.py:52`）不一致。

**建议**：`build_engine(args.corpus)` 显式传下去，删掉前面的 `set_active_corpus`（保留档案校验与友好报错，可用 `corpus.load_profile` 代替）。

**工作量**：15 分钟。

---

## P2-2 多书共享同一个全局 embedding 模型，这是硬约束但没写下来

**现状**：`bootstrap.init_settings()`（`bootstrap.py:30-47`）把 embedding 模型设进全局 `Settings`，而向量检索器在查询时用的就是全局 `Settings.embed_model`。CLAUDE.md 里"查询期语料状态在引擎构建时绑定"的并存契约，对 embedding 这一项其实不成立：**所有书必须共用同一个 embedding 模型和维度**，否则后切的书会污染先构建的引擎。

**建议**：二选一 ——
- （便宜）在 `corpus.py` / CLAUDE.md 显式写明该约束，并在 `list_corpora()` 或构建时校验各书 `data/vector/` 的维度与当前 `EMBED_VECTOR_DIM` 一致，不一致直接报错而不是算出乱七八糟的相似度；
- （彻底）把 embed_model 也绑进引擎（构建时把 retriever 的 embed_model 显式传入），彻底兑现"多引擎并存"的说法。

**工作量**：校验版 1 小时；绑定版半天。

---

## P2-3 重建阶段只能手工 `rm -rf`

**现状**：README/CLAUDE.md 里重建索引的操作是"删掉对应目录再跑入口"。容易删错语料（路径含 slug）、也容易在还没想清楚时误删下游阶段。

**建议**：加 `python -m app.cli --rebuild {chunks,summary,bm25,vector,graph} [-c slug]`，内部按阶段表删除该阶段及下游目录并打印将要重建的阶段清单，`--yes` 才真删。阶段表 `staged_indexer.py:333-338` 已经现成。

**工作量**：1~2 小时。

---

## P2-4 没有查询级缓存

**现状**：完全相同的问题重复提问会重跑整条管线（3 次改写 LLM + 2 次 embedding + rerank + 图谱抽实体 + 回答）。演示和评测场景下这是纯浪费。

**建议**：在 `HybridRetriever._retrieve` 前加一层可选的 LRU（key = 语料 slug + 原始 query + 影响检索的 config 指纹），只缓存检索结果不缓存回答（回答仍流式生成，保持体验）。配 `QUERY_CACHE_ENABLED` 开关，默认关，评测脚本里打开。

**工作量**：2~3 小时。

---

## P2-5 若干健壮性小口子

| 位置 | 问题 | 建议 |
|---|---|---|
| `hybrid_retriever.py:300` | `n.score >= min_score`，`score` 为 `None` 时抛 `TypeError`。当前所有检索路径都会赋分，属潜在风险 | 改成 `(n.score or 0.0) >= min_score` |
| `config.py:21-38` | `.env` 加载用 `os.environ.setdefault`，已存在的（可能是过期的）环境变量会静默压过 `.env` | 保持行为但在 DEBUG 下打一行"来源：env / .env"，排查密钥问题时省事 |
| `config.py` provider 开关 | `ANSWER_PROVIDER` 等只能改代码，不能用环境变量覆盖 | 统一 `os.environ.get("RAG_ANSWER_PROVIDER", "aliyun")`，消融实验不用改代码 |
| `ui.py:48-52` | `@st.cache_resource` 按 slug 缓存引擎，书多了内存无上限增长 | 加 `max_entries=3` |
| `README.md` | 单文件 65KB，架构图 / 操作指引 / 模型配置全挤在一起 | 拆 `docs/`：`architecture.md` / `operations.md` / `models.md`，README 只留导航 |

---

## P1-6 「真流式」实际上没有生效，被上游合成器阻断 —— ✅ 已修复

**修复**（commit 待记）：`GraphAugmentedQueryEngine` 覆盖 `_query` + `synthesize`，流式时绕开 llama_index 的 Refine 合成器，直接用 `QA_TEMPLATE` 拼接参考资料后调 `Settings.llm.stream_chat()` 逐 token 产出；非流式仍走父类成熟的 compact。因 `ALIYUN_CONTEXT_WINDOW` 已如实申报（1M，见 P1-7），单本查询的参考资料必落入单块，compact 本就等价于「一次 QA 调用」，故行为与之一致、只是真正逐 token。实测 `query()` 从「阻塞 7.5s、response_gen 只 yield 1 次」变为「检索 5.7s 返回、迭代 31 次、首块 1.3s、生成在迭代期间」；批次 1 的耗时计量里「步骤 4 回答生成」从 `0.00s` 变为真实的 `1.66s`。新增 `tests/test_query_engine.py`（6 例）锁定逐 token 行为与 `_query` 分发路径。

> 关键坑：父类 `RetrieverQueryEngine._query` 直接调 `self._response_synthesizer.synthesize`，**绕过** `self.synthesize`，故必须同时覆盖 `_query` 让它经 `self.synthesize` 分发。

---

<details><summary>原始分析（保留备查）</summary>

**现状**：README 与 CLAUDE.md 都写着"SSE 逐 token 渲染"，但**端到端并没有实现**。实测（批次 1 的耗时计量直接暴露了它——"步骤 4 回答生成: 0.00s"）：

```
query() 返回耗时: 7.54s          ← 回答已在此期间生成完毕
有 response_gen: True
迭代 1 次，首块耗时: 0.000s      ← 只 yield 一次，瞬间返回
```

用户实际体验是「等 7.5 秒 → 全文一次性出现」，而不是逐字浮现。

**根因**（llama_index 0.14.23 `response_synthesizers/refine.py`，`DefaultRefineProgram.stream_call`）：

```python
answer = ""
# ... We want to mimic that behavior here so it behaves similarly across the two cases
for token in self._llm.stream(self._prompt, **kwds):
    answer += token                                    # 把整个流累积成字符串
yield StructuredRefineResponse(answer=answer.strip(), query_satisfied=True)   # 只 yield 一次
```

上游是**有意为之**（注释说是为了与结构化输出行为一致）。我们这一侧的 `DavyLLM.stream_chat` 是真流式（`tests/test_llm_stream.py` 有断言），问题不在项目代码。

采用了方案 2（绕开合成器）。

</details>

---

## P1-7 未申报上下文窗口，全书查询会触发多轮 refine（已修）

**现状（已修复）**：`DavyLLM.metadata` 此前只声明 `model_name` 与 `is_chat_model`，llama_index 遂按默认值 `context_window=3900` / `num_output=256` 处理。

**后果**：合成器按"可用上下文"把参考资料 `repack` 成多块。全书查询送入 `FINAL_TOP_K=10` × `CHUNK_SIZE=1024` ≈ 1 万字正文，远超默认的 3644 可用 token → 拆成多块 → **多轮 refine**：LLM 调用次数翻几倍（多花钱）、信息在逐轮改写中损耗（答案变差），且 `Refine._run_refine_loop` 的 `if isinstance(response, Generator): response = get_response_text(response)` 会把上一轮的流式生成器消费掉。

小语料（当前 8 个 chunk）不触发，所以此前一直没暴露——**正是批次 7 全书入库时必然踩到的坑**。

**已落地**：`config.ALIYUN_CONTEXT_WINDOW`（1048576 = 1M）/ `ALIYUN_NUM_OUTPUT`（8192），`DavyLLM` 如实申报。取值已按 `LLM.txt` 核准：qwen3.5-flash / qwen-flash 上下文均为 1M；`num_output` 是「为输出预留、从可用上下文扣除」的额度，按 QA 回答的合理上限取 8192（模型实际最大输出 32K~64K，但回答远短于此，预留过多只会无谓压缩可用上下文）。

---

## P2-6 阶段依赖被写死成线性链，图谱被无谓连带重建

**现状**：`staged_indexer.py` 的阶段表是有序列表，重建时「删除 start_from 及之后所有启用阶段的目录」。但**图谱阶段（⑤）的输入是 `raw/` 原文**（`build_graph(load_documents())`，按节抽取，独立于检索分块），它并不依赖分块/摘要/BM25/向量中的任何一个。

**实测**：2026-07-23 换嵌入模型后只删了 `data/vector/`，图谱却被一并删除重建（WuLingChaShi 的实体数因换模型 33→24）。小语料只多花几秒，**全书则是数小时 + 数元的纯浪费**。

**建议**：把阶段表从「有序列表」改成「带依赖声明的图」——`chunks → summary → {bm25, vector}`，`graph → (raw)`。重建时按依赖闭包而非下标区间删除。与批次 4 的 `--rebuild <stage>` 一起做，同一处改动。

**工作量**：2 小时（含 P1-5 的阶段跳转矩阵测试覆盖）。

---

## 成本测算（基于 LLM.txt 2026-07-23 的实际单价）

| 模型 | 输入 | 输出 | 缓存命中 |
|---|---|---|---|
| qwen3.5-flash（主：回答/改写/摘要/图谱抽取） | 0.2 元/M | 2 元/M | 0.02 元/M |
| qwen-flash（图谱校验） | 0.15 元/M | 1.5 元/M | 0.015 元/M |
| qwen3.7-text-embedding | 0.5 元/M | — | |
| qwen3-rerank | 0.5 元/M | — | |

**全书（35 万字 / 约 430 chunk / 约 160 节）一次完整构建**，按 1 token≈1 字的保守上限估：

| 阶段 | 输入 token | 输出 token | 费用 |
|---|---|---|---|
| ① 分块 | 0 | 0 | 0 |
| ② 摘要树 | ~116 万 | ~16.6 万 | **≈ 0.56 元** |
| ③ BM25 | 0 | 0 | 0 |
| ④ 向量（embedding） | ~52 万 | — | **≈ 0.26 元** |
| ⑤ 图谱（抽取+校验+归一） | ~155 万 | ~29 万 | **≈ 0.81 元** |
| **合计** | | | **≈ 1.6 元** |

中文实际约 0.7 token/字，真实开销约 **1.2 元**。单次查询约 **0.005 元**，50 条评测题约 **0.25 元**。费用完全不构成约束。

### ⚠️ 思考（reasoning）token 才是真正的成本变量

Qwen3 系 chat 模型**默认开启思考**，而 reasoning token 按输出价计费、且不进 `message.content`（`<think>` 剥离器看不到它，但照付）。实测同一条摘要请求：

| | 输出 token | 其中 reasoning | 耗时 |
|---|---|---|---|
| `qwen3.5-flash` 默认 | 5019 | 4987（99%） | 45.7s |
| `qwen3.5-flash` + `enable_thinking=False` | 24 | 0 | 0.7s |

概括质量无可见差异。全书构建约 750 次 LLM 调用，若不关思考：额外 ~375 万 reasoning token ≈ **7.5 元**（是正文成本的 5 倍），耗时 750×45s/并发2 ≈ **4.7 小时**。

→ 已落地：`config.ALIYUN_ENABLE_THINKING` 默认 `False`，`DavyLLM` 仅在显式配置时下发该参数（Davy 内网端点不认识它）。**全书构建因此从「约 9 元 / 5 小时」降到「约 1.6 元 / 几十分钟」**。最终回答环节是否值得开启思考，留到批次 8 有评估集后用数据决定。

### 另外两个由单价结构导出的优化点

1. **输出比输入贵 10 倍**（2 vs 0.2）。成本大头是输出量：`SUMMARY_PARENT_RATIO=0.20` / `SUMMARY_CHAPTER_RATIO=0.15` 直接决定摘要输出费用，图谱抽取的 JSON 输出是阶段⑤最贵的单项。想省钱先压输出比例，而不是改 L3 的输入来源。
2. **显式缓存命中价是输入价的 1/10**（0.02 vs 0.2）。图谱对同一节文本读两遍（抽取 + 校验）、摘要 L2/L3 对同一章原文各读一遍，都是天然的缓存场景。需 DashScope explicit cache 支持，列为后续优化。

> 注：单价取自 LLM.txt，实际计费口径待批次 1 的 token 计量 + 一次真实账单校准。

---

## 推进顺序（约束：先做不依赖整书的部分，需要整书的排在最后）

**分界线之前全部不需要整书，累计约 3~4 人日。**

### 批次 0 · 立即（<1 小时，零风险）—— ✅ 已完成（252 单测通过）
1. ✅ Davy key 去硬编码：默认值改为空串，`factory._require_api_key()` 在选用 davy / aliyun provider 且密钥缺失时立刻报错并指出该设哪个环境变量（此前是等到 401）。**⚠️ 旧 key 仍在 git 历史里，需联系内网侧轮换**
2. ✅ CLI `--corpus` 改显式传参：`corpus.load_profile()` 只校验档案不改全局态，语料切换统一由 `build_query_engine(slug)` 在构建锁内完成
3. ✅ `_gap_filter` 的 `score is None` 防御
4. ✅ 7 个 provider + 8 个功能开关全部支持环境变量覆盖（`_env_str` / `_env_bool`），新增 `tests/test_config_env.py` 28 例锁定语义与接线

### 批次 1 · 可观测性（半天）—— ✅ 已完成（283 单测通过）
5. ✅ token + 耗时计量：新增 `rag/metering.py`（contextvars 计量上下文，与 `capture_pipeline_logs` 同构）+ `step_timer`；四条链路全部接入：chat 非流式/流式、rerank、embedding（后者需 `rag/llm/embedding.py` 的子类拦截，因为 llama_index 会丢弃 usage）；`run_parallel_captured` 改用 `contextvars.copy_context()` 让 worker 共享同一 Meter（否则三路改写与子查询的 token 全漏）；CLI/UI 输出用量与耗时汇总

**关键原则**：token 一律取服务端 `usage`（计费同源），**不做本地分词估算**；未回传 usage 的调用记为「未计量」，单价未知的模型不参与合计 —— 绝不用估算值冒充实测值。

**该批次直接暴露了两个此前不知道的缺陷**：P1-6（真流式没生效）与 P1-7（未申报上下文窗口）—— 这正是"先做可观测性"的价值。

### 批次 2 · 防回归（2 小时）—— ✅ 已完成（289 单测 + ruff 零告警）
6. ✅ GitHub Actions 跑 pytest + `pyproject.toml`（pytest 配置 + ruff `E,F,I,UP`，`line-length=120`）（P1-4）；连带修 F841 死变量、F821 前向引用、UP047（PEP 695 泛型）

放在动核心逻辑之前，后续所有改动都有网。

### 批次 3 · 给高风险代码补测试（1 人日）（P1-5）
7. `staged_indexer` 阶段跳转矩阵（`staged_indexer.py:388-430`）
8. `graph_retriever`（mock LLM + 内存 Kuzu）
9. `cache.py` 断点续传 —— 决定全书图谱长跑能否恢复

批次 4 的改动与批次 7 的长跑构建的安全带，必须排在它们之前。

### 批次 4 · 性能与易用（1 人日）
10. 三路检索并发（P1-1）
11. `--rebuild <stage>` 命令（P2-3）
12. 查询级缓存（P2-4，默认关）
13. ~~embedding 维度校验~~ → 已在模型迁移时提前完成（升级为**模型名**校验，见 P2-6）
14. **阶段依赖图**（新增，见 P2-6）：图谱阶段只依赖 `raw/`，却因阶段表是线性链而被向量阶段的重建连带删除。小语料无感，全书是数小时 + 数元的浪费

### 批次 5 · 评测脚手架（1 人日，仍不需要整书）（P0-1 的非人工部分）
14. `eval/<slug>/qa.jsonl` schema —— gold 用**原文锚点**而非 chunk_id，标签才能跨重建/跨分块参数复用
15. `scripts/eval_retrieval.py`：锚点→chunk 解析 + Recall@k / MRR / 按题型分组
16. `scripts/eval_answer.py`：LLM-as-judge
17. 用《雾岭茶事》标 8~10 条题跑通全流程

演示语料只有 3 个 chunk，数字无意义，但足以验证 schema、锚点解析、指标计算与报告格式。等书到位后只剩纯人工的标注工作。

### 批次 6 · 文档（可选，2 小时）
18. README 拆 `docs/`；CLAUDE.md 补 embedding 全局约束与新命令

---
### ⬇️ 以下需要整书 ⬇️

**批次 7 · 入库**：传书 → 只跑阶段①（零 LLM 成本）→ 取真实章节/小节/chunk 数 → 配批次 1 的计量算出精确预算 → 先建 ①③④（仅 embedding 费用，约 0.27 元）

**批次 8 · 基线**：标 30~50 条锚点题 → 出检索基线数字

**批次 9 · 标定与消融**：min-score / gap / RRF 标定 → 开摘要树量增量收益 → 开图谱量增量收益 → 据此决定两个模块的去留
