# 评测脚手架（P0-1）

给检索/回答调参提供**可量化的对照基线**，不再盲调。当前仅覆盖已构建的两本语料
（`YaoYuanDeJiuShiZhu` 第一章 8 块、`WuLingChaShi` 3 块），**不依赖整书**。

## 目录结构

```
eval/
  <slug>/
    qa.jsonl                  # 人工标注的评测集（每行一题）
    baseline_retrieval.json   # scripts/eval_retrieval.py --save 的产物
    baseline_answer.json      # scripts/eval_answer.py --save 的产物
```

## qa.jsonl 每行字段

| 字段 | 说明 |
|---|---|
| `id` | 题号（如 `y03`） |
| `type` | 四类之一：`fact`（事实）/ `relation`（人物关系）/ `cross`（跨段/跨章）/ `macro`（宏观概述） |
| `question` | 用户问题 |
| `gold_chunk_ids` | 标准 chunk 的 **node_id**（引用 `data/chunks/chunks.json`），供检索评测判命中 |
| `gold_answer_points` | 标准答案要点列表，供回答评测（LLM-as-judge）判完整度 |

> **命中判定为什么按内容、不按 node_id**：检索返回节点的 `node_id` 会随索引重建变化，
> 不是稳定锚点。因此 gold 用 chunks.json 的**顺序下标**标识（由 `gold_chunk_ids` 经
> id→下标换算得到），被检索到的原始 chunk 则按**正文内容**反解回下标；摘要节点按其
> `summary_chunk_range` 覆盖区间给分（摘要本就是合法的检索目标）。逻辑见
> `scripts/eval_common.py`，纯函数，由 `tests/test_eval_common.py` 离线覆盖。

## 用法

```bash
# 检索评测（只检索、不合成回答，便宜）——输出按 type 分组的 Recall@k / MRR / 首命中排名
python scripts/eval_retrieval.py                       # 默认语料
python scripts/eval_retrieval.py -c WuLingChaShi
python scripts/eval_retrieval.py -k 3,5,10 --save eval/YaoYuanDeJiuShiZhu/baseline_retrieval.json

# 单参数扫描（消融/标定）：--config-override 可重复，值自动转型，构建引擎前 setattr 到 config
python scripts/eval_retrieval.py --config-override RERANK_ENABLED=false
python scripts/eval_retrieval.py --config-override RRF_K=30 --config-override GRAPH_ENABLED=false

# 回答评测（完整问答 + 异模型 judge，较贵）——忠实度 / 引用 / 完整度（1~5）+ 要点命中率
python scripts/eval_answer.py -c WuLingChaShi --limit 3        # --limit 冒烟
python scripts/eval_answer.py --save eval/YaoYuanDeJiuShiZhu/baseline_answer.json
```

judge 刻意用**与回答模型不同**的模型（`config.ALIYUN_VALIDATE_MODEL`，沿用
`create_validate_llm` 的交叉校验思路，避免自己判自己）。judge 提示词在
`rag/prompts.py::EVAL_JUDGE_TEMPLATE_STR`。

## 用它做什么（解锁的调参项）

- 标定 `BM25_MIN_SCORE` / `VECTOR_MIN_SCORE`（config 注释自己写着「重建后需按实际分布重新标定」）。
- 消融实验：关摘要树 / 关图谱 / 关分解 / 关 rerank 各自的召回收益，对着 `baseline_*.json` 比。
- 扫 `RRF_K` / `GAP_THRESHOLD` / `SUMMARY_REDUNDANCY_THRESHOLD` / `RERANK_CANDIDATE_POOL_SIZE`。

> **规模提示**：当前两本都是极小语料，Recall@10 基本满分、区分度有限；真正的标定要等
> 整书语料入库后（IMPROVEMENTS.md 批次 7）再跑，评测脚手架届时直接复用。
