"""检索评测 —— 跑真实检索管线，按 type 输出 Recall@k / MRR / 首命中排名。

只做检索（不合成回答），因此不花回答 LLM 的钱，但仍走改写 + 三路 + 过滤 + RRF + 重排
（以及可选图增强）。gold_chunk_ids 命中判定见 eval_common.covered_indices（摘要节点按其
覆盖区间给分）。

用法（仓库根目录）：
  python scripts/eval_retrieval.py                       # 默认语料，默认 qa 集
  python scripts/eval_retrieval.py -c WuLingChaShi
  python scripts/eval_retrieval.py -k 3,5,10
  python scripts/eval_retrieval.py --config-override RERANK_ENABLED=false --config-override RRF_K=30
  python scripts/eval_retrieval.py --save eval/baseline_retrieval.json
  python scripts/eval_retrieval.py --limit 3            # 只跑前 3 题（快速冒烟）

--config-override 支持单参数扫描：可重复传，值按 bool/int/float/str 自动转型，
在构建引擎之前 setattr 到 rag.config 上（检索相关配置在引擎构建期固化，故必须先改）。
"""

import argparse
import json
import os
import sys

# 允许 `python scripts/eval_retrieval.py` 直接运行（把仓库根塞进 sys.path）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_common import (  # noqa: E402
    aggregate_retrieval,
    evaluate_retrieval,
    load_chunk_index_map,
    load_chunk_norms,
    load_qa,
    parse_overrides,
)

_TRACKED_CONFIG_KEYS = [
    "REWRITE_ENABLED", "DECOMPOSE_ENABLED", "RERANK_ENABLED", "GRAPH_ENABLED",
    "SUMMARY_TREE_ENABLED", "RETRIEVAL_TOP_K", "RRF_K", "RERANK_CANDIDATE_POOL_SIZE",
    "GAP_THRESHOLD", "SUMMARY_REDUNDANCY_THRESHOLD",
]


def _apply_overrides(overrides: dict) -> None:
    from rag import config
    for key, val in overrides.items():
        if not hasattr(config, key):
            raise SystemExit(f"未知配置项：{key}（rag.config 上不存在）")
        setattr(config, key, val)
        print(f"  覆盖 config.{key} = {val!r}")


def _config_snapshot() -> dict:
    from rag import config
    return {k: getattr(config, k, None) for k in _TRACKED_CONFIG_KEYS}


def run(corpus_slug: str, qa_path: str, ks: list[int], limit: int | None) -> dict:
    from llama_index.core.schema import MetadataMode, QueryBundle

    from rag import config, corpus
    from rag.engine.bootstrap import build_query_engine, init_settings
    from rag.logging_utils import capture_pipeline_logs

    profile = corpus.load_profile(corpus_slug)
    chunks_path = os.path.join(config.CORPORA_ROOT, corpus_slug, "data", "chunks", "chunks.json")
    id2idx = load_chunk_index_map(chunks_path)      # gold_chunk_ids → chunk 下标（稳定）
    chunk_norms = load_chunk_norms(chunks_path)     # 反解被检索原始 chunk 用
    qa = load_qa(qa_path)
    if limit:
        qa = qa[:limit]

    print(f"语料《{profile.title}》（{corpus_slug}）：{len(qa)} 题，chunk 数 {len(id2idx)}，k={ks}")
    init_settings()
    with capture_pipeline_logs():  # 吞掉构建期噪声日志
        engine = build_query_engine(corpus_slug)

    per_q = []
    for row in qa:
        gold_idx = {id2idx[g] for g in row["gold_chunk_ids"] if g in id2idx}
        with capture_pipeline_logs():
            nodes = engine.retrieve(QueryBundle(row["question"]))
        retrieved = [
            (n.node.metadata.get("original_text") or n.node.get_content(MetadataMode.NONE),
             n.node.metadata)
            for n in nodes
        ]
        m = evaluate_retrieval(retrieved, gold_idx, chunk_norms, ks)
        m["id"] = row.get("id")
        m["type"] = row.get("type", "unknown")
        m["question"] = row["question"]
        m["n_retrieved"] = len(retrieved)
        per_q.append(m)
        rank = m["first_hit_rank"]
        rank_s = f"首命中@{rank}" if rank else "未命中"
        r_last = m["recall"][ks[-1]]
        print(f"  [{m['id']} {m['type']}] {rank_s}  recall@{ks[-1]}={r_last:.2f}  {row['question'][:24]}")

    agg = aggregate_retrieval(per_q, ks)
    return {
        "corpus": corpus_slug,
        "qa_path": qa_path,
        "n": len(qa),
        "ks": ks,
        "config": _config_snapshot(),
        "aggregate": agg,
        "per_question": per_q,
    }


def _print_report(result: dict) -> None:
    ks = result["ks"]
    agg = result["aggregate"]
    print("\n" + "=" * 60)
    print("检索评测汇总")
    print("=" * 60)

    def line(name: str, s: dict) -> None:
        if s.get("n", 0) == 0:
            return
        recalls = "  ".join(f"R@{k}={s['recall'][k]:.3f}" for k in ks)
        mfr = s["mean_first_hit_rank"]
        mfr_s = f"{mfr}" if mfr is not None else "-"
        print(f"  {name:<10} n={s['n']:<3} {recalls}  MRR={s['mrr']:.3f}  "
              f"命中率={s['hit_rate']:.2f}  平均首命中={mfr_s}")

    line("总体", agg["overall"])
    for t, s in agg["by_type"].items():
        line(t, s)


def main() -> int:
    p = argparse.ArgumentParser(description="RAG 检索评测")
    p.add_argument("-c", "--corpus", default=None, help="语料 slug（默认取激活语料）")
    p.add_argument("--qa", default=None, help="qa.jsonl 路径（默认 eval/<slug>/qa.jsonl）")
    p.add_argument("-k", "--ks", default="3,5,10", help="逗号分隔的 k 值（默认 3,5,10）")
    p.add_argument("--config-override", action="append", default=[], metavar="KEY=VALUE",
                   help="临时覆盖 rag.config（可重复），须在构建引擎前生效")
    p.add_argument("--save", default=None, metavar="PATH", help="把完整结果写成 JSON")
    p.add_argument("--limit", type=int, default=None, help="只跑前 N 题（冒烟）")
    args = p.parse_args()

    from rag import corpus
    slug = args.corpus or corpus.get_active_slug()
    qa_path = args.qa or f"eval/{slug}/qa.jsonl"
    if not os.path.exists(qa_path):
        print(f"找不到 qa 集：{qa_path}", file=sys.stderr)
        return 1
    ks = [int(x) for x in args.ks.split(",") if x.strip()]

    try:
        overrides = parse_overrides(args.config_override)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    if overrides:
        _apply_overrides(overrides)

    result = run(slug, qa_path, ks, args.limit)
    _print_report(result)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n已保存：{args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
