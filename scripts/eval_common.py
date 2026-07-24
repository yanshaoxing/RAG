"""评测脚手架的纯函数（无网络、无 LLM、无重型依赖）——供 eval_retrieval / eval_answer 复用，
也供 tests/test_eval_common.py 离线覆盖。

只依赖标准库：指标计算、qa.jsonl / chunks.json 读取、--config-override 解析都放这里，
凡是需要真实检索/LLM 的部分留在两个脚本里。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence


# ============================================================
# 数据读取
# ============================================================
def load_qa(path: str) -> list[dict]:
    """读取 qa.jsonl，返回每行一个 dict。空行跳过。"""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_chunk_index_map(chunks_json_path: str) -> dict[str, int]:
    """从 chunks.json 建 node_id → 顺序下标 的映射（gold_chunk_ids 引用的就是 node_id）。"""
    chunks = json.load(open(chunks_json_path, encoding="utf-8"))
    return {n["node_id"]: i for i, n in enumerate(chunks)}


def normalize_text(t: str) -> str:
    """去掉所有空白，得到用于内容比对的指纹（章节标签等前缀不影响包含判定）。"""
    return "".join((t or "").split())


def load_chunk_norms(chunks_json_path: str) -> list[str]:
    """按 chunks.json 顺序返回每个 chunk 的归一化正文，下标即 chunk 位置。"""
    chunks = json.load(open(chunks_json_path, encoding="utf-8"))
    return [normalize_text(n["text"]) for n in chunks]


# ============================================================
# 检索指标
# ============================================================
# 说明：检索返回节点的 node_id 会随重建变化，无法作为稳定锚点；chunk 正文才稳定。
# 因此 gold 用 chunks.json 的顺序下标标识（由 gold_chunk_ids 经 id2idx 换算），
# 被检索到的原始 chunk 则按正文内容反解回下标。

def resolve_chunk_index(node_text: str, chunk_norms: Sequence[str]) -> int | None:
    """把一个原始 chunk 节点的正文内容反解成 chunks.json 里的下标（内容包含判定）。

    相邻 chunk 会有重叠，故命中多个候选时取正文最长者（最具体、即完整覆盖的那块）。
    """
    nn = normalize_text(node_text)
    if not nn:
        return None
    best: int | None = None
    best_len = -1
    for i, cn in enumerate(chunk_norms):
        if cn and (cn in nn or nn in cn) and len(cn) > best_len:
            best, best_len = i, len(cn)
    return best


def covered_indices(node_text: str, metadata: Mapping, chunk_norms: Sequence[str]) -> set[int]:
    """一个被检索节点"覆盖"了哪些原始 chunk 下标。

    - 图谱上下文节点：不覆盖任何 chunk。
    - 摘要节点：覆盖它的 summary_chunk_range 闭区间 [lo, hi]（摘要命中理应给分，
      这正是摘要树作为检索目标的设计意图）。
    - 原始 chunk 节点：按正文内容反解回下标（node_id 不可靠，见上方说明）。
    """
    if metadata.get("is_graph_context"):
        return set()
    if metadata.get("is_summary") and metadata.get("summary_chunk_range"):
        rng = metadata["summary_chunk_range"]
        if len(rng) == 2:
            lo, hi = int(rng[0]), int(rng[1])
            return set(range(lo, hi + 1))
        if len(rng) == 1:  # L1 叶子摘要只覆盖单块
            return {int(rng[0])}
    idx = resolve_chunk_index(node_text, chunk_norms)
    return {idx} if idx is not None else set()


def evaluate_retrieval(
    retrieved: Sequence[tuple[str, Mapping]],
    gold_indices: Iterable[int],
    chunk_norms: Sequence[str],
    ks: Sequence[int],
) -> dict:
    """对单个问题算检索指标。

    Args:
        retrieved: 按排名先后排列的 (node_text, metadata) 列表。
        gold_indices: 标准 chunk 的下标集合（调用方由 gold_chunk_ids 经 id2idx 换算）。
        chunk_norms: chunks.json 各块的归一化正文（下标对齐）。
        ks: 要算 recall 的若干 k 值。

    Returns:
        {"n_gold", "recall": {k: v}, "rr", "first_hit_rank"（1-based，无命中为 None）}。
    """
    gold_idx = set(gold_indices)
    n_gold = len(gold_idx)

    first_hit_rank: int | None = None
    # 逐排名累积已覆盖的 chunk 下标，便于一次遍历算出所有 k 的 recall
    cum_covered: set[int] = set()
    covered_at: dict[int, set[int]] = {}  # rank(1-based) → 累计覆盖集快照
    for rank, (node_text, meta) in enumerate(retrieved, start=1):
        cov = covered_indices(node_text, meta or {}, chunk_norms)
        if first_hit_rank is None and (cov & gold_idx):
            first_hit_rank = rank
        cum_covered |= cov
        covered_at[rank] = set(cum_covered)

    def recall_at(k: int) -> float:
        if n_gold == 0:
            return 0.0
        # 取 rank<=k 的累计覆盖；k 超过实际返回条数时用最后一个快照
        snap = covered_at.get(min(k, len(retrieved)), set()) if retrieved else set()
        return len(snap & gold_idx) / n_gold

    return {
        "n_gold": n_gold,
        "recall": {k: recall_at(k) for k in ks},
        "rr": (1.0 / first_hit_rank) if first_hit_rank else 0.0,
        "first_hit_rank": first_hit_rank,
    }


def aggregate_retrieval(per_q: Sequence[Mapping], ks: Sequence[int]) -> dict:
    """把若干问题的单题指标按 type 分组 + 汇总。

    Args:
        per_q: 每题一个 dict，含 "type" 与 evaluate_retrieval 的返回字段。

    Returns:
        {"overall": {...}, "by_type": {type: {...}}}，每组含
        n、mean recall@k、mrr、命中率 hit_rate、命中样本的平均首命中排名 mean_first_hit_rank。
    """
    groups: dict[str, list[Mapping]] = {"__all__": []}
    for r in per_q:
        groups["__all__"].append(r)
        groups.setdefault(r.get("type", "unknown"), []).append(r)

    def summarize(rows: Sequence[Mapping]) -> dict:
        n = len(rows)
        if n == 0:
            return {"n": 0}
        hit_ranks = [r["first_hit_rank"] for r in rows if r["first_hit_rank"]]
        return {
            "n": n,
            "recall": {k: round(sum(r["recall"][k] for r in rows) / n, 4) for k in ks},
            "mrr": round(sum(r["rr"] for r in rows) / n, 4),
            "hit_rate": round(len(hit_ranks) / n, 4),
            "mean_first_hit_rank": round(sum(hit_ranks) / len(hit_ranks), 2) if hit_ranks else None,
        }

    return {
        "overall": summarize(groups.pop("__all__")),
        "by_type": {t: summarize(rows) for t, rows in sorted(groups.items())},
    }


# ============================================================
# 回答评测汇总
# ============================================================
def aggregate_answer(per_q: Sequence[Mapping]) -> dict:
    """把逐题 judge 分数按 type 分组 + 汇总（faithfulness/citation/completeness + 命中率）。"""
    dims = ("faithfulness", "citation", "completeness")
    groups: dict[str, list[Mapping]] = {"__all__": []}
    for r in per_q:
        groups["__all__"].append(r)
        groups.setdefault(r.get("type", "unknown"), []).append(r)

    def summarize(rows: Sequence[Mapping]) -> dict:
        n = len(rows)
        if n == 0:
            return {"n": 0}
        out = {"n": n}
        for d in dims:
            out[d] = round(sum(r["scores"][d] for r in rows) / n, 3)
        tot = sum(r["scores"].get("num_points", 0) for r in rows)
        hit = sum(r["scores"].get("hit_points", 0) for r in rows)
        out["point_hit_rate"] = round(hit / tot, 4) if tot else None
        return out

    return {
        "overall": summarize(groups.pop("__all__")),
        "by_type": {t: summarize(rows) for t, rows in sorted(groups.items())},
    }


def format_gold_points(points: Iterable[str]) -> str:
    """把标准答案要点列表渲染成带序号的多行文本，供 judge 提示词填充。"""
    return "\n".join(f"{i}. {p}" for i, p in enumerate(points, start=1))


# ============================================================
# --config-override KEY=VALUE 解析
# ============================================================
def coerce_scalar(s: str):
    """把命令行字符串按 bool → int → float → str 顺序尽量转成对应标量。"""
    low = s.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def parse_overrides(items: Iterable[str]) -> dict[str, object]:
    """解析 --config-override 传入的 'KEY=VALUE' 列表为 {KEY: 已转型值}。

    仅解析，不应用；由调用方决定往 config 上 setattr（须在构建引擎之前）。
    """
    out: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"config-override 需形如 KEY=VALUE，收到：{item!r}")
        key, _, val = item.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"config-override 缺少键名：{item!r}")
        out[key] = coerce_scalar(val)
    return out
