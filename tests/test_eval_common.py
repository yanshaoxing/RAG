"""scripts/eval_common.py 纯函数离线单测（不触网、不加载重型依赖）。"""

import importlib.util
import json
import os

import pytest

# eval_common 只依赖标准库，用 importlib 从文件加载，避免把 scripts 变成包
_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "eval_common.py")
_spec = importlib.util.spec_from_file_location("eval_common", _PATH)
ec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ec)


# 10 个"chunk 正文"：内容各不相同、足够长且互为唯一子串
CHUNK_NORMS = [ec.normalize_text(f"这是第{i}块的正文内容编号{i}" * 3) for i in range(10)]


def _chunk_text(i: int) -> str:
    """把归一化正文当成原始节点文本（内容比对不依赖空白）。"""
    return CHUNK_NORMS[i]


# ---------------- resolve_chunk_index / covered_indices ----------------
def test_resolve_by_content_exact():
    assert ec.resolve_chunk_index(_chunk_text(3), CHUNK_NORMS) == 3


def test_resolve_by_content_with_section_prefix():
    # 真实检索节点可能带"第一章 "前缀，包含判定仍应命中
    assert ec.resolve_chunk_index("第一章 " + _chunk_text(5), CHUNK_NORMS) == 5


def test_resolve_prefers_longest_on_overlap():
    # 两块，短块正文是长块的子串（模拟相邻 chunk 重叠），应取更具体的长块
    norms = ["甲乙丙", "甲乙丙丁戊己庚"]
    assert ec.resolve_chunk_index("甲乙丙丁戊己庚", norms) == 1


def test_resolve_miss_returns_none():
    assert ec.resolve_chunk_index("完全无关的文本", CHUNK_NORMS) is None
    assert ec.resolve_chunk_index("", CHUNK_NORMS) is None


def test_covered_raw_chunk():
    assert ec.covered_indices(_chunk_text(1), {}, CHUNK_NORMS) == {1}


def test_covered_summary_range():
    # 摘要节点覆盖闭区间 [1, 3]
    meta = {"is_summary": True, "summary_chunk_range": [1, 3]}
    assert ec.covered_indices("摘要文本", meta, CHUNK_NORMS) == {1, 2, 3}


def test_covered_summary_leaf_single():
    meta = {"is_summary": True, "summary_chunk_range": [4]}
    assert ec.covered_indices("摘要文本", meta, CHUNK_NORMS) == {4}


def test_covered_graph_or_unknown_is_empty():
    assert ec.covered_indices("图谱三元组", {"is_graph_context": True}, CHUNK_NORMS) == set()
    assert ec.covered_indices("无关文本", {}, CHUNK_NORMS) == set()


# ---------------- evaluate_retrieval ----------------
def _ret(*idxs):
    """构造 (node_text, meta) 序列，全部当作原始 chunk（文本取对应块正文）。"""
    return [(_chunk_text(i), {}) for i in idxs]


def test_evaluate_first_hit_and_mrr():
    retrieved = _ret(5, 1, 3)  # gold=块1 在第 2 位
    m = ec.evaluate_retrieval(retrieved, {1}, CHUNK_NORMS, ks=[1, 3])
    assert m["first_hit_rank"] == 2
    assert m["rr"] == 0.5
    assert m["recall"][1] == 0.0   # top-1 没命中
    assert m["recall"][3] == 1.0   # top-3 命中唯一 gold


def test_evaluate_partial_recall():
    retrieved = _ret(1, 9)  # 两 gold 只命中一个
    m = ec.evaluate_retrieval(retrieved, {1, 2}, CHUNK_NORMS, ks=[5])
    assert m["n_gold"] == 2
    assert m["recall"][5] == 0.5


def test_evaluate_no_hit():
    m = ec.evaluate_retrieval(_ret(8, 9), {1}, CHUNK_NORMS, ks=[3])
    assert m["first_hit_rank"] is None
    assert m["rr"] == 0.0
    assert m["recall"][3] == 0.0


def test_evaluate_summary_credits_gold():
    # 只检索到一个覆盖 [0,4] 的摘要节点，gold=块2 应算命中
    retrieved = [("摘要文本", {"is_summary": True, "summary_chunk_range": [0, 4]})]
    m = ec.evaluate_retrieval(retrieved, {2}, CHUNK_NORMS, ks=[1])
    assert m["first_hit_rank"] == 1
    assert m["recall"][1] == 1.0


def test_evaluate_k_exceeds_retrieved():
    m = ec.evaluate_retrieval(_ret(1), {1}, CHUNK_NORMS, ks=[3, 10])
    # 只返回 1 条，k=3/10 都用最后一个快照，命中率仍 1.0
    assert m["recall"][3] == 1.0
    assert m["recall"][10] == 1.0


def test_evaluate_empty_gold():
    m = ec.evaluate_retrieval(_ret(1), set(), CHUNK_NORMS, ks=[3])
    assert m["n_gold"] == 0
    assert m["recall"][3] == 0.0  # n_gold=0 时约定为 0


# ---------------- aggregate_retrieval ----------------
def test_aggregate_groups_by_type():
    per_q = [
        {"type": "fact", "recall": {3: 1.0}, "rr": 1.0, "first_hit_rank": 1},
        {"type": "fact", "recall": {3: 0.0}, "rr": 0.0, "first_hit_rank": None},
        {"type": "macro", "recall": {3: 0.5}, "rr": 0.5, "first_hit_rank": 2},
    ]
    agg = ec.aggregate_retrieval(per_q, ks=[3])
    assert agg["overall"]["n"] == 3
    assert agg["overall"]["recall"][3] == pytest.approx((1.0 + 0.0 + 0.5) / 3, abs=1e-4)
    fact = agg["by_type"]["fact"]
    assert fact["n"] == 2
    assert fact["hit_rate"] == 0.5
    assert fact["mean_first_hit_rank"] == 1.0  # 只有命中样本参与
    macro = agg["by_type"]["macro"]
    assert macro["mrr"] == 0.5


def test_aggregate_all_miss_mean_rank_none():
    per_q = [{"type": "fact", "recall": {3: 0.0}, "rr": 0.0, "first_hit_rank": None}]
    agg = ec.aggregate_retrieval(per_q, ks=[3])
    assert agg["overall"]["mean_first_hit_rank"] is None
    assert agg["overall"]["hit_rate"] == 0.0


# ---------------- aggregate_answer ----------------
def test_aggregate_answer():
    per_q = [
        {"type": "fact", "scores": {"faithfulness": 5, "citation": 4, "completeness": 5,
                                    "hit_points": 2, "num_points": 2}},
        {"type": "fact", "scores": {"faithfulness": 3, "citation": 2, "completeness": 1,
                                    "hit_points": 0, "num_points": 3}},
    ]
    agg = ec.aggregate_answer(per_q)
    o = agg["overall"]
    assert o["n"] == 2
    assert o["faithfulness"] == 4.0
    assert o["point_hit_rate"] == pytest.approx(2 / 5, abs=1e-4)


# ---------------- overrides ----------------
def test_coerce_scalar_types():
    assert ec.coerce_scalar("true") is True
    assert ec.coerce_scalar("False") is False
    assert ec.coerce_scalar("none") is None
    assert ec.coerce_scalar("42") == 42
    assert ec.coerce_scalar("3.5") == 3.5
    assert ec.coerce_scalar("qwen-flash") == "qwen-flash"


def test_parse_overrides_ok():
    out = ec.parse_overrides(["RERANK_ENABLED=false", "RRF_K=30", "GAP_THRESHOLD=0.5"])
    assert out == {"RERANK_ENABLED": False, "RRF_K": 30, "GAP_THRESHOLD": 0.5}


def test_parse_overrides_bad():
    with pytest.raises(ValueError):
        ec.parse_overrides(["NO_EQUALS_SIGN"])
    with pytest.raises(ValueError):
        ec.parse_overrides(["=novalue"])


# ---------------- io helpers ----------------
def test_load_qa_and_chunk_map(tmp_path):
    qa = tmp_path / "qa.jsonl"
    qa.write_text('{"id":"q1","question":"问","gold_chunk_ids":["a"],"type":"fact"}\n\n',
                  encoding="utf-8")
    rows = ec.load_qa(str(qa))
    assert len(rows) == 1 and rows[0]["id"] == "q1"

    chunks = tmp_path / "chunks.json"
    chunks.write_text(json.dumps([{"node_id": "a"}, {"node_id": "b"}]), encoding="utf-8")
    m = ec.load_chunk_index_map(str(chunks))
    assert m == {"a": 0, "b": 1}


def test_format_gold_points():
    s = ec.format_gold_points(["甲", "乙"])
    assert s == "1. 甲\n2. 乙"
