"""rag/retrieval/query_rewriter.py 单测 —— 术语映射与关键词去重（不含 LLM 路径）。"""

from rag.retrieval.query_rewriter import QueryRewriter


def _rewriter_with_terms(term_map: dict[str, str], enabled: bool = False) -> QueryRewriter:
    r = QueryRewriter(enabled=enabled, llm=None)
    # 注入固定术语表（按最长匹配优先排序，与 _load_term_map 行为一致）
    r._term_map = dict(sorted(term_map.items(), key=lambda kv: len(kv[0]), reverse=True))
    return r


class TestApplyTermMap:
    def test_basic_replacement(self):
        r = _rewriter_with_terms({"小丹": "芮小丹"})
        mapped, log = r._apply_term_map("小丹是谁")
        assert mapped == "芮小丹是谁"
        assert log == ["小丹 → 芮小丹"]

    def test_longest_match_first(self):
        # "王庙村扶贫" 应优先于 "王庙村" 被整体替换
        r = _rewriter_with_terms({"王庙村": "王庙村（贫困村）", "王庙村扶贫": "王庙村扶贫神话"})
        mapped, _ = r._apply_term_map("王庙村扶贫的结局")
        assert mapped == "王庙村扶贫神话的结局"

    def test_no_match_passthrough(self):
        r = _rewriter_with_terms({"小丹": "芮小丹"})
        mapped, log = r._apply_term_map("丁元英是谁")
        assert mapped == "丁元英是谁"
        assert log == []

    def test_empty_term_map(self):
        r = _rewriter_with_terms({})
        assert r._apply_term_map("任意查询") == ("任意查询", [])


class TestRewriteDisabled:
    def test_term_map_runs_even_when_disabled(self):
        # 回归：REWRITE_ENABLED=False 时术语映射（纯字符串替换）仍须执行
        r = _rewriter_with_terms({"小丹": "芮小丹"}, enabled=False)
        nl, hyde, kw = r.rewrite("小丹是谁")
        assert nl == hyde == kw == "芮小丹是谁"


class TestDedupKw:
    def test_order_preserving_dedup(self):
        assert QueryRewriter._dedup_kw("甲 乙 甲 丙 乙") == "甲 乙 丙"

    def test_truncation(self):
        kw = " ".join(f"词{i}" for i in range(60))
        out = QueryRewriter._dedup_kw(kw, max_kw=50)
        assert len(out.split()) == 50
