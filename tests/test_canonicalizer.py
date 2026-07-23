"""rag/graph/canonicalizer.py 快速规则单测 —— 确定性匹配（不依赖 LLM）。"""

from rag.graph.canonicalizer import Canonicalizer


def _c() -> Canonicalizer:
    # llm=None：快速规则未命中时 LLM 调用会抛异常并被捕获 → 返回 None
    return Canonicalizer(llm=None)


class TestFastRules:
    def test_exact_match(self):
        assert _c().canonicalize("丁元英", ["丁元英", "韩楚风"]) == "丁元英"

    def test_candidate_inside_known(self):
        assert _c().canonicalize("元英", ["韩楚风", "丁元英"]) == "丁元英"

    def test_known_inside_candidate_with_suffix(self):
        assert _c().canonicalize("丁元英先生", ["丁元英", "韩楚风"]) == "丁元英"

    def test_containing_prefers_shortest_regardless_of_order(self):
        # 多个已知名都包含候选：取最短者，且与列表顺序无关
        known_a = ["格律诗音响公司", "格律诗"]
        known_b = ["格律诗", "格律诗音响公司"]
        assert _c().canonicalize("律诗", known_a) == "格律诗"
        assert _c().canonicalize("律诗", known_b) == "格律诗"

    def test_contained_prefers_longest_regardless_of_order(self):
        # 候选包含多个已知名：取最长者（最具体），与列表顺序无关
        known_a = ["元英", "丁元英"]
        known_b = ["丁元英", "元英"]
        assert _c().canonicalize("丁元英哥", known_a) == "丁元英"
        assert _c().canonicalize("丁元英哥", known_b) == "丁元英"

    def test_no_match_and_no_llm_returns_none(self):
        assert _c().canonicalize("欧阳雪", ["丁元英", "韩楚风"]) is None

    def test_local_map_cached(self):
        c = _c()
        assert c.canonicalize("元英", ["丁元英"]) == "丁元英"
        # 二次调用走本地缓存（即使 known_names 为其他内容）
        assert c.canonicalize("元英", ["韩楚风"]) == "丁元英"
