"""rag/summarization/summary_tree.py 纯函数单测 —— 批量摘要解析、中文数字、章节排序、父级兜底。"""

import rag.summarization.summary_tree as summary_tree
from rag import config
from rag.summarization.summary_tree import (
    SummaryGroup,
    SummaryNode,
    _cn_numeral_to_int,
    _fallback_parent_summary,
    _generate_subsection_summaries,
    _parse_batch_leaf_response,
    _section_order,
)


class TestParseBatchLeafResponse:
    def test_normal_response(self):
        resp = "[1] 第一条摘要\n[2] 第二条摘要\n[3] 第三条摘要"
        assert _parse_batch_leaf_response(resp, 3) == ["第一条摘要", "第二条摘要", "第三条摘要"]

    def test_missing_index_left_empty(self):
        resp = "[1] 第一条摘要\n[3] 第三条摘要"
        out = _parse_batch_leaf_response(resp, 3)
        assert out == ["第一条摘要", "", "第三条摘要"]

    def test_multiline_content_flattened(self):
        resp = "[1] 第一行\n续行内容\n[2] 第二条"
        out = _parse_batch_leaf_response(resp, 2)
        assert out[0] == "第一行 续行内容"

    def test_out_of_range_index_ignored(self):
        resp = "[1] 有效\n[9] 越界"
        out = _parse_batch_leaf_response(resp, 2)
        assert out == ["有效", ""]

    def test_garbage_response(self):
        assert _parse_batch_leaf_response("完全没有编号格式", 2) == ["", ""]


class TestCnNumeralToInt:
    def test_basic_digits(self):
        assert _cn_numeral_to_int("一") == 1
        assert _cn_numeral_to_int("九") == 9

    def test_tens(self):
        assert _cn_numeral_to_int("十") == 10
        assert _cn_numeral_to_int("十二") == 12
        assert _cn_numeral_to_int("二十三") == 23

    def test_hundreds(self):
        assert _cn_numeral_to_int("一百零五") == 105
        assert _cn_numeral_to_int("一百二十") == 120

    def test_liang(self):
        assert _cn_numeral_to_int("两") == 2

    def test_invalid_returns_zero(self):
        assert _cn_numeral_to_int("abc") == 0
        assert _cn_numeral_to_int("") == 0


def _summary_node(section: str, chunk_start: int = 0) -> SummaryNode:
    return SummaryNode(text="摘要", node_id=f"L3_{section}", level=3, section=section,
                       chunk_range=(chunk_start, chunk_start + 1))


class TestSectionOrder:
    def test_hui_chinese_numerals(self):
        # "第十二回" 这类中文数字必须正确排序（曾全部得 0 导致产物不确定）
        n2 = _summary_node("第二回 承")
        n12 = _summary_node("第十二回 转")
        assert _section_order(n2)[0] == 2
        assert _section_order(n12)[0] == 12

    def test_dun_style(self):
        assert _section_order(_summary_node("三、发展"))[0] == 3

    def test_arabic_fallback(self):
        assert _section_order(_summary_node("第3章 起航"))[0] == 3

    def test_no_number_falls_back_to_chunk_start(self):
        # 无法解析序号时以起始 chunk 兜底，保证排序确定
        n = _summary_node("概述", chunk_start=7)
        assert _section_order(n) == (0, 7)

    def test_sort_is_deterministic(self):
        nodes = [_summary_node("第十二回"), _summary_node("第二回"), _summary_node("第一回")]
        ordered = sorted(nodes, key=_section_order)
        assert [n.section for n in ordered] == ["第一回", "第二回", "第十二回"]


# ---------- 父级摘要兜底（新缺陷 7） ----------

def _leaf(nid: str, chunk_idx: int, section: str = "第一章", subsection: str = "1") -> SummaryNode:
    return SummaryNode(
        text=f"叶子摘要{nid}", node_id=f"summary_leaf_{nid}", level=1,
        child_ids=[nid], file_name="book.txt", section=section,
        subsection=subsection, chunk_range=(chunk_idx, chunk_idx),
        original_text=f"原文内容{nid}",
    )


class TestFallbackParentSummary:
    def test_fields_aggregated(self):
        group = SummaryGroup(children=[_leaf("a", 3), _leaf("b", 5)],
                             file_name="book.txt", section="第一章", subsection="1")
        node = _fallback_parent_summary(group, level=2)
        assert node.is_fallback
        assert node.level == 2
        assert node.chunk_range == (3, 5)
        assert node.child_ids == ["a", "b"]
        assert node.text.startswith("原文内容a")
        assert node.node_id == "summary_L2_3_5"


class TestL2ExceptionFallback:
    """L2 worker 整体异常时该小节不能在摘要树中缺失，应降级为原文截断。"""

    def _run(self, monkeypatch, concurrency: int) -> list[SummaryNode]:
        def _boom(*args, **kwargs):
            raise RuntimeError("worker 崩溃")

        monkeypatch.setattr(summary_tree, "_generate_parent_summary", _boom)
        monkeypatch.setattr(config, "SUMMARY_MAX_CONCURRENCY", concurrency)
        leaves = [_leaf("a", 0, subsection="1"), _leaf("b", 1, subsection="2")]
        return _generate_subsection_summaries(leaves)

    def test_parallel_path_falls_back(self, monkeypatch):
        out = self._run(monkeypatch, concurrency=2)
        assert len(out) == 2                      # 两个小节都有节点
        assert all(n.is_fallback for n in out)

    def test_serial_path_falls_back(self, monkeypatch):
        out = self._run(monkeypatch, concurrency=1)
        assert len(out) == 2
        assert all(n.is_fallback for n in out)
