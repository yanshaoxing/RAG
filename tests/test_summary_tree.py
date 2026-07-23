"""rag/summarization/summary_tree.py 纯函数单测 —— 批量摘要解析、中文数字、章节排序。"""

from rag.summarization.summary_tree import (
    SummaryNode,
    _cn_numeral_to_int,
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
