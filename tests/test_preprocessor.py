"""rag/ingestion/preprocessor.py 单测 —— 章节切分、小节切分（锁定缺陷 #5）、语义边界分块。"""

import pytest

from rag.ingestion.preprocessor import (
    HierarchicalChunker,
    _detect_l1_format,
    _find_semantic_boundary_right,
    _split_body_by_subsections,
    split_by_section,
)


class TestDetectL1Format:
    def test_dun_style(self):
        text = "一、开篇\n正文。\n二、发展\n正文。\n三、高潮\n正文。"
        assert _detect_l1_format(text) == "dun"

    def test_hui_style(self):
        text = "第一回 灵根育孕\n正文。\n第二回 悟彻菩提\n正文。\n第三回 四海千山\n正文。"
        assert _detect_l1_format(text) == "hui"


class TestSplitBySection:
    def test_dun_sections(self):
        text = "一、开篇\n第一章正文。\n二、发展\n第二章正文。"
        sections = split_by_section(text)
        assert [s["section"] for s in sections] == ["一、开篇", "二、发展"]
        assert sections[0]["content"] == "第一章正文。"
        assert sections[1]["content"] == "第二章正文。"

    def test_hui_sections(self):
        text = "第一回 起\n甲正文。\n第二回 承\n乙正文。\n第十二回 转\n丙正文。"
        sections = split_by_section(text)
        assert [s["section"] for s in sections] == ["第一回 起", "第二回 承", "第十二回 转"]
        assert sections[2]["content"] == "丙正文。"

    def test_preface_becomes_overview(self):
        # 第一个章节标题之前的文本归入"概述"段
        text = "这是序言内容。\n一、开篇\n正文。"
        sections = split_by_section(text)
        assert sections[0]["section"] == "概述"
        assert "序言内容" in sections[0]["content"]

    def test_no_content_lost(self):
        # 切分后正文内容不丢失（标题行进入 section 元数据）
        text = "一、开篇\n甲乙丙丁。\n二、发展\n戊己庚辛。"
        sections = split_by_section(text)
        merged = "".join(s["content"] for s in sections)
        for fragment in ("甲乙丙丁。", "戊己庚辛。"):
            assert fragment in merged


class TestSplitBodyBySubsections:
    def test_no_markers(self):
        body = "没有小节标记的正文。"
        subs = _split_body_by_subsections(body)
        assert subs == [{"subsection": "", "content": body}]

    def test_basic_markers(self):
        body = "　　1\n第一节内容。\n　　2\n第二节内容。"
        subs = _split_body_by_subsections(body)
        assert [s["subsection"] for s in subs] == ["1", "2"]
        assert subs[0]["content"] == "第一节内容。"
        assert subs[1]["content"] == "第二节内容。"

    def test_preamble_preserved(self):
        # 缺陷 #5 回归：章节标题与第一个小节标记之间的正文必须保留
        body = "章首正文，不可丢弃。\n　　1\n第一节内容。"
        subs = _split_body_by_subsections(body)
        assert subs[0] == {"subsection": "", "content": "章首正文，不可丢弃。"}
        assert subs[1]["subsection"] == "1"

    def test_all_text_covered(self):
        body = "章首。\n　　1\n甲。\n　　2\n乙。"
        subs = _split_body_by_subsections(body)
        merged = "".join(s["content"] for s in subs)
        for fragment in ("章首。", "甲。", "乙。"):
            assert fragment in merged


class TestFindSemanticBoundaryRight:
    def test_single_punct_before_hanzi(self):
        text = "前句。后句"
        # 从 0 扫描：句号在 idx 2，后跟汉字 → 切在句号右边
        assert _find_semantic_boundary_right(text, 0) == 3

    def test_double_punct_not_split(self):
        # 。”+汉字 应切在双标点之后（不能从中间切开）；右引号用显式转义避免引号字符混淆
        text = "他说完。”然后走了"
        pos = _find_semantic_boundary_right(text, 0)
        assert text[:pos].endswith("。”")

    def test_ellipsis(self):
        text = "话没说完……接着说"
        pos = _find_semantic_boundary_right(text, 0)
        assert text[:pos].endswith("……")

    def test_no_boundary_returns_len(self):
        text = "没有任何句末标点"
        assert _find_semantic_boundary_right(text, 0) == len(text)


def _make_long_text(n_sentences: int) -> str:
    return "".join(f"这是第{i}句测试文本，用来验证分块。" for i in range(n_sentences))


class TestSlidingWindow:
    def test_short_text_single_chunk(self):
        chunks = HierarchicalChunker._sliding_window("短文本。", 1024, 102)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "短文本。"

    def test_empty_text(self):
        assert HierarchicalChunker._sliding_window("   ", 1024, 102) == []

    @pytest.mark.parametrize("n_sentences", [80, 200, 400])
    def test_full_coverage_and_size(self, n_sentences):
        text = _make_long_text(n_sentences)
        chunk_size, overlap = 256, 25
        chunks = HierarchicalChunker._sliding_window(text, chunk_size, overlap)
        assert chunks, "长文本必须产出 chunk"
        # 相邻 chunk 无间隙（left ≤ 前一个 right），首尾覆盖全文
        assert chunks[0]["left"] == 0
        assert chunks[-1]["right"] == len(text)
        for prev, cur in zip(chunks, chunks[1:]):
            assert cur["left"] <= prev["right"], "相邻 chunk 之间不能有间隙"
        # chunk 大小不失控（语义边界最多右移一句）
        for c in chunks:
            assert len(c["text"]) <= chunk_size * 2
