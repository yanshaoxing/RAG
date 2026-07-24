"""rag/ingestion/structure_detector.py 单测 —— 采样、正则校验、LLM 检测与档案写回（mock LLM，离线）。"""

import json
from types import SimpleNamespace

import pytest

from rag import config, corpus
from rag.ingestion import structure_detector as sd
from rag.ingestion.preprocessor import load_documents


class _FakeLLM:
    """离线假 LLM：complete 固定返回构造时给定的文本。"""

    def __init__(self, text: str, fail: bool = False):
        self._text = text
        self._fail = fail
        self.calls = 0

    def complete(self, prompt: str):
        self.calls += 1
        if self._fail:
            raise RuntimeError("模拟 LLM 故障")
        return SimpleNamespace(text=self._text)


def _arabic_book(n_chapters: int = 5) -> str:
    """内置正则（中文数字）无法识别的阿拉伯数字回目文本。"""
    return "\n".join(
        f"第{i}回 测试回目{i}\n这是第{i}回的正文内容，讲述了一段故事。"
        for i in range(1, n_chapters + 1)
    )


# ======================== 采样 ========================

class TestSampleText:
    def test_short_text_head_only(self):
        text = "短文本" * 100  # 300 字 < HEAD_CHARS，切片全部落在开头样本内
        sample = sd.sample_text(text)
        assert sample.startswith("【样本：开头】")
        assert "25%" not in sample

    def test_long_text_has_slices(self):
        text = "长" * (config.STRUCTURE_SAMPLE_HEAD_CHARS * 10)
        sample = sd.sample_text(text)
        for marker in ("【样本：开头】", "【样本：全文25%处】",
                       "【样本：全文50%处】", "【样本：全文75%处】"):
            assert marker in sample


# ======================== 章节正则校验 ========================

class TestValidateChapterPattern:
    def test_valid_pattern(self):
        assert sd.validate_chapter_pattern(r"第\d+回", _arabic_book())

    def test_empty_pattern(self):
        assert not sd.validate_chapter_pattern("", _arabic_book())

    def test_uncompilable_pattern(self):
        assert not sd.validate_chapter_pattern(r"第[", _arabic_book())

    def test_empty_match_pattern(self):
        # 可匹配空串的正则会把切分变成逐字符爆炸，必须拒绝
        assert not sd.validate_chapter_pattern(r"\d*", _arabic_book())

    def test_too_few_sections(self):
        text = "第1回 起\n正文。"  # 1 < STRUCTURE_MIN_SECTIONS
        assert not sd.validate_chapter_pattern(r"第\d+回", text)

    def test_overlong_title_rejected(self):
        # 正则过宽：匹配到正文行开头（"这"），命中行超长
        text = "\n".join("这是一段没有真正章节标题的超长正文行，" * 5 for _ in range(5))
        assert not sd.validate_chapter_pattern(r"这", text)


class TestValidateSubsectionPattern:
    def test_valid(self):
        assert sd.validate_subsection_pattern(r"^§\d+\s*$")

    def test_empty_and_bad(self):
        assert not sd.validate_subsection_pattern("")
        assert not sd.validate_subsection_pattern(r"[")
        assert not sd.validate_subsection_pattern(r"\d*")  # 匹配空串


# ======================== LLM 检测 ========================

class TestDetectStructure:
    def test_valid_llm_output(self):
        text = _arabic_book()
        llm = _FakeLLM(json.dumps(
            {"chapter_pattern": r"第\d+回", "subsection_pattern": None}))
        result = sd.detect_structure(text, llm)
        assert result == {"chapter_pattern": r"第\d+回"}

    def test_with_subsection(self):
        text = _arabic_book()
        llm = _FakeLLM(json.dumps(
            {"chapter_pattern": r"第\d+回", "subsection_pattern": r"^§\d+$"}))
        result = sd.detect_structure(text, llm)
        assert result["subsection_pattern"] == r"^§\d+$"

    def test_invalid_pattern_dropped(self):
        # LLM 给出的正则通不过全文校验 → 不采用
        llm = _FakeLLM(json.dumps(
            {"chapter_pattern": r"Chapter \d+", "subsection_pattern": None}))
        assert sd.detect_structure(_arabic_book(), llm) == {}

    def test_garbage_output(self):
        assert sd.detect_structure(_arabic_book(), _FakeLLM("我不知道")) == {}

    def test_llm_failure(self):
        assert sd.detect_structure(_arabic_book(), _FakeLLM("", fail=True)) == {}


# ======================== 档案写回 + load_documents 端到端触发 ========================

def _make_corpus_with_raw(root, slug, title, raw_text):
    corpus_dir = root / slug
    (corpus_dir / "raw").mkdir(parents=True)
    (corpus_dir / "corpus.json").write_text(
        json.dumps({"title": title, "context": f"《{title}》背景"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (corpus_dir / "raw" / "book.txt").write_text(raw_text, encoding="utf-8")
    return corpus_dir


@pytest.fixture
def _tmp_corpus(tmp_path, monkeypatch):
    """临时语料（阿拉伯数字回目，内置正则零命中）+ 激活切换，测试后还原。"""
    _make_corpus_with_raw(tmp_path, "StructBook", "结构书", _arabic_book())
    monkeypatch.setattr(config, "CORPORA_ROOT", str(tmp_path))
    original = corpus.get_active_slug()
    corpus.set_active_corpus("StructBook")
    yield tmp_path
    corpus._active_slug = original


def test_detect_and_persist_writes_profile(_tmp_corpus, monkeypatch):
    import rag.llm.factory as factory
    llm = _FakeLLM(json.dumps({"chapter_pattern": r"第\d+回", "subsection_pattern": None}))
    monkeypatch.setattr(factory, "create_summary_llm", lambda: llm)

    result = sd.detect_and_persist("StructBook", _arabic_book())
    assert result["chapter_pattern"] == r"第\d+回"
    # 写回档案 + 缓存失效后重新加载能读到
    data = json.loads((_tmp_corpus / "StructBook" / "corpus.json").read_text(encoding="utf-8"))
    assert data["chapter_pattern"] == r"第\d+回"
    assert corpus.get_active_profile().chapter_pattern == r"第\d+回"


def test_load_documents_triggers_detection(_tmp_corpus, monkeypatch):
    import rag.llm.factory as factory
    llm = _FakeLLM(json.dumps({"chapter_pattern": r"第\d+回", "subsection_pattern": None}))
    monkeypatch.setattr(factory, "create_summary_llm", lambda: llm)

    docs = load_documents()
    assert llm.calls == 1
    assert [d.metadata["section"] for d in docs[:2]] == ["第1回 测试回目1", "第2回 测试回目2"]
    # 检测结果已持久化 → 再次加载不再调 LLM
    docs2 = load_documents()
    assert llm.calls == 1
    assert len(docs2) == len(docs)


def test_load_documents_detection_failure_falls_back(_tmp_corpus, monkeypatch):
    import rag.llm.factory as factory
    monkeypatch.setattr(factory, "create_summary_llm", lambda: _FakeLLM("", fail=True))

    docs = load_documents()  # 不中断，整书落入"概述"单章
    assert docs
    assert all(d.metadata["section"] == "概述" for d in docs)


def test_load_documents_builtin_format_skips_llm(tmp_path, monkeypatch):
    """内置格式可识别的文本绝不触发 LLM（离线保证）。"""
    import rag.llm.factory as factory

    def _boom():
        raise AssertionError("内置格式不应触发 LLM 结构检测")

    monkeypatch.setattr(factory, "create_summary_llm", _boom)
    text = "一、开篇\n甲正文。\n二、发展\n乙正文。"
    _make_corpus_with_raw(tmp_path, "DunBook", "顿号书", text)
    monkeypatch.setattr(config, "CORPORA_ROOT", str(tmp_path))
    original = corpus.get_active_slug()
    try:
        corpus.set_active_corpus("DunBook")
        docs = load_documents()
        assert [d.metadata["section"] for d in docs] == ["一、开篇", "二、发展"]
    finally:
        corpus._active_slug = original


def test_structure_detect_disabled(_tmp_corpus, monkeypatch):
    import rag.llm.factory as factory
    monkeypatch.setattr(factory, "create_summary_llm",
                        lambda: (_ for _ in ()).throw(AssertionError("开关关闭不应调 LLM")))
    monkeypatch.setattr(config, "STRUCTURE_DETECT_ENABLED", False)
    docs = load_documents()
    assert all(d.metadata["section"] == "概述" for d in docs)
