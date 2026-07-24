"""rag/corpus.py + 语料级规则合并 + 激活语料切换单测（离线，无 LLM）。"""

import json
import os

import pytest

from rag import config, corpus
from rag.graph import extractor


def _make_corpus(root, slug, title):
    corpus_dir = root / slug
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "corpus.json").write_text(
        json.dumps({"title": title, "context": f"《{title}》背景"}, ensure_ascii=False),
        encoding="utf-8",
    )
    return corpus_dir


def test_load_active_profile():
    profile = corpus.load_profile(config.ACTIVE_CORPUS)
    assert profile.slug == config.ACTIVE_CORPUS
    assert profile.title
    assert profile.context
    # 路径派生指向语料目录
    assert profile.raw_dir == os.path.join(profile.corpus_dir, "raw")
    assert profile.terminology_path.endswith("terminology.json")


def test_load_profile_missing_corpus():
    with pytest.raises(FileNotFoundError):
        corpus.load_profile("不存在的语料")


def test_load_profile_missing_required_fields(tmp_path, monkeypatch):
    slug = "empty_book"
    corpus_dir = tmp_path / slug
    corpus_dir.mkdir()
    (corpus_dir / "corpus.json").write_text(
        json.dumps({"title": "某书"}), encoding="utf-8"
    )
    monkeypatch.setattr(config, "CORPORA_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="context"):
        corpus.load_profile(slug)


def test_config_paths_derive_from_active_corpus():
    for path in (config.DATA_DIR, config.PERSIST_DIR, config.BM25_DIR,
                 config.CHUNKS_DIR, config.SUMMARY_TREE_DIR, config.GRAPH_DB_DIR,
                 config.EMBED_CHECKPOINT_DIR, config.TERM_MAP_PATH,
                 config.GRAPH_RULES_PATH):
        assert os.path.normpath(config.CORPUS_DIR) in os.path.normpath(path)


def test_graph_rules_merge_with_corpus_rules(tmp_path, monkeypatch):
    corpus_rules = {
        "known_male_characters": ["测试角色甲", "测试角色乙"],
        "min_entity_name_length": 3,
    }
    rules_path = tmp_path / "graph_rules.json"
    rules_path.write_text(json.dumps(corpus_rules, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(config, "GRAPH_RULES_PATH", str(rules_path))

    rules = extractor._load_rules()
    # 语料列表并入基础规则
    assert "测试角色甲" in rules["known_male_characters"]
    # 标量被语料值覆盖
    assert rules["min_entity_name_length"] == 3
    # 基础规则（语料无关黑名单）仍在
    assert "我" in rules["pronoun_blacklist"]
    assert "某人" in rules["generic_blacklist"]


def test_graph_rules_missing_corpus_file(monkeypatch):
    monkeypatch.setattr(config, "GRAPH_RULES_PATH", "/nonexistent/graph_rules.json")
    rules = extractor._load_rules()
    # 仅基础规则，known_male_characters 为空列表
    assert rules["known_male_characters"] == []
    assert "我" in rules["pronoun_blacklist"]


def test_active_corpus_graph_rules_loaded():
    # 默认语料的补充规则应真实并入（《遥远的救世主》角色名单）
    rules = extractor._load_rules()
    assert "丁元英" in rules["known_male_characters"]


# ======================== 激活语料切换（多书） ========================

def test_set_active_corpus_switches_config_paths(tmp_path, monkeypatch):
    _make_corpus(tmp_path, "BookA", "甲书")
    _make_corpus(tmp_path, "BookB", "乙书")
    monkeypatch.setattr(config, "CORPORA_ROOT", str(tmp_path))
    original = corpus.get_active_slug()
    try:
        corpus.set_active_corpus("BookA")
        assert corpus.get_active_profile().title == "甲书"
        # config 动态路径实时跟随激活语料
        assert os.path.normpath(config.CHUNKS_DIR) == os.path.normpath(
            str(tmp_path / "BookA" / "data" / "chunks"))
        corpus.set_active_corpus("BookB")
        assert corpus.get_active_profile().title == "乙书"
        assert "BookB" in config.PERSIST_DIR
        assert "BookB" in config.TERM_MAP_PATH
    finally:
        corpus._active_slug = original


def test_set_active_corpus_invalid_keeps_state(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CORPORA_ROOT", str(tmp_path))
    original = corpus.get_active_slug()
    with pytest.raises(FileNotFoundError):
        corpus.set_active_corpus("不存在")
    # 切换失败不改变激活语料
    assert corpus.get_active_slug() == original


def test_list_corpora(tmp_path, monkeypatch):
    _make_corpus(tmp_path, "BookB", "乙书")
    _make_corpus(tmp_path, "BookA", "甲书")
    (tmp_path / "not_a_corpus").mkdir()  # 无 corpus.json，应被跳过
    bad = _make_corpus(tmp_path, "BookC", "残书")
    (bad / "corpus.json").write_text("{}", encoding="utf-8")  # 缺必需字段，跳过
    monkeypatch.setattr(config, "CORPORA_ROOT", str(tmp_path))
    profiles = corpus.list_corpora()
    assert [p.slug for p in profiles] == ["BookA", "BookB"]
    assert [p.title for p in profiles] == ["甲书", "乙书"]


# ======================== 跨书 embedding 一致性诊断（P2-2） ========================

def _write_vector_marker(corpus_dir, **info):
    from rag.utils.files import mark_stage_done
    vector_dir = corpus_dir / "data" / "vector"
    vector_dir.mkdir(parents=True)
    mark_stage_done(str(vector_dir), **info)


def test_probe_vector_index_not_built(tmp_path, monkeypatch):
    _make_corpus(tmp_path, "BookA", "甲书")
    monkeypatch.setattr(config, "CORPORA_ROOT", str(tmp_path))
    status = corpus.probe_vector_index(corpus.load_profile("BookA"))
    assert status.built is False
    assert status.matches_current is True   # 未构建不算冲突


def test_probe_vector_index_matches_current(tmp_path, monkeypatch):
    cd = _make_corpus(tmp_path, "BookA", "甲书")
    monkeypatch.setattr(config, "CORPORA_ROOT", str(tmp_path))
    monkeypatch.setattr(config, "ACTIVE_EMBED_MODEL_NAME", "model-x")
    _write_vector_marker(cd, embed_model="model-x", vector_dim=1024)
    status = corpus.probe_vector_index(corpus.load_profile("BookA"))
    assert status.built is True
    assert status.embed_model == "model-x"
    assert status.vector_dim == 1024
    assert status.matches_current is True


def test_probe_vector_index_mismatch(tmp_path, monkeypatch):
    cd = _make_corpus(tmp_path, "BookA", "甲书")
    monkeypatch.setattr(config, "CORPORA_ROOT", str(tmp_path))
    monkeypatch.setattr(config, "ACTIVE_EMBED_MODEL_NAME", "model-new")
    _write_vector_marker(cd, embed_model="model-old", vector_dim=4096)
    status = corpus.probe_vector_index(corpus.load_profile("BookA"))
    assert status.built is True
    assert status.matches_current is False   # 换模型 → 冲突，list 会提示重建


def test_probe_vector_index_legacy_marker_no_model(tmp_path, monkeypatch):
    # 旧产物无 embed_model 字段：按「一致」处理（与加载期检查同策略）
    cd = _make_corpus(tmp_path, "BookA", "甲书")
    monkeypatch.setattr(config, "CORPORA_ROOT", str(tmp_path))
    _write_vector_marker(cd, num_nodes=8)
    status = corpus.probe_vector_index(corpus.load_profile("BookA"))
    assert status.built is True
    assert status.embed_model == ""
    assert status.matches_current is True


def test_query_rewriter_binds_prompts_at_init(tmp_path, monkeypatch):
    """引擎与语料绑定：Rewriter 构造时固化 prompt，之后切书不影响已建实例。"""
    from rag.retrieval.query_rewriter import QueryRewriter

    _make_corpus(tmp_path, "BookA", "甲书")
    _make_corpus(tmp_path, "BookB", "乙书")
    monkeypatch.setattr(config, "CORPORA_ROOT", str(tmp_path))
    original = corpus.get_active_slug()
    try:
        corpus.set_active_corpus("BookA")
        rewriter_a = QueryRewriter(enabled=False)
        assert "甲书" in rewriter_a._nl_prompt
        assert "《甲书》背景" in rewriter_a._hyde_prompt
        # 切书后：新实例绑定新语料，旧实例保持原语料
        corpus.set_active_corpus("BookB")
        rewriter_b = QueryRewriter(enabled=False)
        assert "乙书" in rewriter_b._nl_prompt
        assert "甲书" in rewriter_a._nl_prompt
        assert "乙书" not in rewriter_a._kw_prompt
    finally:
        corpus._active_slug = original
