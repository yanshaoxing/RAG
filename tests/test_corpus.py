"""rag/corpus.py + 语料级规则合并单测（离线，无 LLM）。"""

import json
import os

import pytest

from rag import config, corpus
from rag.graph import extractor


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
