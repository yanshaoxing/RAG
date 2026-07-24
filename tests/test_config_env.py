"""配置的环境变量覆盖 —— provider 选择与功能开关必须能免改代码切换。

消融实验（关摘要树 / 关图谱 / 关重排 …）依赖这些开关，
故此处锁定 _env_str / _env_bool 的解析语义与各开关的接线。
"""

import importlib

import pytest

from rag import config

# ---------- 纯解析语义 ----------

@pytest.mark.parametrize("raw,expected", [
    ("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True),
    ("0", False), ("false", False), ("no", False), ("off", False), ("随便", False),
])
def test_env_bool_解析(monkeypatch, raw, expected):
    monkeypatch.setenv("RAG_TEST_FLAG", raw)
    assert config._env_bool("RAG_TEST_FLAG", not expected) is expected


def test_env_bool_未设置或空串取默认值(monkeypatch):
    monkeypatch.delenv("RAG_TEST_FLAG", raising=False)
    assert config._env_bool("RAG_TEST_FLAG", True) is True
    assert config._env_bool("RAG_TEST_FLAG", False) is False
    # 空串/纯空白视为未设置，避免 `export RAG_X=` 意外关掉功能
    monkeypatch.setenv("RAG_TEST_FLAG", "   ")
    assert config._env_bool("RAG_TEST_FLAG", True) is True


def test_env_str_未设置或空串取默认值(monkeypatch):
    monkeypatch.delenv("RAG_TEST_STR", raising=False)
    assert config._env_str("RAG_TEST_STR", "aliyun") == "aliyun"
    monkeypatch.setenv("RAG_TEST_STR", "  ")
    assert config._env_str("RAG_TEST_STR", "aliyun") == "aliyun"
    monkeypatch.setenv("RAG_TEST_STR", " ollama ")
    assert config._env_str("RAG_TEST_STR", "aliyun") == "ollama"


# ---------- 各开关确实接到了环境变量 ----------

@pytest.fixture
def reloaded_config(monkeypatch):
    """在打了环境变量的前提下重载 config，测试结束后还原为干净默认值。

    config 是进程级单例模块，reload 是原地更新，其他模块持有的引用仍有效。
    """
    def _reload(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return importlib.reload(config)

    yield _reload

    monkeypatch.undo()
    importlib.reload(config)


@pytest.mark.parametrize("env_name,attr,value,expected", [
    ("RAG_ANSWER_PROVIDER", "ANSWER_PROVIDER", "ollama", "ollama"),
    ("RAG_REWRITE_PROVIDER", "REWRITE_PROVIDER", "davy", "davy"),
    ("RAG_RERANK_PROVIDER", "RERANK_PROVIDER", "vllm", "vllm"),
    ("RAG_SUMMARY_LLM_PROVIDER", "SUMMARY_LLM_PROVIDER", "ollama", "ollama"),
    ("RAG_GRAPH_EXTRACT_LLM_PROVIDER", "GRAPH_EXTRACT_LLM_PROVIDER", "davy", "davy"),
    ("RAG_GRAPH_VALIDATE_LLM_PROVIDER", "GRAPH_VALIDATE_LLM_PROVIDER", "ollama", "ollama"),
    ("RAG_SUMMARY_TREE_ENABLED", "SUMMARY_TREE_ENABLED", "0", False),
    ("RAG_GRAPH_ENABLED", "GRAPH_ENABLED", "0", False),
    ("RAG_RERANK_ENABLED", "RERANK_ENABLED", "0", False),
    ("RAG_REWRITE_ENABLED", "REWRITE_ENABLED", "0", False),
    ("RAG_DECOMPOSE_ENABLED", "DECOMPOSE_ENABLED", "0", False),
    ("RAG_GRAPH_VALIDATE_ENABLED", "GRAPH_VALIDATE_ENABLED", "0", False),
    ("RAG_ANSWER_STREAM_ENABLED", "ANSWER_STREAM_ENABLED", "0", False),
    ("RAG_STRUCTURE_DETECT_ENABLED", "STRUCTURE_DETECT_ENABLED", "0", False),
])
def test_开关可被环境变量覆盖(reloaded_config, env_name, attr, value, expected):
    cfg = reloaded_config(**{env_name: value})
    assert getattr(cfg, attr) == expected


def test_嵌入维度随_provider_切换(reloaded_config):
    """EMBED_VECTOR_DIM 由 provider 派生，切 provider 后必须跟着变（否则 FAISS 维度不匹配）。"""
    assert reloaded_config(RAG_EMBED_PROVIDER="ollama").EMBED_VECTOR_DIM == 4096
    assert reloaded_config(RAG_EMBED_PROVIDER="aliyun").EMBED_VECTOR_DIM == 1024


def test_密钥不留明文兜底值(reloaded_config, monkeypatch):
    """所有 API key 未配置时必须为空串 —— 代码里不允许出现明文密钥。"""
    monkeypatch.delenv("RAG_DAVY_API_KEY", raising=False)
    cfg = reloaded_config()
    assert cfg.DAVY_API_KEY == ""
