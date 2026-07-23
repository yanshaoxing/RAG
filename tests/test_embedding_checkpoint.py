"""rag/indexing/embedding_checkpoint.py 单测 —— 分段落盘、断点续传、指纹失效。"""

import os

import pytest
from llama_index.core.schema import TextNode

from rag import config
from rag.indexing.embedding_checkpoint import clear_checkpoint, embed_nodes_with_checkpoint


class _FakeEmbedModel:
    """确定性假 embedding 模型：向量 = [文本长度, 调用序号]，记录调用。"""

    model_name = "fake-embed"

    def __init__(self, fail_after_calls=None):
        self.calls = []
        self._fail_after = fail_after_calls

    def get_text_embedding_batch(self, texts, **kwargs):
        if self._fail_after is not None and len(self.calls) >= self._fail_after:
            raise ConnectionError("模拟网络中断")
        self.calls.append(list(texts))
        return [[float(len(t)), float(len(self.calls))] for t in texts]


def _nodes(n):
    return [TextNode(id_=f"node_{i}", text=f"文本内容{i}") for i in range(n)]


@pytest.fixture
def _cache_dir(tmp_path, monkeypatch):
    d = str(tmp_path / "embed_cache")
    monkeypatch.setattr(config, "EMBED_CHECKPOINT_DIR", d)
    monkeypatch.setattr(config, "EMBED_CHECKPOINT_SEGMENT_NODES", 2)
    return d


class TestEmbedWithCheckpoint:
    def test_all_nodes_embedded(self, _cache_dir):
        nodes = _nodes(5)
        model = _FakeEmbedModel()
        embed_nodes_with_checkpoint(nodes, model)
        assert all(n.embedding is not None for n in nodes)
        # 5 个节点 / 每段 2 个 = 3 段 = 3 次批量调用
        assert len(model.calls) == 3

    def test_resume_only_embeds_missing_segments(self, _cache_dir):
        nodes = _nodes(6)
        # 第一次运行：段 2（第 3 次调用）时网络中断
        broken = _FakeEmbedModel(fail_after_calls=2)
        with pytest.raises(ConnectionError):
            embed_nodes_with_checkpoint(_nodes(6), broken)
        assert len(broken.calls) == 2  # 前 2 段已落盘

        # 续跑：只补缺失的第 3 段
        model = _FakeEmbedModel()
        embed_nodes_with_checkpoint(nodes, model)
        assert len(model.calls) == 1
        assert model.calls[0] == ["文本内容4", "文本内容5"]
        assert all(n.embedding is not None for n in nodes)

    def test_fingerprint_change_invalidates_cache(self, _cache_dir):
        embed_nodes_with_checkpoint(_nodes(4), _FakeEmbedModel())
        # 节点集合变化 → 指纹不匹配 → 缓存作废，全部重算
        other = [TextNode(id_=f"other_{i}", text=f"新文本{i}") for i in range(4)]
        model = _FakeEmbedModel()
        embed_nodes_with_checkpoint(other, model)
        assert len(model.calls) == 2  # 4 节点 / 2 = 2 段全部重算

    def test_cached_rerun_makes_no_calls(self, _cache_dir):
        nodes = _nodes(4)
        embed_nodes_with_checkpoint(nodes, _FakeEmbedModel())
        model = _FakeEmbedModel()
        rerun = _nodes(4)
        embed_nodes_with_checkpoint(rerun, model)
        assert model.calls == []
        assert all(n.embedding is not None for n in rerun)

    def test_clear_checkpoint(self, _cache_dir):
        embed_nodes_with_checkpoint(_nodes(2), _FakeEmbedModel())
        assert os.path.exists(_cache_dir)
        clear_checkpoint()
        assert not os.path.exists(_cache_dir)

    def test_empty_nodes_noop(self, _cache_dir):
        embed_nodes_with_checkpoint([], _FakeEmbedModel())
