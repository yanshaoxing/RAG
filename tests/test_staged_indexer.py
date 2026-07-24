"""rag/indexing/staged_indexer.py 单测 —— 向量阶段的嵌入模型一致性守卫。

背景：2026-07-23 换嵌入模型（text-embedding-v4 → qwen3.7-text-embedding）时发现，
两者**维度相同（都是 1024）但向量空间不同**——旧索引配新查询向量算出的相似度
毫无意义，且不会报任何错，属于静默数据损坏。故向量阶段完成标记记录 embed_model，
加载时比对不一致直接报错。
"""

import pytest

from rag import config
from rag.indexing.staged_indexer import _check_embed_model_match
from rag.utils.files import mark_stage_done, read_stage_info


class TestReadStageInfo:
    def test_读取完成标记内容(self, tmp_path):
        mark_stage_done(str(tmp_path), num_nodes=7, embed_model="m-1")
        info = read_stage_info(str(tmp_path))
        assert info["num_nodes"] == 7
        assert info["embed_model"] == "m-1"
        assert "completed_at" in info

    def test_标记不存在返回空字典(self, tmp_path):
        assert read_stage_info(str(tmp_path)) == {}

    def test_标记损坏返回空字典(self, tmp_path):
        (tmp_path / "_DONE.json").write_text("{不是合法 json", encoding="utf-8")
        assert read_stage_info(str(tmp_path)) == {}


class TestEmbedModelGuard:
    @pytest.fixture
    def stage_dir(self, tmp_path, monkeypatch):
        """把向量阶段目录指向 tmp_path（PERSIST_DIR 是语料派生的动态属性，需整体替换）。"""
        monkeypatch.setattr(config, "PERSIST_DIR", str(tmp_path), raising=False)
        return tmp_path

    def test_模型一致时通过(self, stage_dir, monkeypatch):
        monkeypatch.setattr(config, "ACTIVE_EMBED_MODEL_NAME", "qwen3.7-text-embedding")
        mark_stage_done(str(stage_dir), embed_model="qwen3.7-text-embedding")
        _check_embed_model_match()  # 不抛异常即通过

    def test_模型不一致时报错并给出修复指引(self, stage_dir, monkeypatch):
        monkeypatch.setattr(config, "ACTIVE_EMBED_MODEL_NAME", "qwen3.7-text-embedding")
        mark_stage_done(str(stage_dir), embed_model="text-embedding-v4")
        with pytest.raises(RuntimeError) as exc:
            _check_embed_model_match()
        msg = str(exc.value)
        # 报错必须同时给出「旧模型、新模型、怎么修」——否则用户不知道该删哪个目录
        assert "text-embedding-v4" in msg
        assert "qwen3.7-text-embedding" in msg
        assert str(stage_dir) in msg

    def test_旧产物无该字段时跳过检查(self, stage_dir, monkeypatch):
        """兼容此前构建的产物（标记里没有 embed_model 字段），不能一升级就全体报错。"""
        monkeypatch.setattr(config, "ACTIVE_EMBED_MODEL_NAME", "qwen3.7-text-embedding")
        mark_stage_done(str(stage_dir), num_nodes=10)
        _check_embed_model_match()

    def test_无完成标记时跳过检查(self, stage_dir, monkeypatch):
        monkeypatch.setattr(config, "ACTIVE_EMBED_MODEL_NAME", "任意模型")
        _check_embed_model_match()
