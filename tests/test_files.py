"""rag/utils/files.py 单测 —— 原子写入与阶段完成标记（锁定缺陷 #1 的机制）。"""

import json
import os

from rag.utils.files import DONE_MARKER, atomic_write_json, mark_stage_done, stage_complete


class TestAtomicWriteJson:
    def test_roundtrip(self, tmp_path):
        path = str(tmp_path / "out.json")
        atomic_write_json(path, {"键": "值", "n": 1})
        with open(path, encoding="utf-8") as f:
            assert json.load(f) == {"键": "值", "n": 1}

    def test_no_tmp_file_left(self, tmp_path):
        path = str(tmp_path / "out.json")
        atomic_write_json(path, [1, 2, 3])
        assert not os.path.exists(path + ".tmp")

    def test_overwrite_existing(self, tmp_path):
        path = str(tmp_path / "out.json")
        atomic_write_json(path, {"v": 1})
        atomic_write_json(path, {"v": 2})
        with open(path, encoding="utf-8") as f:
            assert json.load(f) == {"v": 2}


class TestStageMarker:
    def test_dir_without_marker_incomplete(self, tmp_path):
        # 缺陷 #1 回归：目录存在但无标记 = 中断的构建，不能视为完成
        stage_dir = tmp_path / "chunks"
        stage_dir.mkdir()
        (stage_dir / "partial.json").write_text("{}")
        assert not stage_complete(str(stage_dir))

    def test_mark_then_complete(self, tmp_path):
        stage_dir = str(tmp_path / "chunks")
        os.makedirs(stage_dir)
        mark_stage_done(stage_dir, node_count=42)
        assert stage_complete(stage_dir)
        with open(os.path.join(stage_dir, DONE_MARKER), encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["node_count"] == 42
        assert "completed_at" in payload

    def test_missing_dir_incomplete(self, tmp_path):
        assert not stage_complete(str(tmp_path / "nonexistent"))
