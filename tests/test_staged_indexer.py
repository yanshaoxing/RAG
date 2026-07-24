"""rag/indexing/staged_indexer.py 单测 —— 向量阶段的嵌入模型一致性守卫。

背景：2026-07-23 换嵌入模型（text-embedding-v4 → qwen3.7-text-embedding）时发现，
两者**维度相同（都是 1024）但向量空间不同**——旧索引配新查询向量算出的相似度
毫无意义，且不会报任何错，属于静默数据损坏。故向量阶段完成标记记录 embed_model，
加载时比对不一致直接报错。
"""

import types

import pytest

from rag import config
from rag.indexing import staged_indexer
from rag.indexing.staged_indexer import _check_embed_model_match, get_or_build_index
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


# ======================== 阶段跳转矩阵 ========================
#
# get_or_build_index() 的 if 链（staged_indexer.py:388-466）是全项目最容易改错的地方：
# 删掉第 N 阶段目录后从哪一步开始重建、哪些前置产物该「加载」而非「重建」，全靠这段。
# 改错时不会报错，只会静默多跑或少跑阶段（重则整本书 LLM 白烧一遍）。
#
# 这里用桩函数替换全部 5 个 build 阶段 + 5 个 load 分支，只断言【调用了哪些、按什么顺序】，
# 不做任何真实索引/IO。_stage_complete 被替换为「路径是否在 completed 集合里」，
# 借此精确控制 start_from。删除循环里的 rmtree / os.path.exists 也一并桩掉，
# 绝不能让测试误删真实语料的 data/ 目录。

_STAGE_PATHS = ["CHUNKS_DIR", "SUMMARY_TREE_DIR", "BM25_DIR", "PERSIST_DIR", "GRAPH_DB_DIR"]


class TestStageTransitionMatrix:
    @pytest.fixture
    def harness(self, monkeypatch):
        """返回 run(completed_attrs) → (calls, deleted)。

        completed_attrs：已完成阶段的 config 属性名集合（如 {"CHUNKS_DIR"}）。
        calls：按实际调用顺序记录的阶段函数名列表。
        deleted：删除循环里被 rmtree 的路径列表。
        """
        calls: list[str] = []
        deleted: list[str] = []

        def _rec(name, ret):
            def _fn(*args, **kwargs):
                calls.append(name)
                return ret
            return _fn

        # 5 个 build 阶段 + 5 个 load 分支：桩掉，只记名，返回下游能用的占位值
        monkeypatch.setattr(staged_indexer, "_stage_chunk", _rec("stage_chunk", []))
        monkeypatch.setattr(staged_indexer, "_load_chunks", _rec("load_chunks", []))
        monkeypatch.setattr(staged_indexer, "_stage_summary", _rec("stage_summary", ([], {})))
        monkeypatch.setattr(staged_indexer, "_load_summary", _rec("load_summary", ([], {})))
        monkeypatch.setattr(staged_indexer, "_stage_bm25", _rec("stage_bm25", "BM25"))
        monkeypatch.setattr(staged_indexer, "_load_bm25", _rec("load_bm25", "BM25"))
        monkeypatch.setattr(staged_indexer, "_stage_vector", _rec("stage_vector", "VEC"))
        monkeypatch.setattr(staged_indexer, "_load_vector", _rec("load_vector", "VEC"))
        monkeypatch.setattr(staged_indexer, "build_graph", _rec("build_graph", "GRAPH"))
        monkeypatch.setattr(staged_indexer, "load_graph", _rec("load_graph", "GRAPH"))
        monkeypatch.setattr(staged_indexer, "load_documents", lambda: [])

        # Settings.llm 未初始化时被访问会触发 llama_index 回退解析默认 OpenAI 并报错；
        # build_graph 已桩掉，这里只需给一个不触发解析的假 Settings
        monkeypatch.setattr(staged_indexer, "Settings",
                            types.SimpleNamespace(llm=None, embed_model=None))

        # 删除循环：桩掉，绝不真删；os.path.exists 恒 True 以便记录会被删的阶段
        monkeypatch.setattr(staged_indexer.shutil, "rmtree",
                            lambda p, **kw: deleted.append(p))
        monkeypatch.setattr(staged_indexer.os.path, "exists", lambda p: True)

        def run(completed_attrs):
            calls.clear()
            deleted.clear()
            completed_paths = {getattr(config, a) for a in completed_attrs}
            monkeypatch.setattr(staged_indexer, "_stage_complete",
                                lambda path: path in completed_paths)
            get_or_build_index()
            return calls, deleted

        return run

    @pytest.fixture(autouse=True)
    def _all_enabled(self, monkeypatch):
        monkeypatch.setattr(config, "SUMMARY_TREE_ENABLED", True)
        monkeypatch.setattr(config, "GRAPH_ENABLED", True)

    # ---- 全启用：删第 N 阶段 → 恰好 rebuild 第 N..4、load 第 0..N-1 ----

    def test_全部完成时全部走加载(self, harness):
        calls, deleted = harness(set(_STAGE_PATHS))
        assert calls == ["load_chunks", "load_summary", "load_bm25", "load_vector", "load_graph"]
        assert deleted == []  # 什么都不重建，就不该删任何东西

    def test_删分块阶段_从头重建(self, harness):
        calls, _ = harness(set())  # 没有任何阶段完成 → start_from=0
        assert calls == ["stage_chunk", "stage_summary", "stage_bm25", "stage_vector", "build_graph"]

    def test_删摘要阶段_加载分块其余重建(self, harness):
        calls, _ = harness({"CHUNKS_DIR"})
        assert calls == ["load_chunks", "stage_summary", "stage_bm25", "stage_vector", "build_graph"]

    def test_删bm25阶段(self, harness):
        calls, _ = harness({"CHUNKS_DIR", "SUMMARY_TREE_DIR"})
        assert calls == ["load_chunks", "load_summary", "stage_bm25", "stage_vector", "build_graph"]

    def test_删向量阶段(self, harness):
        calls, _ = harness({"CHUNKS_DIR", "SUMMARY_TREE_DIR", "BM25_DIR"})
        assert calls == ["load_chunks", "load_summary", "load_bm25", "stage_vector", "build_graph"]

    def test_只删图谱阶段_前四阶段全加载且不加载chunks(self, harness):
        # 关键分支：仅图谱重建时 chunks 用不到，既不 build 也不 load（省一次反序列化）
        calls, _ = harness({"CHUNKS_DIR", "SUMMARY_TREE_DIR", "BM25_DIR", "PERSIST_DIR"})
        assert calls == ["load_summary", "load_bm25", "load_vector", "build_graph"]
        assert "load_chunks" not in calls and "stage_chunk" not in calls

    # ---- 删除范围：只删 start_from 及其下游 ----

    def test_删向量阶段只清理向量与图谱目录(self, harness):
        _, deleted = harness({"CHUNKS_DIR", "SUMMARY_TREE_DIR", "BM25_DIR"})
        # start_from=3（向量）→ 清理向量 + 图谱，不动分块/摘要/bm25
        assert config.PERSIST_DIR in deleted
        assert config.GRAPH_DB_DIR in deleted
        assert config.CHUNKS_DIR not in deleted
        assert config.BM25_DIR not in deleted

    # ---- 功能开关：禁用阶段既不参与完成度扫描，也不 build/load ----

    def test_禁用摘要树时跳过摘要阶段(self, harness, monkeypatch):
        monkeypatch.setattr(config, "SUMMARY_TREE_ENABLED", False)
        # 摘要目录「未完成」也不该触发重建；从分块之后直接到 bm25
        calls, _ = harness({"CHUNKS_DIR", "BM25_DIR", "PERSIST_DIR", "GRAPH_DB_DIR"})
        assert "stage_summary" not in calls and "load_summary" not in calls
        assert calls == ["load_chunks", "load_bm25", "load_vector", "load_graph"]

    def test_禁用图谱时跳过图谱阶段(self, harness, monkeypatch):
        # 注意：全部完成的加载分支里 load_graph() 是无条件调用的，
        # GRAPH_ENABLED 的守卫只在【重建路径】生效，故这里用删向量触发重建。
        monkeypatch.setattr(config, "GRAPH_ENABLED", False)
        calls, _ = harness({"CHUNKS_DIR", "SUMMARY_TREE_DIR", "BM25_DIR"})
        assert "build_graph" not in calls and "load_graph" not in calls
        assert calls == ["load_chunks", "load_summary", "load_bm25", "stage_vector"]

    def test_返回值形状(self, harness):
        """返回 (向量索引, bm25, 摘要元数据, 图谱索引) 四元组，桩值按序对上。"""
        harness(set(_STAGE_PATHS))  # 全加载
        result = get_or_build_index()
        # get_or_build_index 在 harness.run 内已被调过一次，这里直接再取一次校验形状
        vec, bm25, meta_map, graph = result
        assert (vec, bm25, graph) == ("VEC", "BM25", "GRAPH")
        assert meta_map == {}
