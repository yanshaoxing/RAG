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


# ======================== 阶段依赖图 / 跳转矩阵 ========================
#
# get_or_build_index() 按【依赖闭包】判定各阶段是否需要重建，是全项目最容易改错的地方：
# 删掉某阶段目录后哪些下游该连带重建、哪些前置产物该「加载」而非「重建」，全靠这段。
# 改错时不会报错，只会静默多跑或少跑阶段（重则整本书 LLM 白烧一遍）。
#
# 依赖图（raw/ 是隐含源）：chunks → summary → {bm25, vector}；graph → (raw)。
# 关键性质（P2-6）：图谱只依赖 raw/，检索阶段（分块/摘要/bm25/向量）的重建【不连带】它；
# bm25 与 vector 互不依赖，也不再互相牵连。
#
# 这里用桩函数替换全部 5 个 build 阶段 + 5 个 load 分支，只断言【调用了哪些、按什么顺序】，
# 不做任何真实索引/IO。_stage_complete 被替换为「路径是否在 completed 集合里」，
# 借此精确控制哪些阶段「已完成」。删除循环里的 rmtree / os.path.exists 也一并桩掉，
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

    # ---- 全部完成：全部加载（chunks 非返回值、无下游重建 → 不加载） ----

    def test_全部完成时全部走加载(self, harness):
        calls, deleted = harness(set(_STAGE_PATHS))
        # chunks 仅作下游构建的输入，全加载时无人需要它 → 省一次反序列化
        assert calls == ["load_summary", "load_bm25", "load_vector", "load_graph"]
        assert "load_chunks" not in calls
        assert deleted == []  # 什么都不重建，就不该删任何东西

    def test_全空_从头重建(self, harness):
        calls, _ = harness(set())  # 没有任何阶段完成
        assert calls == ["stage_chunk", "stage_summary", "stage_bm25", "stage_vector", "build_graph"]

    # ---- 单阶段删除：依赖闭包内重建，闭包外加载 ----

    def test_只删分块_下游检索阶段连带重建但图谱不连带(self, harness):
        calls, _ = harness({"SUMMARY_TREE_DIR", "BM25_DIR", "PERSIST_DIR", "GRAPH_DB_DIR"})
        # chunks dirty → summary/bm25/vector 连带重建；graph 只依赖 raw/ → 加载不重建
        assert calls == ["stage_chunk", "stage_summary", "stage_bm25", "stage_vector", "load_graph"]

    def test_只删摘要_下游bm25向量重建但图谱不连带(self, harness):
        calls, _ = harness({"CHUNKS_DIR", "BM25_DIR", "PERSIST_DIR", "GRAPH_DB_DIR"})
        assert calls == ["load_chunks", "stage_summary", "stage_bm25", "stage_vector", "load_graph"]

    def test_只删向量_图谱不连带重建(self, harness):
        # P2-6 核心：删向量只重建向量，图谱（只依赖 raw/）被加载而非连带重建
        calls, _ = harness({"CHUNKS_DIR", "SUMMARY_TREE_DIR", "BM25_DIR", "GRAPH_DB_DIR"})
        assert calls == ["load_chunks", "load_summary", "load_bm25", "stage_vector", "load_graph"]

    def test_只删bm25_向量不连带重建(self, harness):
        # bm25 与 vector 互不依赖：删 bm25 时 vector 应加载而非重建（旧线性链会连带重建）
        calls, _ = harness({"CHUNKS_DIR", "SUMMARY_TREE_DIR", "PERSIST_DIR", "GRAPH_DB_DIR"})
        assert calls == ["load_chunks", "load_summary", "stage_bm25", "load_vector", "load_graph"]

    def test_只删图谱_检索阶段全加载且不加载chunks(self, harness):
        # 仅图谱重建时 chunks 用不到，既不 build 也不 load（省一次反序列化）
        calls, _ = harness({"CHUNKS_DIR", "SUMMARY_TREE_DIR", "BM25_DIR", "PERSIST_DIR"})
        assert calls == ["load_summary", "load_bm25", "load_vector", "build_graph"]
        assert "load_chunks" not in calls and "stage_chunk" not in calls

    # ---- 删除范围：只删 dirty 阶段（依赖闭包），闭包外产物原样保留 ----

    def test_删向量只清理向量目录_不动图谱(self, harness):
        _, deleted = harness({"CHUNKS_DIR", "SUMMARY_TREE_DIR", "BM25_DIR", "GRAPH_DB_DIR"})
        # 只删向量；图谱/分块/摘要/bm25 全部原样保留（这正是 P2-6 修的浪费）
        assert deleted == [config.PERSIST_DIR]
        assert config.GRAPH_DB_DIR not in deleted

    def test_删分块清理检索链但不删图谱(self, harness):
        _, deleted = harness({"SUMMARY_TREE_DIR", "BM25_DIR", "PERSIST_DIR", "GRAPH_DB_DIR"})
        assert set(deleted) == {config.CHUNKS_DIR, config.SUMMARY_TREE_DIR,
                                config.BM25_DIR, config.PERSIST_DIR}
        assert config.GRAPH_DB_DIR not in deleted

    # ---- 功能开关：禁用阶段既不参与完成度扫描，也不 build/load ----

    def test_禁用摘要树时跳过摘要阶段(self, harness, monkeypatch):
        monkeypatch.setattr(config, "SUMMARY_TREE_ENABLED", False)
        # 摘要禁用 → 不参与 dirty 判定、不向 bm25/vector 传播；这里其余阶段均完成
        calls, _ = harness({"CHUNKS_DIR", "BM25_DIR", "PERSIST_DIR", "GRAPH_DB_DIR"})
        assert "stage_summary" not in calls and "load_summary" not in calls
        assert calls == ["load_bm25", "load_vector", "load_graph"]

    def test_禁用图谱时跳过图谱阶段(self, harness, monkeypatch):
        monkeypatch.setattr(config, "GRAPH_ENABLED", False)
        # 图谱禁用：无论重建还是加载路径都不该碰它（新版无「无条件 load_graph」分支）
        calls, _ = harness({"CHUNKS_DIR", "SUMMARY_TREE_DIR", "BM25_DIR"})
        assert "build_graph" not in calls and "load_graph" not in calls
        assert calls == ["load_chunks", "load_summary", "load_bm25", "stage_vector"]

    def test_禁用图谱_全完成也不加载图谱(self, harness, monkeypatch):
        # 回归：旧线性链在「全完成」分支里无条件 load_graph()，新版依赖图不会
        monkeypatch.setattr(config, "GRAPH_ENABLED", False)
        calls, _ = harness({"CHUNKS_DIR", "SUMMARY_TREE_DIR", "BM25_DIR", "PERSIST_DIR"})
        assert "load_graph" not in calls and "build_graph" not in calls
        assert calls == ["load_summary", "load_bm25", "load_vector"]

    def test_返回值形状(self, harness):
        """返回 (向量索引, bm25, 摘要元数据, 图谱索引) 四元组，桩值按序对上。"""
        harness(set(_STAGE_PATHS))  # 全加载
        result = get_or_build_index()
        # get_or_build_index 在 harness.run 内已被调过一次，这里直接再取一次校验形状
        vec, bm25, meta_map, graph = result
        assert (vec, bm25, graph) == ("VEC", "BM25", "GRAPH")
        assert meta_map == {}


# ======================== 依赖闭包 / --rebuild 计划 ========================

class TestRebuildClosure:
    def test_闭包_删分块波及全部检索阶段但不含图谱(self):
        assert [s.key for s in staged_indexer.plan_rebuild("chunks")] == \
            ["chunks", "summary", "bm25", "vector"]

    def test_闭包_删摘要波及bm25与向量(self):
        assert [s.key for s in staged_indexer.plan_rebuild("summary")] == \
            ["summary", "bm25", "vector"]

    def test_闭包_bm25与向量互不牵连(self):
        assert [s.key for s in staged_indexer.plan_rebuild("bm25")] == ["bm25"]
        assert [s.key for s in staged_indexer.plan_rebuild("vector")] == ["vector"]

    def test_闭包_图谱独立(self):
        assert [s.key for s in staged_indexer.plan_rebuild("graph")] == ["graph"]

    def test_未知阶段报错(self):
        with pytest.raises(ValueError, match="未知阶段"):
            staged_indexer.plan_rebuild("bogus")

    def test_rebuild_预览不删除(self, monkeypatch):
        deleted = []
        monkeypatch.setattr(staged_indexer.shutil, "rmtree", lambda p, **kw: deleted.append(p))
        monkeypatch.setattr(staged_indexer.os.path, "exists", lambda p: True)
        stages = staged_indexer.rebuild_stages("vector", apply=False)
        assert [s.key for s in stages] == ["vector"]
        assert deleted == []  # 预览态绝不删

    def test_rebuild_apply真删闭包内目录(self, monkeypatch):
        deleted = []
        monkeypatch.setattr(staged_indexer.shutil, "rmtree", lambda p, **kw: deleted.append(p))
        monkeypatch.setattr(staged_indexer.os.path, "exists", lambda p: True)
        staged_indexer.rebuild_stages("summary", apply=True)
        # 删除摘要 + 下游 bm25/向量，不含分块/图谱
        assert set(deleted) == {config.SUMMARY_TREE_DIR, config.BM25_DIR, config.PERSIST_DIR}
