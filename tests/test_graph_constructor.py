"""rag/graph/graph_constructor.py 单测 —— 模型名解析、流水线、记账去重（新缺陷 4/5/6）。"""

import threading

from rag import config
from rag.graph.cache import GraphCache
from rag.graph.graph_constructor import (
    _book_chunk_result,
    _resolve_graph_models,
    _run_pipelined,
)
from rag.graph.metrics import MetricsCollector
from rag.graph.models import ChunkResult, Entity, Relation
from rag.graph.schema import Schema

# ---------- 模型名解析（新缺陷 4） ----------

class _FakeDavyLLM:
    model_name = "nemotron-3-ultra"


class _FakeOllamaLLM:
    model = "qwen3.5:9b"


class TestResolveGraphModels:
    def test_davy_validate_model(self, monkeypatch):
        monkeypatch.setattr(config, "GRAPH_VALIDATE_LLM_PROVIDER", "davy")
        extract, validate = _resolve_graph_models(_FakeDavyLLM())
        assert extract == "nemotron-3-ultra"        # 实际模型名，而非 provider
        assert validate == config.GRAPH_VALIDATE_DAVY_MODEL

    def test_ollama_validate_model(self, monkeypatch):
        monkeypatch.setattr(config, "GRAPH_VALIDATE_LLM_PROVIDER", "ollama")
        extract, validate = _resolve_graph_models(_FakeOllamaLLM())
        assert extract == "qwen3.5:9b"
        assert validate == config.GRAPH_VALIDATE_LLM_MODEL


# ---------- 有界流水线（新缺陷 5） ----------

class TestRunPipelined:
    def test_handle_order_matches_submission_order(self):
        handled = []
        _run_pipelined(
            list(range(20)),
            worker_fn=lambda i: i * 10,
            handle_fn=lambda i, out: handled.append((i, out)),
            max_workers=4,
        )
        assert handled == [(i, i * 10) for i in range(20)]

    def test_worker_concurrency_bounded(self):
        lock = threading.Lock()
        state = {"active": 0, "peak": 0}
        barrier = threading.Event()

        def _worker(i):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            barrier.wait(timeout=0.05)   # 制造重叠窗口
            with lock:
                state["active"] -= 1
            return i

        _run_pipelined(list(range(12)), _worker, lambda i, out: None, max_workers=3)
        assert state["peak"] <= 3
        assert state["peak"] >= 2       # 确实发生了并发

    def test_empty_and_single(self):
        _run_pipelined([], lambda i: i, lambda i, o: None, max_workers=2)  # 不抛错
        out = []
        _run_pipelined([7], lambda i: i, lambda i, o: out.append(o), max_workers=2)
        assert out == [7]


# ---------- 记账：merge/canonicalize 按实体去重（新缺陷 6） ----------

class _FakeCache:
    def __init__(self):
        self.completed = []
        self.failed = []
        self.canonical_updates = []
        self.desc_updates = []
        self.saved_relations = []

    def mark_chunk_processing(self, idx, text): pass
    def mark_chunk_completed(self, idx): self.completed.append(idx)
    def mark_chunk_failed(self, idx): self.failed.append(idx)
    def save_entities(self, entities): pass
    def update_entity_canonical(self, orig, canonical): self.canonical_updates.append((orig, canonical))
    def update_entity_description(self, name, desc): self.desc_updates.append(name)
    def save_relations(self, relations): self.saved_relations.extend(relations)


class _CountingMerger:
    def __init__(self):
        self.calls = []

    def merge(self, existing, new, name):
        self.calls.append(name)
        return f"{existing}+{new}" if existing else new


class _ScriptedCanonicalizer:
    def __init__(self, mapping):
        self._mapping = mapping
        self.calls = []

    def canonicalize(self, candidate, known_names):
        self.calls.append(candidate)
        return self._mapping.get(candidate)


def _rel(subj, obj, pred="合作"):
    return Relation(subject=subj, predicate=pred, object=obj, chunk_id=1)


class TestBookChunkResult:
    def _book(self, relations, entities, canonical_map=None, known=None):
        cache = _FakeCache()
        merger = _CountingMerger()
        canonicalizer = _ScriptedCanonicalizer(canonical_map or {})
        metrics = MetricsCollector()
        descs: dict[str, str] = {}
        rel_map: dict[tuple, Relation] = {}
        known_list = list(known or [])
        result = ChunkResult(chunk_id=1, entities=entities, relations=relations)
        _book_chunk_result(
            1, "原文", result, relations, cache, merger, canonicalizer,
            metrics, descs, rel_map, known_list, lambda m: None,
        )
        return cache, merger, canonicalizer, descs, rel_map, known_list

    def test_merge_called_once_per_unique_entity(self):
        # 丁元英出现在 3 条关系里，merge 只能调用 1 次（此前最多 3 次）
        entities = [Entity(name="丁元英", description="商界怪才，私募基金操盘手"),
                    Entity(name="韩楚风", description="正天集团总裁，丁元英好友"),
                    Entity(name="肖亚文", description="丁元英的前助理"),
                    Entity(name="欧阳雪", description="饭店老板，芮小丹好友")]
        relations = [_rel("丁元英", "韩楚风"), _rel("丁元英", "肖亚文"), _rel("丁元英", "欧阳雪")]
        _, merger, _, descs, rel_map, _ = self._book(relations, entities)
        assert merger.calls.count("丁元英") == 1
        assert sorted(set(merger.calls)) == ["丁元英", "欧阳雪", "肖亚文", "韩楚风"]
        assert len(rel_map) == 3

    def test_canonicalize_called_once_per_unique_unknown(self):
        entities = [Entity(name="元英", description="即丁元英，商界怪才")]
        relations = [_rel("元英", "韩楚风"), _rel("元英", "肖亚文")]
        cache, _, canonicalizer, _, rel_map, _ = self._book(
            relations, entities,
            canonical_map={"元英": "丁元英"},
            known=["丁元英", "韩楚风", "肖亚文"],
        )
        assert canonicalizer.calls.count("元英") == 1
        assert cache.canonical_updates == [("元英", "丁元英")]
        # 关系端点已重写为 canonical 名
        assert all(k[0] == "丁元英" for k in rel_map)

    def test_new_entities_join_known_list(self):
        entities = [Entity(name="王庙村", description="贫困村，格律诗扶贫基地")]
        relations = [_rel("王庙村", "格律诗")]
        _, _, _, _, _, known_list = self._book(relations, entities)
        assert "王庙村" in known_list and "格律诗" in known_list

    def test_extraction_failure_marks_failed(self):
        cache = _FakeCache()
        metrics = MetricsCollector()
        _book_chunk_result(
            5, "原文", None, [], cache, _CountingMerger(), _ScriptedCanonicalizer({}),
            metrics, {}, {}, [], lambda m: None,
        )
        assert cache.failed == [5]


# ---------- cache：无描述实体也参与归一化参照 ----------

class TestGetAllEntityNames:
    def test_includes_entities_without_description(self, tmp_path):
        cache = GraphCache(str(tmp_path / "g.db"))
        cache.save_entities([
            Entity(name="丁元英", description="商界怪才", chunk_id=0),
            Entity(name="王庙村", description="", chunk_id=0),   # 无描述
        ])
        names = cache.get_all_entity_names()
        assert set(names) == {"丁元英", "王庙村"}
        # 对照：描述映射只含有描述的
        assert set(cache.get_all_entity_descriptions()) == {"丁元英"}
        cache.close()


# ---------- Schema：并发 resolve_type 计数正确 ----------

class TestSchemaThreadSafety:
    def test_concurrent_unknown_type_counting(self):
        schema = Schema(growth_threshold=10_000)   # 不触发升级，专注计数
        n_threads, n_calls = 4, 50

        def _hammer():
            for _ in range(n_calls):
                schema.resolve_type("稀有类型")

        threads = [threading.Thread(target=_hammer) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert schema.get_unknown_type_report()["稀有类型"] == n_threads * n_calls
