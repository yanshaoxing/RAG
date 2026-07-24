"""rag/graph/cache.py 单测 —— 断点续传的正确性（决定全书图谱长跑能否恢复）。

核心不变量：
  - chunk 状态机 pending/processing/completed/failed，只有 completed 且文本未变才算「已处理」；
    processing（构建中途崩溃残留）绝不能被当成已完成，否则续跑时会漏掉这个 chunk；
  - 状态跨进程持久化：重开 DB（模拟重启）后已完成集合原样恢复；
  - 实体/关系写入-读取字段无损；归一化重写关系端点；描述合并取最长。
"""

from rag.graph.cache import GraphCache
from rag.graph.models import Entity, Relation


def _cache(tmp_path, name="g.db"):
    return GraphCache(str(tmp_path / name))


# ---------- Chunk 状态机：断点续传的地基 ----------

class TestChunkStateMachine:
    def test_未处理的chunk不算已完成(self, tmp_path):
        c = _cache(tmp_path)
        assert c.chunk_exists(0, "原文") is False
        c.close()

    def test_processing状态不算已完成(self, tmp_path):
        # 中途崩溃残留的 processing chunk 必须重新处理，否则续跑漏数据
        c = _cache(tmp_path)
        c.mark_chunk_processing(0, "原文")
        assert c.chunk_exists(0, "原文") is False
        assert c.get_completed_chunk_ids() == set()
        c.close()

    def test_completed状态算已完成(self, tmp_path):
        c = _cache(tmp_path)
        c.mark_chunk_processing(0, "原文")
        c.mark_chunk_completed(0)
        assert c.chunk_exists(0, "原文") is True
        assert c.get_completed_chunk_ids() == {0}
        c.close()

    def test_文本变化则视为未处理(self, tmp_path):
        # text_hash 不匹配 → 需重新抽取（比如换了分块参数）
        c = _cache(tmp_path)
        c.mark_chunk_processing(0, "原文")
        c.mark_chunk_completed(0)
        assert c.chunk_exists(0, "改过的原文") is False
        c.close()

    def test_failed状态不算已完成(self, tmp_path):
        c = _cache(tmp_path)
        c.mark_chunk_processing(3, "原文")
        c.mark_chunk_failed(3)
        assert c.chunk_exists(3, "原文") is False
        assert c.get_completed_chunk_ids() == set()
        c.close()

    def test_失败后重试可转完成(self, tmp_path):
        c = _cache(tmp_path)
        c.mark_chunk_processing(3, "原文")
        c.mark_chunk_failed(3)
        c.mark_chunk_processing(3, "原文")
        c.mark_chunk_completed(3)
        assert c.chunk_exists(3, "原文") is True
        c.close()


# ---------- 跨进程持久化：模拟重启后恢复 ----------

class TestPersistenceAcrossReopen:
    def test_已完成集合重开后恢复(self, tmp_path):
        c = _cache(tmp_path)
        for i in (0, 1, 2):
            c.mark_chunk_processing(i, f"chunk-{i}")
            c.mark_chunk_completed(i)
        c.mark_chunk_processing(3, "chunk-3")   # 未完成（模拟崩溃点）
        c.close()

        # 新进程重开同一个 DB 文件
        c2 = _cache(tmp_path)
        assert c2.get_completed_chunk_ids() == {0, 1, 2}
        assert c2.chunk_exists(3, "chunk-3") is False   # 崩溃点会被重跑
        c2.close()

    def test_实体关系重开后可读(self, tmp_path):
        c = _cache(tmp_path)
        c.save_entities([Entity(name="丁元英", description="商界怪才", chunk_id=0)])
        c.save_relations([Relation(subject="丁元英", predicate="好友", object="韩楚风", chunk_id=0)])
        c.close()

        c2 = _cache(tmp_path)
        assert c2.get_all_entity_names() == ["丁元英"]
        rels = c2.get_all_relations()
        assert len(rels) == 1 and rels[0].triple_key == ("丁元英", "好友", "韩楚风")
        c2.close()


# ---------- 关系读写：字段无损 ----------

class TestRelationRoundTrip:
    def test_字段完整往返(self, tmp_path):
        c = _cache(tmp_path)
        r = Relation(
            subject="丁元英", predicate="指点", object="王庙村",
            subject_type="人物", object_type="地点",
            description="扶贫神话的策划", confidence=0.9,
            validated=True, chunk_id=7, source_text="原文片段",
            extract_model="qwen3.5-flash", validate_model="qwen-flash",
        )
        c.save_relations([r])
        got = c.get_all_relations()[0]
        assert got.subject_type == "人物" and got.object_type == "地点"
        assert got.description == "扶贫神话的策划"
        assert got.confidence == 0.9
        assert got.validated is True          # int 1 正确还原为 bool
        assert got.chunk_id == 7
        assert got.extract_model == "qwen3.5-flash"
        assert got.validate_model == "qwen-flash"
        c.close()

    def test_按chunk查询关系(self, tmp_path):
        c = _cache(tmp_path)
        c.save_relations([
            Relation(subject="a", predicate="p", object="b", chunk_id=1),
            Relation(subject="c", predicate="p", object="d", chunk_id=2),
        ])
        rels = c.get_relations_by_chunk(2)
        assert len(rels) == 1 and rels[0].subject == "c"
        c.close()

    def test_关系映射按三元组键(self, tmp_path):
        c = _cache(tmp_path)
        c.save_relations([Relation(subject="丁元英", predicate="好友", object="韩楚风", chunk_id=0)])
        rel_map = c.build_relation_map()
        assert ("丁元英", "好友", "韩楚风") in rel_map
        c.close()


# ---------- 实体归一化 / 描述合并 ----------

class TestEntityCanonicalAndDesc:
    def test_归一化重写关系端点(self, tmp_path):
        c = _cache(tmp_path)
        c.save_entities([Entity(name="元英", description="即丁元英", chunk_id=0)])
        c.save_relations([
            Relation(subject="元英", predicate="好友", object="韩楚风", chunk_id=0),
            Relation(subject="肖亚文", predicate="助理", object="元英", chunk_id=0),
        ])
        c.update_entity_canonical("元英", "丁元英")
        keys = {r.triple_key for r in c.get_all_relations()}
        assert ("丁元英", "好友", "韩楚风") in keys
        assert ("肖亚文", "助理", "丁元英") in keys
        assert c.get_all_entity_names() == ["丁元英"]
        c.close()

    def test_描述合并取最长(self, tmp_path):
        # 同一 canonical 有多行（多 chunk），get_all_entity_descriptions 取最长那条
        c = _cache(tmp_path)
        c.save_entities([
            Entity(name="丁元英", description="短描述", chunk_id=0),
            Entity(name="丁元英", description="更长更完整的人物描述", chunk_id=1),
        ])
        descs = c.get_all_entity_descriptions()
        assert descs["丁元英"] == "更长更完整的人物描述"
        c.close()

    def test_update描述取最大置信度(self, tmp_path):
        c = _cache(tmp_path)
        c.save_entities([Entity(name="丁元英", description="原始", confidence=0.3, chunk_id=0)])
        c.update_entity_description("丁元英", "合并后描述", confidence=0.9)
        assert c.get_all_entity_descriptions()["丁元英"] == "合并后描述"
        c.close()

    def test_无描述实体不进描述映射但进名称列表(self, tmp_path):
        c = _cache(tmp_path)
        c.save_entities([
            Entity(name="丁元英", description="有描述", chunk_id=0),
            Entity(name="王庙村", description="", chunk_id=0),
        ])
        assert set(c.get_all_entity_names()) == {"丁元英", "王庙村"}
        assert set(c.get_all_entity_descriptions()) == {"丁元英"}
        c.close()


# ---------- 指纹 / 元数据 ----------

class TestFingerprintAndMetadata:
    def test_指纹设置与校验(self, tmp_path):
        c = _cache(tmp_path)
        assert c.is_fingerprint_valid("fp-1") is False
        c.set_fingerprint("fp-1")
        assert c.is_fingerprint_valid("fp-1") is True
        assert c.is_fingerprint_valid("fp-2") is False
        c.close()

    def test_元数据读写(self, tmp_path):
        c = _cache(tmp_path)
        assert c.get_metadata("缺失键") is None
        c.set_metadata("k", "v")
        assert c.get_metadata("k") == "v"
        c.set_metadata("k", "v2")            # INSERT OR REPLACE
        assert c.get_metadata("k") == "v2"
        c.close()

    def test_上下文管理器自动关闭(self, tmp_path):
        with GraphCache(str(tmp_path / "ctx.db")) as c:
            c.set_metadata("k", "v")
        assert c._conn is None
