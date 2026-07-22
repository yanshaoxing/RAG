"""
知识图谱构建模块 —— 使用 LLM 从 chunk 中抽取实体和关系，构建 PropertyGraph，持久化到 Kuzu。

架构：
  Chunk → Extractor → Validator → Merger → Canonicalizer → Kuzu
                   ↑            ↑          ↑            ↑
               rules.yaml    Schema    SQLite Cache   Metrics

对外接口保持不变：
  build_graph(documents, llm, ...) → PropertyGraphIndex
  load_graph(...) → PropertyGraphIndex
"""

import logging
import os
import shutil
from typing import Optional

import kuzu
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.schema import Document
from llama_index.graph_stores.kuzu import KuzuPropertyGraphStore

from rag import config
from rag.llm.factory import create_validate_llm
from rag.utils.files import mark_stage_done

from .cache import GraphCache
from .canonicalizer import Canonicalizer
from .extractor import Extractor
from .merger import DescriptionMerger
from .metrics import MetricsCollector
from .models import Relation as RelModel
from .schema import Schema
from .validator import Validator

logger = logging.getLogger(__name__)


def _log(msg: str) -> None:
    """管线日志：经标准 logging 输出，入口层用 capture_pipeline_logs 捕获。"""
    logger.info(msg)


def _prepare_sections_for_graph(documents: list[Document]) -> list[tuple[int, str]]:
    """按节直接送入 LLM，不做分块。"""
    sections = []
    for i, doc in enumerate(documents):
        text = (doc.text or "").strip()
        if len(text) >= 50:
            sections.append((i, text))
    return sections


def _process_one_chunk(
    idx: int,
    text: str,
    cache: GraphCache,
    extractor: Extractor,
    validator: Validator,
    merger: DescriptionMerger,
    canonicalizer: Canonicalizer,
    metrics: MetricsCollector,
    global_entity_descriptions: dict[str, str],
    global_relation_map: dict[tuple, RelModel],
    all_known_entities: list[str],
    _log,
) -> None:
    """处理单个 chunk：抽取 → 校验 → 归一化 → 描述合并 → 入缓存。"""
    cache.mark_chunk_processing(idx, text)

    # 抽取
    result = extractor.extract(text, idx)
    if result is None:
        _log(f"  ⚠️ chunk #{idx}: 抽取失败")
        cache.mark_chunk_failed(idx)
        metrics.record_chunk_failed()
        return

    if result.is_empty:
        # 只有实体、没有关系：保存实体描述并标记完成，
        # 不再标记失败（失败会导致每次续跑都重抽，实体描述被反复丢弃）
        if result.entities:
            cache.save_entities(result.entities)
            metrics.record_entity(len(result.entities))
            _log(f"  ⚠️ chunk #{idx}: 无关系，已保存 {len(result.entities)} 个实体")
        else:
            _log(f"  ⚠️ chunk #{idx}: 未抽取到实体和关系")
        cache.mark_chunk_completed(idx)
        metrics.record_chunk_success()
        return

    # 校验
    validated_relations = validator.validate(result.relations, text)
    filtered_count = len(result.relations) - len(validated_relations)
    if filtered_count > 0:
        metrics.record_filtered_by_llm(filtered_count)

    if not validated_relations:
        # 关系全部被校验过滤：实体描述仍然保留，chunk 视为已完成
        if result.entities:
            cache.save_entities(result.entities)
            metrics.record_entity(len(result.entities))
        _log(f"  ⚠️ chunk #{idx}: 关系全部被校验过滤")
        cache.mark_chunk_completed(idx)
        metrics.record_chunk_success()
        return

    # 先把本 chunk 的实体入库 —— update_entity_canonical 才能同时
    # 修正本 chunk 实体行的 canonical 映射（此前先归一化后入库，映射丢失）
    if result.entities:
        cache.save_entities(result.entities)

    # 合并描述 + 规范化实体
    new_relations: list[RelModel] = []
    for rel in validated_relations:
        # 保存原始名称（用于在 result.entities 中查找 description）
        orig_subj = rel.subject
        orig_obj = rel.object

        # 实体规范化
        for orig_name in [orig_subj, orig_obj]:
            if orig_name in all_known_entities:
                continue
            canonical = canonicalizer.canonicalize(orig_name, all_known_entities)
            if canonical and canonical != orig_name:
                cache.update_entity_canonical(orig_name, canonical)
                metrics.record_canonicalized()
                if rel.subject == orig_name:
                    rel.subject = canonical
                if rel.object == orig_name:
                    rel.object = canonical
            else:
                # 真正的新实体：加入已知列表，后续 chunk 的归一化才能参照它
                all_known_entities.append(orig_name)

        # Description 合并（用原始名称在 result.entities 中查找）
        for orig_name, final_name in [(orig_subj, rel.subject), (orig_obj, rel.object)]:
            new_desc = ""
            for ent in result.entities:
                if ent.name == orig_name:
                    new_desc = ent.description
                    break

            if new_desc and len(new_desc) >= 5:
                existing_desc = global_entity_descriptions.get(final_name, "")
                merged_desc = merger.merge(existing_desc, new_desc, final_name)
                global_entity_descriptions[final_name] = merged_desc
                cache.update_entity_description(final_name, merged_desc)
                if existing_desc and merged_desc != existing_desc:
                    metrics.record_merged_description()

        # 去重
        if rel.triple_key not in global_relation_map:
            global_relation_map[rel.triple_key] = rel
            new_relations.append(rel)

    if new_relations:
        cache.save_relations(new_relations)

    cache.mark_chunk_completed(idx)
    metrics.record_chunk_success()
    metrics.record_entity(len(result.entities))
    metrics.record_relation(len(new_relations))

    _log(f"  ✅ chunk #{idx}: +{len(new_relations)} 条，累计 {len(global_relation_map)} 条")


def build_graph(
    documents: list[Document],
    llm,
    force_rebuild: bool = False,
    resume_from: Optional[int] = None,
) -> Optional[PropertyGraphIndex]:
    """从原始文档构建 PropertyGraph 索引并持久化到 Kuzu。

    Args:
        documents: 原始 Document 列表
        llm: 用于抽取的 LLM 实例
        force_rebuild: 是否强制删除缓存从头开始
        resume_from: 从第几个 chunk 开始续传

    Returns:
        PropertyGraphIndex 实例，失败返回 None
    """
    if not config.GRAPH_ENABLED:
        _log("阶段 5：知识图谱 — 未启用（GRAPH_ENABLED=False）")
        return None

    _log("阶段 5：构建知识图谱（PropertyGraph）")

    # ---- 1. 初始化路径 ----
    db_name = os.path.basename(config.GRAPH_DB_DIR)
    cache_dir = os.path.join(os.path.dirname(config.GRAPH_DB_DIR), "graph_cache")
    cache_db_path = os.path.join(cache_dir, f"{db_name}.db")
    schema_cache_path = os.path.join(cache_dir, f"{db_name}_schema.json")

    # ---- 2. 初始化 Schema ----
    schema = Schema.load_or_create(
        schema_cache_path,
        growth_threshold=config.GRAPH_SCHEMA_GROWTH_THRESHOLD,
    )

    # ---- 3. 计算 Build Fingerprint ----
    extract_model = getattr(config, "GRAPH_EXTRACT_LLM_PROVIDER", "unknown")
    validate_model = getattr(config, "GRAPH_VALIDATE_LLM_MODEL", "unknown")
    fingerprint = schema.compute_fingerprint(
        extract_prompt=config.GRAPH_EXTRACT_PROMPT,
        validate_prompt=config.GRAPH_VALIDATE_PROMPT,
        extract_model=extract_model,
        validate_model=validate_model,
        code_version="2.0",
    )

    # ---- 4. 初始化 SQLite 缓存 ----
    cache = GraphCache(cache_db_path)

    if force_rebuild:
        _log("  强制重建：清除旧缓存")
        cache.close()
        if os.path.exists(cache_db_path):
            os.remove(cache_db_path)
        cache = GraphCache(cache_db_path)
        cache.set_fingerprint(fingerprint)
    elif not cache.is_fingerprint_valid(fingerprint):
        _log("  Build Fingerprint 已变化，缓存失效，将重新抽取")
        cache.close()
        if os.path.exists(cache_db_path):
            os.remove(cache_db_path)
        cache = GraphCache(cache_db_path)
        cache.set_fingerprint(fingerprint)
    else:
        _log("  Build Fingerprint 匹配，可复用缓存")

    # ---- 5. 初始化各模块 ----
    extractor = Extractor(
        llm=llm,
        schema=schema,
        extract_prompt=config.GRAPH_EXTRACT_PROMPT,
        model_name=extract_model,
    )

    validate_llm = create_validate_llm()
    validator = Validator(
        validate_llm=validate_llm,
        validate_prompt=config.GRAPH_VALIDATE_PROMPT,
        model_name=validate_model,
        confidence_threshold=getattr(config, "GRAPH_VALIDATE_CONFIDENCE_THRESHOLD", 0.7),
        enabled=config.GRAPH_VALIDATE_ENABLED and validate_llm is not None,
    )
    if validate_llm and config.GRAPH_VALIDATE_ENABLED:
        _log(f"  启用 LLM 二次校验: {validate_model}（阈值: {validator._confidence_threshold}）")

    merger = DescriptionMerger(llm=llm, model_name=extract_model)
    canonicalizer = Canonicalizer(llm=llm, model_name=extract_model)
    metrics = MetricsCollector()

    # ---- 6. 准备 sections ----
    sections = _prepare_sections_for_graph(documents)
    _log(f"  按节抽取: {len(sections)} 个节")

    # 续传处理
    completed_ids = cache.get_completed_chunk_ids()
    if resume_from is not None:
        sections = [(i, t) for i, t in sections if i >= resume_from]
        _log(f"  从第 {resume_from} 节开始，剩余 {len(sections)} 个节")
    elif completed_ids and not force_rebuild:
        sections = [(i, t) for i, t in sections if i not in completed_ids]
        if len(sections) < len(documents):
            _log(f"  🔄 自动续传：{len(completed_ids)} 个已完成，跳过，剩余 {len(sections)} 个")

    # ---- 7. 全局状态 ----
    global_entity_descriptions: dict[str, str] = cache.get_all_entity_descriptions()
    global_relation_map: dict[tuple, RelModel] = cache.build_relation_map()
    all_known_entities: list[str] = list(global_entity_descriptions.keys())

    # ---- 8. 主循环：逐 chunk 处理（单 chunk 异常不中断整体构建） ----
    for idx, text in sections:
        _log(f"  📝 chunk #{idx} 抽取中...")

        # 检查缓存
        if cache.chunk_exists(idx, text):
            _log(f"  ⏭️ chunk #{idx} 已在缓存中，跳过")
            metrics.record_chunk_success()
            # 从缓存加载该 chunk 的关系
            cached_rels = cache.get_relations_by_chunk(idx)
            metrics.record_relation(len(cached_rels))
            continue

        try:
            _process_one_chunk(
                idx, text, cache, extractor, validator, merger, canonicalizer,
                metrics, global_entity_descriptions, global_relation_map,
                all_known_entities, _log,
            )
        except Exception as e:
            # 逐 chunk 兜底：一个 chunk 异常不能崩掉整个图构建
            logger.exception(f"chunk #{idx} 处理异常")
            _log(f"  ❌ chunk #{idx} 处理异常: {e}，跳过该 chunk 继续")
            cache.mark_chunk_failed(idx)
            metrics.record_chunk_failed()

    # 保存 Schema
    schema.save(schema_cache_path)

    # ---- 9. 检查是否有数据 ----
    all_relations = cache.get_all_relations()
    if not all_relations:
        _log("  未抽取到有效三元组，知识图谱为空")
        cache.close()
        return None

    _log(f"  去重后: {len(global_entity_descriptions)} 实体, {len(all_relations)} 关系")

    # ---- 10. 构建 Kuzu 图 ----
    # 清理旧数据库
    if os.path.exists(config.GRAPH_DB_DIR):
        shutil.rmtree(config.GRAPH_DB_DIR)
    os.makedirs(config.GRAPH_DB_DIR, exist_ok=True)

    db_path = os.path.join(config.GRAPH_DB_DIR, "kuzu.db")
    db = kuzu.Database(db_path)
    graph_store = KuzuPropertyGraphStore(db=db, use_vector_index=False)

    # 收集实体（按 canonical name 去重，使用真实 Schema Label）
    entity_nodes_map: dict[str, EntityNode] = {}
    for rel in all_relations:
        for name, node_type in [(rel.subject, rel.subject_type), (rel.object, rel.object_type)]:
            if name not in entity_nodes_map:
                desc = global_entity_descriptions.get(name, "")
                entity_nodes_map[name] = EntityNode(
                    name=name,
                    label=node_type if node_type != "Entity" else "Entity",
                    properties={"description": desc},
                )

    graph_store.upsert_nodes(list(entity_nodes_map.values()))
    _log(f"  已插入 {len(entity_nodes_map)} 个实体（含 Schema Label）")

    # 构建关系索引（O(1) 查找）
    relation_index: dict[tuple, RelModel] = {}
    for rel in all_relations:
        key = (rel.subject, rel.predicate, rel.object)
        if key not in relation_index:
            relation_index[key] = rel

    # 插入关系
    rel_objects = []
    for rel in all_relations:
        rel_objects.append(Relation(
            label=rel.predicate,
            source_id=rel.subject,
            target_id=rel.object,
            properties={
                "description": rel.description,
                "chunk_id": rel.chunk_id,
                "source_text": rel.source_text,
                "confidence": rel.confidence,
                "validated": 1 if rel.validated else 0,
            },
        ))

    graph_store.upsert_relations(rel_objects)
    _log(f"  已插入 {len(rel_objects)} 个关系")

    # 从已填充的数据库加载索引
    from llama_index.core import Settings
    _saved_embed = getattr(Settings, "_embed_model", Settings.embed_model)
    Settings.embed_model = MockEmbedding(embed_dim=1)
    try:
        index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
        )
    finally:
        Settings.embed_model = _saved_embed

    # ---- 11. 输出统计 ----
    metrics.log_summary()
    unknown_report = schema.get_unknown_type_report()
    if unknown_report:
        _log("  Schema 未知类型池:")
        for tname, tcount in sorted(unknown_report.items(), key=lambda x: -x[1]):
            _log(f"    {tname}: {tcount} 次")

    cache.close()

    # 写入阶段完成标记（staged_indexer 的完成检测以此为准）
    mark_stage_done(config.GRAPH_DB_DIR, num_entities=len(entity_nodes_map), num_relations=len(rel_objects))

    _log(f"阶段 5：知识图谱构建完成，持久化到 {config.GRAPH_DB_DIR}")
    _log(f"  共 {len(all_relations)} 条三元组，{len(entity_nodes_map)} 实体，{len(rel_objects)} 关系")

    return index


def load_graph() -> Optional[PropertyGraphIndex]:
    """从 Kuzu 加载已有的 PropertyGraph 索引。

    Returns:
        PropertyGraphIndex 实例，不存在或未启用返回 None
    """
    if not config.GRAPH_ENABLED:
        return None

    db_path = os.path.join(config.GRAPH_DB_DIR, "kuzu.db")
    if not os.path.exists(db_path):
        _log("  知识图谱数据库不存在，跳过加载")
        return None

    try:
        db = kuzu.Database(db_path)
        graph_store = KuzuPropertyGraphStore(db=db, use_vector_index=False)
        index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
        )
        _log(f"  已加载知识图谱（来自 {config.GRAPH_DB_DIR}）")
        return index
    except Exception as e:
        wal_path = db_path + ".wal"
        if os.path.exists(wal_path):
            try:
                os.remove(wal_path)
                db = kuzu.Database(db_path)
                graph_store = KuzuPropertyGraphStore(db=db, use_vector_index=False)
                index = PropertyGraphIndex.from_existing(
                    property_graph_store=graph_store,
                )
                _log(f"  已加载知识图谱（清理 WAL 后重试成功）")
                return index
            except Exception as e2:
                _log(f"  知识图谱加载失败（清理 WAL 后仍失败）: {e2}")
                return None
        _log(f"  知识图谱加载失败: {e}")
        return None