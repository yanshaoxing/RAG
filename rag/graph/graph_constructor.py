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
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import kuzu
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.schema import Document
from llama_index.graph_stores.kuzu import KuzuPropertyGraphStore

from rag import config, prompts
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


def _resolve_graph_models(llm) -> tuple[str, str]:
    """解析抽取/校验实际使用的模型名（用于 build fingerprint 与记账）。

    此前 extract_model 记的是 provider（"davy"）、validate_model 记的是
    ollama 分支的模型名 —— 换真实模型时缓存该失效而不失效，记账也失真。
    """
    extract_model = (getattr(llm, "model_name", None)
                     or getattr(llm, "model", None) or "unknown")
    if config.GRAPH_VALIDATE_LLM_PROVIDER == "aliyun":
        validate_model = config.ALIYUN_VALIDATE_MODEL
    elif config.GRAPH_VALIDATE_LLM_PROVIDER == "davy":
        validate_model = config.GRAPH_VALIDATE_DAVY_MODEL
    else:
        validate_model = config.GRAPH_VALIDATE_LLM_MODEL
    return str(extract_model), str(validate_model)


def _extract_and_validate(
    idx: int, text: str, extractor: Extractor, validator: Validator,
):
    """worker 阶段：LLM 抽取 + 校验（不触碰缓存与全局记账，可安全并发）。

    返回 (ChunkResult 或 None, 校验后的关系列表, 异常或 None)。
    异常在 worker 内捕获，由主线程统一按"该 chunk 失败"处理。
    """
    try:
        result = extractor.extract(text, idx)
        if result is None or result.is_empty:
            return result, [], None
        validated = validator.validate(result.relations, text)
        return result, validated, None
    except Exception as e:  # noqa: BLE001 —— 单 chunk 异常不中断整体构建
        return None, [], e


def _run_pipelined(
    items: list,
    worker_fn: Callable,
    handle_fn: Callable,
    max_workers: int,
) -> None:
    """有界流水线：worker 线程并发跑 worker_fn(item)（LLM 重活），
    主线程严格按提交顺序执行 handle_fn(item, outcome)（缓存/记账，无需加锁）。

    in-flight 上限 2×max_workers，避免 worker 跑得太远 —— 结果只有经
    handle_fn 落缓存后才算数，中断时最多丢弃在途的少量抽取结果。
    worker_fn 必须自行捕获异常（把异常作为 outcome 的一部分返回）。
    """
    if not items:
        return
    max_workers = max(1, max_workers)
    window = max_workers * 2

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        in_flight: deque = deque()
        item_iter = iter(items)

        def _submit_next() -> bool:
            try:
                item = next(item_iter)
            except StopIteration:
                return False
            in_flight.append((item, executor.submit(worker_fn, item)))
            return True

        for _ in range(window):
            if not _submit_next():
                break

        while in_flight:
            item, fut = in_flight.popleft()
            outcome = fut.result()
            _submit_next()          # 先补位再处理，保持 worker 满载
            handle_fn(item, outcome)


def _book_chunk_result(
    idx: int,
    text: str,
    result,
    validated_relations: list[RelModel],
    cache: GraphCache,
    merger: DescriptionMerger,
    canonicalizer: Canonicalizer,
    metrics: MetricsCollector,
    global_entity_descriptions: dict[str, str],
    global_relation_map: dict[tuple, RelModel],
    all_known_entities: list[str],
    _log,
) -> None:
    """主线程记账阶段：归一化 → 描述合并 → 入缓存（串行执行，保证记账确定性）。"""
    cache.mark_chunk_processing(idx, text)

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

    # ---- 实体规范化：每个端点名只处理一次 ----
    # （此前逐关系逐端点处理，同名实体出现在多条关系中时
    #   canonicalize/merge LLM 调用成倍放大）
    endpoint_names: list[str] = []
    for rel in validated_relations:
        for name in (rel.subject, rel.object):
            if name not in endpoint_names:
                endpoint_names.append(name)

    name_map: dict[str, str] = {}
    for orig_name in endpoint_names:
        if orig_name in all_known_entities:
            name_map[orig_name] = orig_name
            continue
        canonical = canonicalizer.canonicalize(orig_name, all_known_entities)
        if canonical and canonical != orig_name:
            cache.update_entity_canonical(orig_name, canonical)
            metrics.record_canonicalized()
            name_map[orig_name] = canonical
        else:
            # 真正的新实体：加入已知列表，后续 chunk 的归一化才能参照它
            name_map[orig_name] = orig_name
            all_known_entities.append(orig_name)

    for rel in validated_relations:
        rel.subject = name_map.get(rel.subject, rel.subject)
        rel.object = name_map.get(rel.object, rel.object)

    # ---- Description 合并：每个实体每 chunk 最多一次 LLM 调用 ----
    entity_desc = {e.name: e.description for e in result.entities}
    for orig_name in endpoint_names:
        new_desc = entity_desc.get(orig_name, "")
        if not new_desc or len(new_desc) < 5:
            continue
        final_name = name_map[orig_name]
        existing_desc = global_entity_descriptions.get(final_name, "")
        merged_desc = merger.merge(existing_desc, new_desc, final_name)
        global_entity_descriptions[final_name] = merged_desc
        cache.update_entity_description(final_name, merged_desc)
        if existing_desc and merged_desc != existing_desc:
            metrics.record_merged_description()

    # ---- 去重 + 入缓存 ----
    new_relations: list[RelModel] = []
    for rel in validated_relations:
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
    resume_from: int | None = None,
) -> PropertyGraphIndex | None:
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

    # ---- 3. 计算 Build Fingerprint（使用实际模型名，换模型必须使缓存失效） ----
    extract_model, validate_model = _resolve_graph_models(llm)
    fingerprint = schema.compute_fingerprint(
        extract_prompt=prompts.GRAPH_EXTRACT_PROMPT,
        validate_prompt=prompts.GRAPH_VALIDATE_PROMPT,
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
        extract_prompt=prompts.GRAPH_EXTRACT_PROMPT,
        model_name=extract_model,
    )

    validate_llm = create_validate_llm()
    validator = Validator(
        validate_llm=validate_llm,
        validate_prompt=prompts.GRAPH_VALIDATE_PROMPT,
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
    # 已知实体 = 有描述的 ∪ 全部 canonical 名（无描述实体也要参与归一化参照）
    all_known_entities: list[str] = list(dict.fromkeys(
        list(global_entity_descriptions.keys()) + cache.get_all_entity_names()
    ))

    # ---- 8. 主循环：有界流水线 ----
    # worker 线程并发跑「抽取 + 校验」（每 chunk 独立、无共享状态），
    # 主线程严格按提交顺序做「归一化 + 合并 + 入缓存」（记账确定、SQLite 免锁）。
    # 注意：主线程记账中的 merge/canonicalize 也是 LLM 调用，
    # 实际 LLM 并发最坏 = GRAPH_EXTRACT_MAX_CONCURRENCY + 1（Davy 429 有重试兜底）。

    # 先过滤缓存已完成的 chunk
    pending: list[tuple[int, str]] = []
    for idx, text in sections:
        if cache.chunk_exists(idx, text):
            _log(f"  ⏭️ chunk #{idx} 已在缓存中，跳过")
            metrics.record_chunk_success()
            cached_rels = cache.get_relations_by_chunk(idx)
            metrics.record_relation(len(cached_rels))
        else:
            pending.append((idx, text))

    if pending:
        max_workers = max(1, config.GRAPH_EXTRACT_MAX_CONCURRENCY)
        _log(f"  待抽取 {len(pending)} 个节（抽取/校验并发={max_workers}，记账串行）")

        def _worker(item: tuple[int, str]):
            idx, text = item
            return _extract_and_validate(idx, text, extractor, validator)

        def _handle(item: tuple[int, str], outcome) -> None:
            idx, text = item
            result, validated_relations, error = outcome
            _log(f"  📝 chunk #{idx} 记账中...")
            if error is not None:
                logger.error(f"chunk #{idx} 抽取/校验异常: {error}")
                _log(f"  ❌ chunk #{idx} 抽取/校验异常: {error}，跳过该 chunk 继续")
                cache.mark_chunk_processing(idx, text)
                cache.mark_chunk_failed(idx)
                metrics.record_chunk_failed()
                return
            try:
                _book_chunk_result(
                    idx, text, result, validated_relations, cache, merger,
                    canonicalizer, metrics, global_entity_descriptions,
                    global_relation_map, all_known_entities, _log,
                )
            except Exception as e:
                # 逐 chunk 兜底：一个 chunk 异常不能崩掉整个图构建
                logger.exception(f"chunk #{idx} 记账异常")
                _log(f"  ❌ chunk #{idx} 记账异常: {e}，跳过该 chunk 继续")
                cache.mark_chunk_failed(idx)
                metrics.record_chunk_failed()

        _run_pipelined(pending, _worker, _handle, max_workers)

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


def load_graph() -> PropertyGraphIndex | None:
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
                _log("  已加载知识图谱（清理 WAL 后重试成功）")
                return index
            except Exception as e2:
                _log(f"  知识图谱加载失败（清理 WAL 后仍失败）: {e2}")
                return None
        _log(f"  知识图谱加载失败: {e}")
        return None