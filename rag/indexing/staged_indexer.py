"""
分阶段索引构建管线 —— 每个阶段的产物持久化到独立文件夹。
若中途失败，只需删除未完成阶段的文件夹即可从中断处继续。

阶段（依赖图见下方 _STAGES，重建按依赖闭包传播而非线性区间）：
  1. 分块       → chunks/     （依赖 raw/）
  2. 摘要树     → summary_tree/（依赖 chunks）
  3. BM25 索引  → bm25/        （依赖 chunks + summary）
  4. 向量索引   → vector/      （依赖 chunks + summary）
  5. 知识图谱   → graph_db/    （只依赖 raw/，与检索阶段互不连带）
"""

import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from llama_index.core.indices.property_graph import PropertyGraphIndex

import faiss
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.faiss import FaissVectorStore

from rag import config
from rag.graph.graph_constructor import build_graph, load_graph
from rag.indexing.embedding_checkpoint import clear_checkpoint, embed_nodes_with_checkpoint
from rag.ingestion.preprocessor import (
    CHUNK_EXCLUDED_META_KEYS,
    create_chunking_pipeline,
    load_documents,
)
from rag.summarization.summary_tree import SUMMARY_EXCLUDED_META_KEYS, build_summary_tree
from rag.utils.files import atomic_write_json, mark_stage_done, read_stage_info, stage_complete
from rag.utils.text import tokenize_for_bm25

logger = logging.getLogger(__name__)


def _log(msg: str) -> None:
    """管线日志：经标准 logging 输出，入口层用 capture_pipeline_logs 捕获。"""
    logger.info(msg)


# 阶段完成标记 / 原子写入工具（与 graph_constructor 共用）
_atomic_write_json = atomic_write_json
_mark_stage_done = mark_stage_done
_stage_complete = stage_complete


# ======================== 序列化 / 反序列化 ========================

def _serialize_nodes(nodes: list) -> list[dict]:
    """将节点列表序列化为 JSON 兼容的字典列表。"""
    result = []
    for node in nodes:
        result.append({
            "node_id": node.node_id,
            "text": node.text,
            "metadata": dict(node.metadata),
        })
    return result


def _deserialize_nodes(data: list[dict]) -> list:
    """从 JSON 字典列表反序列化为节点列表（尽力还原为 TextNode）。

    序列化不保存元数据排除键，加载时必须重新应用 ——
    否则 section_path 等键会随 metadata 拼进嵌入输入与 LLM 上下文。
    """
    nodes = []
    for item in data:
        node = TextNode(
            text=item["text"],
            node_id=item.get("node_id", ""),
            metadata=item.get("metadata", {}),
            excluded_embed_metadata_keys=list(CHUNK_EXCLUDED_META_KEYS),
            excluded_llm_metadata_keys=list(CHUNK_EXCLUDED_META_KEYS),
        )
        nodes.append(node)
    return nodes


def _serialize_summary_docs(docs: list[Document]) -> list[dict]:
    """序列化摘要 Document 列表。"""
    result = []
    for doc in docs:
        result.append({
            "doc_id": doc.doc_id or "",
            "text": doc.text or "",
            "metadata": dict(doc.metadata),
        })
    return result


def _deserialize_summary_docs(data: list[dict]) -> list[Document]:
    """反序列化摘要 Document 列表。

    重新应用摘要元数据排除键（序列化不保存）——否则 original_text（≤8192 字）
    与 child_ids UUID 列表会随 metadata 主导嵌入输入并污染 QA prompt。
    """
    docs = []
    for item in data:
        doc = Document(
            text=item.get("text", ""),
            doc_id=item.get("doc_id", ""),
            metadata=item.get("metadata", {}),
            excluded_embed_metadata_keys=list(SUMMARY_EXCLUDED_META_KEYS),
            excluded_llm_metadata_keys=list(SUMMARY_EXCLUDED_META_KEYS),
        )
        docs.append(doc)
    return docs


# ======================== 各阶段实现 ========================

def _stage_chunk() -> list:
    """阶段 1：文档分块 → chunks/"""
    # 清理旧 chunks（防止残留不完整数据）
    if os.path.exists(config.CHUNKS_DIR):
        shutil.rmtree(config.CHUNKS_DIR)
    os.makedirs(config.CHUNKS_DIR, exist_ok=True)

    # 加载文档
    print("阶段 1：加载文档...", flush=True)
    raw_documents = load_documents()
    _log(f"阶段 1：加载文档（共 {len(raw_documents)} 个章节）")

    # 分块
    print("阶段 1：开始分块...", flush=True)
    pipeline = create_chunking_pipeline()
    nodes = list(pipeline.run(documents=raw_documents))
    _log(f"阶段 1：分块 → 共 {len(nodes)} 个节点")

    if len(nodes) == 0:
        raise RuntimeError("分块后节点数为 0，请检查文档内容或分块配置")

    # 持久化（原子写入 + 完成标记）
    chunks_path = os.path.join(config.CHUNKS_DIR, "chunks.json")
    _atomic_write_json(chunks_path, _serialize_nodes(nodes))
    _mark_stage_done(config.CHUNKS_DIR, num_nodes=len(nodes))
    _log(f"阶段 1：chunks 已持久化到 {chunks_path}")

    return nodes


def _load_chunks() -> list:
    """从 chunks/ 加载分块节点。"""
    chunks_path = os.path.join(config.CHUNKS_DIR, "chunks.json")
    with open(chunks_path, encoding="utf-8") as f:
        data = json.load(f)
    nodes = _deserialize_nodes(data)
    _log(f"  已加载 {len(nodes)} 个 chunk 节点")
    return nodes


def _stage_summary(chunk_nodes: list) -> tuple[list, dict]:
    """阶段 2：摘要树构建 → summary_tree/"""
    print("阶段 2：构建摘要树...", flush=True)
    _log("阶段 2：构建摘要树")

    summary_docs, summary_meta_map = build_summary_tree(chunk_nodes)
    summary_docs = list(summary_docs)

    # 确保目录存在（stage 开始前可能被清理过）
    os.makedirs(config.SUMMARY_TREE_DIR, exist_ok=True)

    # 持久化（原子写入，两个文件都写完才落完成标记）
    map_path = os.path.join(config.SUMMARY_TREE_DIR, "summary_meta_map.json")
    _atomic_write_json(map_path, summary_meta_map)
    _log(f"  摘要树元数据已持久化到 {map_path}")

    summary_nodes_path = os.path.join(config.SUMMARY_TREE_DIR, "summary_nodes.json")
    _atomic_write_json(summary_nodes_path, _serialize_summary_docs(summary_docs))
    _log(f"  摘要节点已持久化到 {summary_nodes_path}（共 {len(summary_docs)} 个）")

    _mark_stage_done(config.SUMMARY_TREE_DIR, num_summary_docs=len(summary_docs))
    _log(f"阶段 2：摘要树构建完成，新增 {len(summary_docs)} 个摘要节点")
    return summary_docs, summary_meta_map


def _load_summary() -> tuple[list, dict]:
    """从 summary_tree/ 加载摘要节点和元数据。"""
    # 完成标记保证两个文件都存在；缺失说明产物损坏，显式报错而非静默返回空
    map_path = os.path.join(config.SUMMARY_TREE_DIR, "summary_meta_map.json")
    summary_nodes_path = os.path.join(config.SUMMARY_TREE_DIR, "summary_nodes.json")
    for p in (map_path, summary_nodes_path):
        if not os.path.exists(p):
            raise RuntimeError(
                f"摘要树产物缺失: {p}。"
                f"请删除 {config.SUMMARY_TREE_DIR} 后重新运行以重建该阶段。"
            )

    with open(map_path, encoding="utf-8") as f:
        summary_meta_map = json.load(f)
    _log(f"  已加载 {len(summary_meta_map)} 条摘要树元数据")

    with open(summary_nodes_path, encoding="utf-8") as f:
        data = json.load(f)
    summary_docs = _deserialize_summary_docs(data)
    _log(f"  已加载 {len(summary_docs)} 个摘要节点")

    return summary_docs, summary_meta_map


def _stage_bm25(all_nodes: list) -> BM25Retriever:
    """阶段 3：BM25 索引构建 → bm25/"""
    print(f"阶段 3：构建 BM25 索引（共 {len(all_nodes)} 个节点）...", flush=True)
    _log(f"阶段 3：构建 BM25 索引（共 {len(all_nodes)} 个节点）")

    t_start = datetime.now()
    bm25_nodes = []
    for node in all_nodes:
        section = node.metadata.get("section", "")
        combined = f"{section} {node.text}" if section else node.text
        # 分词版本用于 BM25 索引，原始文本存入 metadata 供后续恢复。
        # 注意：摘要节点已在 summary_tree 中把"原始出处文本"写入 original_text
        # 供 rerank/展示使用，此处不能覆盖（setdefault 保留已有值）
        meta = dict(node.metadata)
        meta.setdefault("original_text", combined)
        # original_text 仅供检索后恢复正文，绝不能拼进 LLM 上下文
        # （BM25 命中的节点会直接进入 QA prompt，不排除等于正文翻倍）
        n = node.model_copy(update={
            "text": tokenize_for_bm25(combined),
            "metadata": meta,
            "excluded_embed_metadata_keys": sorted(
                set(node.excluded_embed_metadata_keys or []) | {"original_text"}),
            "excluded_llm_metadata_keys": sorted(
                set(node.excluded_llm_metadata_keys or []) | {"original_text"}),
        })
        bm25_nodes.append(n)

    bm25_retriever = BM25Retriever.from_defaults(nodes=bm25_nodes)
    bm25_retriever.similarity_top_k = config.RETRIEVAL_TOP_K

    os.makedirs(config.BM25_DIR, exist_ok=True)
    bm25_retriever.persist(config.BM25_DIR)
    _mark_stage_done(config.BM25_DIR, num_nodes=len(bm25_nodes))

    t_elapsed = (datetime.now() - t_start).total_seconds()
    _log(f"  BM25 索引构建完成，持久化到 {config.BM25_DIR}，耗时 {t_elapsed:.1f}s")

    return bm25_retriever


def _load_bm25() -> BM25Retriever:
    """从 bm25/ 加载 BM25 检索器。"""
    bm25_retriever = BM25Retriever.from_persist_dir(config.BM25_DIR)
    bm25_retriever.similarity_top_k = config.RETRIEVAL_TOP_K
    _log("  BM25 索引已加载")
    return bm25_retriever


def _stage_vector(all_nodes: list) -> VectorStoreIndex:
    """阶段 4：向量索引构建 → vector/（embedding 分段落盘，中断可续传）"""
    _seg = config.EMBED_CHECKPOINT_SEGMENT_NODES
    _num_segments = (len(all_nodes) + _seg - 1) // _seg
    print(f"阶段 4：构建向量索引（共 {len(all_nodes)} 个节点，每段 {_seg} 个，"
          f"{_num_segments} 段，断点可续传）...", flush=True)
    _log(f"阶段 4：构建向量索引（共 {len(all_nodes)} 个节点，"
         f"每段 {_seg} 个，{_num_segments} 段）")

    t_start = datetime.now()

    os.makedirs(config.FAISS_PERSIST_DIR, exist_ok=True)

    # 清理旧索引文件，防止残留旧维度索引导致维度不匹配
    old_faiss = os.path.join(config.FAISS_PERSIST_DIR, "index.faiss")
    if os.path.exists(old_faiss):
        print(f"  清理旧 HNSW 索引文件: {old_faiss}", flush=True)
        shutil.rmtree(config.FAISS_PERSIST_DIR, ignore_errors=True)
        os.makedirs(config.FAISS_PERSIST_DIR, exist_ok=True)

    # 必须用 IndexHNSWFlat 子类：基类 IndexHNSW 没有序列化编码，
    # write_index 持久化时会报 "'h != 0' failed"
    faiss_index = faiss.IndexHNSWFlat(
        config.EMBED_VECTOR_DIM, config.HNSW_M, faiss.METRIC_INNER_PRODUCT,
    )
    faiss_index.hnsw.efConstruction = config.HNSW_EF_CONSTRUCTION
    faiss_index.hnsw.efSearch = config.HNSW_EF_SEARCH
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 分段计算 embedding 并落盘（语料的 data/embed_cache/），中断后续跑只补缺失段；
    # 节点已带 embedding，VectorStoreIndex 不会再调用 embed 模型
    embed_nodes_with_checkpoint(all_nodes, Settings.embed_model, label="向量索引")
    index = VectorStoreIndex(all_nodes, storage_context=storage_context)
    index.storage_context.persist(persist_dir=config.PERSIST_DIR)
    # 记下建索引所用的嵌入模型，加载时比对（换模型后旧索引不可复用，见 _load_vector）
    _mark_stage_done(config.PERSIST_DIR, num_nodes=len(all_nodes),
                     embed_model=config.ACTIVE_EMBED_MODEL_NAME,
                     vector_dim=config.EMBED_VECTOR_DIM)
    clear_checkpoint()  # 向量索引已持久化，embedding 缓存不再需要

    t_elapsed = (datetime.now() - t_start).total_seconds()
    _log(f"  向量索引构建完成，持久化到 {config.PERSIST_DIR}，耗时 {t_elapsed:.1f}s")

    return index


def _check_embed_model_match() -> None:
    """比对向量索引的构建模型与当前嵌入模型，不一致则显式报错。

    ⚠️ 维度相同不代表可复用：换嵌入模型后向量空间不同，旧索引与新查询向量算出的
    相似度毫无意义，且不会报任何错 —— 静默给出垃圾检索结果。此处让它显式失败。
    （旧版本产物没有 embed_model 字段，此时跳过检查以兼容。）
    """
    info = read_stage_info(config.PERSIST_DIR)
    built_with = info.get("embed_model")
    if built_with and built_with != config.ACTIVE_EMBED_MODEL_NAME:
        raise RuntimeError(
            f"向量索引由嵌入模型 {built_with!r} 构建，当前配置为 "
            f"{config.ACTIVE_EMBED_MODEL_NAME!r}。两者向量空间不同，索引不可复用"
            f"（即使维度相同）。请删除 {config.PERSIST_DIR} 后重新运行以重建向量阶段。"
        )


def _load_vector() -> VectorStoreIndex:
    """从 vector/ 加载向量索引（FAISS HNSW）。"""
    t_start = datetime.now()

    _check_embed_model_match()

    faiss_index_path = os.path.join(config.PERSIST_DIR, "default__vector_store.json")
    if not os.path.exists(faiss_index_path):
        # 完成标记存在但索引文件缺失 → 产物损坏，显式报错而非新建空索引
        raise RuntimeError(
            f"向量索引文件缺失: {faiss_index_path}。"
            f"请删除 {config.PERSIST_DIR} 后重新运行以重建该阶段。"
        )
    faiss_index = faiss.read_index(faiss_index_path)

    if hasattr(faiss_index, 'hnsw'):
        faiss_index.hnsw.efSearch = config.HNSW_EF_SEARCH

    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(
        persist_dir=config.PERSIST_DIR,
        vector_store=vector_store,
    )
    index = load_index_from_storage(storage_context)
    t_elapsed = (datetime.now() - t_start).total_seconds()
    _log(f"  向量索引已加载（共 {len(index.docstore.docs)} 个文档节点），耗时 {t_elapsed:.2f}s")
    return index


# ======================== 阶段依赖图 ========================
#
# 阶段不再是线性链，而是带依赖声明的有向图（raw/ 原文是所有构建的隐含源，不列为阶段）：
#
#     raw ─┬─▶ chunks ─▶ summary ─┬─▶ bm25
#          │                      └─▶ vector
#          └─▶ graph
#
# 重建按【依赖闭包】传播：某阶段重建 → 依赖它的下游阶段一并重建；不依赖它的阶段
# （尤其图谱——只依赖 raw/）不受连带。这修掉了旧线性链「删向量却连带重建图谱」
# 的浪费（小语料几秒，全书是数小时 + 数元）。bm25 与 vector 互不依赖，也不再互相牵连。


@dataclass(frozen=True)
class _Stage:
    key: str                    # 稳定标识（供 --rebuild 使用，勿改）
    name: str                   # 中文显示名
    path_attr: str              # config 上的动态路径属性名（随激活语料变化）
    deps: tuple[str, ...]       # 直接依赖的上游阶段 key（不含隐含源 raw/）

    @property
    def path(self) -> str:
        return getattr(config, self.path_attr)


# 拓扑序排列（上游在前），_compute_dirty 一次线性扫描即可传播 dirty
_STAGES: list[_Stage] = [
    _Stage("chunks",  "分块",       "CHUNKS_DIR",       ()),
    _Stage("summary", "摘要树",     "SUMMARY_TREE_DIR", ("chunks",)),
    _Stage("bm25",    "BM25 索引",  "BM25_DIR",         ("chunks", "summary")),
    _Stage("vector",  "向量索引",   "PERSIST_DIR",      ("chunks", "summary")),
    _Stage("graph",   "知识图谱",   "GRAPH_DB_DIR",     ()),
]
_STAGE_BY_KEY: dict[str, _Stage] = {s.key: s for s in _STAGES}
STAGE_KEYS: list[str] = [s.key for s in _STAGES]  # 供 CLI --rebuild 的 choices


def _stage_enabled(key: str) -> bool:
    """阶段是否启用（受 config 功能开关控制）。"""
    if key == "summary":
        return config.SUMMARY_TREE_ENABLED
    if key == "graph":
        return config.GRAPH_ENABLED
    return True


def _compute_dirty() -> dict[str, bool]:
    """按依赖闭包判定各阶段是否需要（重）建。

    某阶段 dirty ⇔ 它已启用，且（自身产物缺失/中断残留 或 任一上游阶段 dirty）。
    _STAGES 已按拓扑序排列，故一次线性扫描即可把 dirty 沿依赖边传播下去。
    未启用阶段恒 False（既不参与完成度扫描，也不向下游传播）。
    """
    dirty: dict[str, bool] = {}
    for st in _STAGES:
        if not _stage_enabled(st.key):
            dirty[st.key] = False
            continue
        d = not _stage_complete(st.path)
        if any(dirty.get(dep) for dep in st.deps):
            d = True
        dirty[st.key] = d
    return dirty


def _downstream_closure(key: str) -> list[str]:
    """key 阶段自身 + 所有（传递）依赖它的下游阶段 key，按拓扑序返回。"""
    closure = {key}
    changed = True
    while changed:
        changed = False
        for st in _STAGES:
            if st.key not in closure and any(dep in closure for dep in st.deps):
                closure.add(st.key)
                changed = True
    return [s.key for s in _STAGES if s.key in closure]


def plan_rebuild(key: str) -> list[_Stage]:
    """给定阶段 key，返回重建将波及的阶段（自身 + 下游依赖闭包，拓扑序）。

    不做删除，仅用于向用户展示「将删除/重建哪些阶段」。未知 key 抛 ValueError。
    """
    if key not in _STAGE_BY_KEY:
        raise ValueError(f"未知阶段 {key!r}，可选：{', '.join(STAGE_KEYS)}")
    return [_STAGE_BY_KEY[k] for k in _downstream_closure(key)]


def rebuild_stages(key: str, apply: bool = False) -> list[_Stage]:
    """删除阶段 key 及其下游闭包的产物目录（apply=False 仅返回计划、不删）。

    删除后由入口重新运行 get_or_build_index() 检测缺失并重建。
    """
    stages = plan_rebuild(key)
    if apply:
        for st in stages:
            if os.path.exists(st.path):
                shutil.rmtree(st.path, ignore_errors=True)
                _log(f"  已删除阶段产物: {st.name} — {st.path}")
    return stages


# ======================== 管线控制器 ========================

def get_or_build_index() -> tuple[VectorStoreIndex, BM25Retriever, dict | None, Optional["PropertyGraphIndex"]]:
    """分阶段索引管线控制器（依赖图版）。

    按依赖闭包判定各阶段是否需要重建：删除全部 dirty 阶段目录，重建 dirty 阶段，
    其余从磁盘加载。图谱只依赖 raw/，故检索阶段的重建不会连带它。

    返回: (向量索引, BM25 检索器, 摘要树元数据映射, 知识图谱索引或 None)
    """
    dirty = _compute_dirty()

    # ---- 删除所有 dirty 阶段的目录（仅这些；依赖闭包外的产物原样保留） ----
    to_rebuild = [st for st in _STAGES if dirty[st.key]]
    if to_rebuild:
        names = "、".join(st.name for st in to_rebuild)
        print(f"检测到需（重）建阶段：{names}（产物缺失/中断残留或上游变动）...", flush=True)
        _log(f"需（重）建阶段：{names}")
        for st in to_rebuild:
            if os.path.exists(st.path):
                shutil.rmtree(st.path, ignore_errors=True)
                _log(f"  已清理旧产物: {st.path}")
    else:
        print("所有阶段产物已就绪，从磁盘加载...", flush=True)
        _log("所有阶段产物已就绪，从磁盘加载")

    # ---- 分块（自身不是返回值，仅作下游构建的输入；仅在需要时加载） ----
    need_chunks = any(dirty[k] for k in ("summary", "bm25", "vector"))
    chunk_nodes: list | None = None
    if dirty["chunks"]:
        chunk_nodes = _stage_chunk()
    elif need_chunks:
        chunk_nodes = _load_chunks()

    # ---- 摘要树（meta_map 是返回值，启用则必 build/load） ----
    summary_docs: list = []
    summary_meta_map: dict | None = None
    if config.SUMMARY_TREE_ENABLED:
        if dirty["summary"]:
            summary_docs, summary_meta_map = _stage_summary(chunk_nodes)
        else:
            summary_docs, summary_meta_map = _load_summary()
    else:
        _log("摘要树未启用，跳过")

    # ---- BM25 / 向量的输入节点集（仅任一需重建时才拼装） ----
    if dirty["bm25"] or dirty["vector"]:
        all_nodes = list(chunk_nodes) if chunk_nodes else []
        if summary_docs:
            all_nodes = all_nodes + list(summary_docs)

    # ---- BM25 索引 ----
    bm25_retriever = _stage_bm25(all_nodes) if dirty["bm25"] else _load_bm25()

    # ---- 向量索引 ----
    vector_index = _stage_vector(all_nodes) if dirty["vector"] else _load_vector()

    # ---- 知识图谱（按节抽取，只依赖 raw/，与检索阶段互不连带） ----
    graph_index = None  # Optional[PropertyGraphIndex]
    if config.GRAPH_ENABLED:
        if dirty["graph"]:
            raw_documents = load_documents()
            graph_index = build_graph(raw_documents, Settings.llm)
        else:
            graph_index = load_graph()
    else:
        _log("知识图谱未启用，跳过")

    return vector_index, bm25_retriever, summary_meta_map, graph_index