"""
分阶段索引构建管线 —— 每个阶段的产物持久化到独立文件夹。
若中途失败，只需删除未完成阶段的文件夹即可从中断处继续。

阶段：
  1. 分块       → chunks/
  2. 摘要树     → summary_tree/
  3. BM25 索引  → bm25/
  4. 向量索引   → vector/
"""

import json
import logging
import os
import shutil
from datetime import datetime
from typing import Optional, Tuple, List

import faiss
import jieba
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.faiss import FaissVectorStore

from rag import config
from rag.ingestion.preprocessor import load_documents, create_chunking_pipeline
from rag.indexing.embedding_progress import ProgressOllamaEmbedding
from rag.summarization.summary_tree import build_summary_tree
from rag.graph.graph_constructor import build_graph, load_graph

logger = logging.getLogger(__name__)


def tokenize_for_bm25(text: str) -> str:
    """中文分词后空格连接，供 BM25 索引构建和检索使用。"""
    return " ".join(jieba.cut(text))


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
    """从 JSON 字典列表反序列化为节点列表（尽力还原为 TextNode）。"""
    nodes = []
    for item in data:
        node = TextNode(
            text=item["text"],
            node_id=item.get("node_id", ""),
            metadata=item.get("metadata", {}),
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
    """反序列化摘要 Document 列表。"""
    docs = []
    for item in data:
        doc = Document(
            text=item.get("text", ""),
            doc_id=item.get("doc_id", ""),
            metadata=item.get("metadata", {}),
        )
        docs.append(doc)
    return docs


# ======================== 各阶段实现 ========================

def _stage_chunk(log_list: Optional[list] = None) -> list:
    """阶段 1：文档分块 → chunks/"""
    def _log(msg: str):
        logger.info(msg)
        if log_list is not None:
            log_list.append(msg)

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

    # 持久化
    chunks_path = os.path.join(config.CHUNKS_DIR, "chunks.json")
    serialized = _serialize_nodes(nodes)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=2)
    _log(f"阶段 1：chunks 已持久化到 {chunks_path}")

    return nodes


def _load_chunks(log_list: Optional[list] = None) -> list:
    """从 chunks/ 加载分块节点。"""
    def _log(msg: str):
        logger.info(msg)
        if log_list is not None:
            log_list.append(msg)

    chunks_path = os.path.join(config.CHUNKS_DIR, "chunks.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    nodes = _deserialize_nodes(data)
    _log(f"  已加载 {len(nodes)} 个 chunk 节点")
    return nodes


def _stage_summary(chunk_nodes: list, log_list: Optional[list] = None) -> Tuple[list, dict]:
    """阶段 2：摘要树构建 → summary_tree/"""
    def _log(msg: str):
        logger.info(msg)
        if log_list is not None:
            log_list.append(msg)

    print("阶段 2：构建摘要树...", flush=True)
    _log("阶段 2：构建摘要树")

    summary_docs, summary_meta_map = build_summary_tree(chunk_nodes, log_list)
    summary_docs = list(summary_docs)

    # 确保目录存在（stage 开始前可能被清理过）
    os.makedirs(config.SUMMARY_TREE_DIR, exist_ok=True)

    # 持久化 summary_meta_map
    map_path = os.path.join(config.SUMMARY_TREE_DIR, "summary_meta_map.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(summary_meta_map, f, ensure_ascii=False, indent=2)
    _log(f"  摘要树元数据已持久化到 {map_path}")

    # 持久化 summary_nodes
    summary_nodes_path = os.path.join(config.SUMMARY_TREE_DIR, "summary_nodes.json")
    serialized = _serialize_summary_docs(summary_docs)
    with open(summary_nodes_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=2)
    _log(f"  摘要节点已持久化到 {summary_nodes_path}（共 {len(summary_docs)} 个）")

    _log(f"阶段 2：摘要树构建完成，新增 {len(summary_docs)} 个摘要节点")
    return summary_docs, summary_meta_map


def _load_summary(log_list: Optional[list] = None) -> Tuple[list, dict]:
    """从 summary_tree/ 加载摘要节点和元数据。"""
    def _log(msg: str):
        logger.info(msg)
        if log_list is not None:
            log_list.append(msg)

    # 加载 summary_meta_map
    map_path = os.path.join(config.SUMMARY_TREE_DIR, "summary_meta_map.json")
    if os.path.exists(map_path):
        with open(map_path, "r", encoding="utf-8") as f:
            summary_meta_map = json.load(f)
        _log(f"  已加载 {len(summary_meta_map)} 条摘要树元数据")
    else:
        summary_meta_map = {}

    # 加载 summary_nodes
    summary_nodes_path = os.path.join(config.SUMMARY_TREE_DIR, "summary_nodes.json")
    if os.path.exists(summary_nodes_path):
        with open(summary_nodes_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        summary_docs = _deserialize_summary_docs(data)
        _log(f"  已加载 {len(summary_docs)} 个摘要节点")
    else:
        summary_docs = []

    return summary_docs, summary_meta_map


def _stage_bm25(all_nodes: list, log_list: Optional[list] = None) -> BM25Retriever:
    """阶段 3：BM25 索引构建 → bm25/"""
    def _log(msg: str):
        logger.info(msg)
        if log_list is not None:
            log_list.append(msg)

    print(f"阶段 3：构建 BM25 索引（共 {len(all_nodes)} 个节点）...", flush=True)
    _log(f"阶段 3：构建 BM25 索引（共 {len(all_nodes)} 个节点）")

    t_start = datetime.now()
    bm25_nodes = []
    for node in all_nodes:
        section = node.metadata.get("section", "")
        combined = f"{section} {node.text}" if section else node.text
        # 分词版本用于 BM25 索引，原始文本存入 metadata 供后续恢复
        n = node.model_copy(update={
            "text": tokenize_for_bm25(combined),
            "metadata": {**dict(node.metadata), "original_text": combined},
        })
        bm25_nodes.append(n)

    bm25_retriever = BM25Retriever.from_defaults(nodes=bm25_nodes)
    bm25_retriever.similarity_top_k = config.RETRIEVAL_TOP_K

    os.makedirs(config.BM25_DIR, exist_ok=True)
    bm25_retriever.persist(config.BM25_DIR)

    t_elapsed = (datetime.now() - t_start).total_seconds()
    _log(f"  BM25 索引构建完成，持久化到 {config.BM25_DIR}，耗时 {t_elapsed:.1f}s")

    return bm25_retriever


def _load_bm25(log_list: Optional[list] = None) -> BM25Retriever:
    """从 bm25/ 加载 BM25 检索器。"""
    def _log(msg: str):
        logger.info(msg)
        if log_list is not None:
            log_list.append(msg)

    bm25_retriever = BM25Retriever.from_persist_dir(config.BM25_DIR)
    bm25_retriever.similarity_top_k = config.RETRIEVAL_TOP_K
    _log("  BM25 索引已加载")
    return bm25_retriever


def _stage_vector(all_nodes: list, log_list: Optional[list] = None) -> VectorStoreIndex:
    """阶段 4：向量索引构建 → vector/"""
    def _log(msg: str):
        logger.info(msg)
        if log_list is not None:
            log_list.append(msg)

    _embed_bs = getattr(Settings.embed_model, "embed_batch_size", config.EMBED_BATCH_SIZE)
    _num_batches = (len(all_nodes) + _embed_bs - 1) // _embed_bs
    print(f"阶段 4：构建向量索引（共 {len(all_nodes)} 个节点，batch_size={_embed_bs}，"
          f"{_num_batches} 个批次）...", flush=True)
    _log(f"阶段 4：构建向量索引（共 {len(all_nodes)} 个节点，"
         f"batch_size={_embed_bs}，{_num_batches} 个批次）")

    t_start = datetime.now()

    os.makedirs(config.FAISS_PERSIST_DIR, exist_ok=True)

    # 清理旧索引文件，防止残留旧维度索引导致维度不匹配
    old_faiss = os.path.join(config.FAISS_PERSIST_DIR, "index.faiss")
    if os.path.exists(old_faiss):
        print(f"  清理旧 HNSW 索引文件: {old_faiss}", flush=True)
        shutil.rmtree(config.FAISS_PERSIST_DIR, ignore_errors=True)
        os.makedirs(config.FAISS_PERSIST_DIR, exist_ok=True)

    faiss_index = faiss.IndexHNSW(faiss.IndexFlatIP(config.EMBED_VECTOR_DIM), config.HNSW_M)
    faiss_index.hnsw.efConstruction = config.HNSW_EF_CONSTRUCTION
    faiss_index.hnsw.efSearch = config.HNSW_EF_SEARCH
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    _original_embed = Settings.embed_model
    Settings.embed_model = ProgressOllamaEmbedding(
        total_nodes=len(all_nodes),
        log_list=log_list,
        label="向量索引",
        model_name=getattr(_original_embed, "model_name", config.EMBED_MODEL_NAME),
        base_url=getattr(_original_embed, "base_url", config.EMBED_OLLAMA_BASE_URL),
        request_timeout=getattr(_original_embed, "request_timeout", config.ANSWER_OLLAMA_TIMEOUT),
        embed_batch_size=_embed_bs,
    )
    index = VectorStoreIndex(all_nodes, storage_context=storage_context)
    Settings.embed_model = _original_embed
    index.storage_context.persist(persist_dir=config.PERSIST_DIR)

    t_elapsed = (datetime.now() - t_start).total_seconds()
    _log(f"  向量索引构建完成，持久化到 {config.PERSIST_DIR}，耗时 {t_elapsed:.1f}s")

    return index


def _load_vector(log_list: Optional[list] = None) -> VectorStoreIndex:
    """从 vector/ 加载向量索引（FAISS HNSW）。"""
    def _log(msg: str):
        logger.info(msg)
        if log_list is not None:
            log_list.append(msg)

    t_start = datetime.now()

    faiss_index_path = os.path.join(config.PERSIST_DIR, "default__vector_store.json")
    if os.path.exists(faiss_index_path):
        faiss_index = faiss.read_index(faiss_index_path)
    else:
        faiss_index = faiss.IndexHNSW(faiss.IndexFlatIP(config.EMBED_VECTOR_DIM), config.HNSW_M)
        faiss_index.hnsw.efConstruction = config.HNSW_EF_CONSTRUCTION

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


# ======================== 管线控制器 ========================

def get_or_build_index(
    log_list: Optional[list] = None,
) -> Tuple[VectorStoreIndex, BM25Retriever, Optional[dict], Optional["PropertyGraphIndex"]]:
    """分阶段索引管线控制器。

    按序检查各阶段产物目录，从第一个缺失的阶段开始执行。
    已完成阶段的文件夹保留不动，无需重建。

    返回: (向量索引, BM25 检索器, 摘要树元数据映射, 知识图谱索引或 None)
    """
    def _log(msg: str):
        logger.info(msg)
        if log_list is not None:
            log_list.append(msg)

    # 定义阶段列表：(阶段名, 产物目录, 是否启用)
    stages: List[Tuple[str, str, bool]] = [
        ("分块",       config.CHUNKS_DIR,       True),
        ("摘要树",     config.SUMMARY_TREE_DIR, config.SUMMARY_TREE_ENABLED),
        ("BM25 索引",  config.BM25_DIR,         True),
        ("向量索引",   config.PERSIST_DIR,      True),
        ("知识图谱",   config.GRAPH_DB_DIR,     config.GRAPH_ENABLED),
    ]

    # 找到第一个缺失的【启用】阶段
    start_from = None
    for i, (name, path, enabled) in enumerate(stages):
        if enabled and not os.path.exists(path):
            start_from = i
            break

    if start_from is None:
        # ---- 全部存在：从磁盘加载 ----
        print("所有阶段产物已就绪，从磁盘加载...", flush=True)
        _log("所有阶段产物已就绪，从磁盘加载")

        chunk_nodes = _load_chunks(log_list)

        if config.SUMMARY_TREE_ENABLED:
            summary_docs, summary_meta_map = _load_summary(log_list)
        else:
            summary_docs, summary_meta_map = [], None
            _log("摘要树未启用，跳过")

        bm25_retriever = _load_bm25(log_list)
        vector_index = _load_vector(log_list)
        graph_index = load_graph(log_list)

        return vector_index, bm25_retriever, summary_meta_map, graph_index

    # ---- 需要重建：从 start_from 开始 ----
    print(f"检测到阶段「{stages[start_from][0]}」产物缺失，"
          f"从该阶段开始重建...", flush=True)
    _log(f"阶段「{stages[start_from][0]}」产物缺失，从该阶段开始重建")

    # 删除 start_from 及之后所有【启用】阶段的目录
    for i in range(start_from, len(stages)):
        name, path, enabled = stages[i]
        if enabled and os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
            _log(f"  已清理旧产物: {path}")

    # ---- 逐步执行各阶段 ----
    chunk_nodes: Optional[list] = None
    summary_docs: list = []
    summary_meta_map: Optional[dict] = None
    bm25_retriever: Optional[BM25Retriever] = None
    vector_index: Optional[VectorStoreIndex] = None
    graph_index = None  # Optional[PropertyGraphIndex]

    # 阶段 0：分块（如果 start_from <= 0 则执行，否则加载缓存）
    if start_from <= 0:
        chunk_nodes = _stage_chunk(log_list)
    else:
        chunk_nodes = _load_chunks(log_list)

    # 阶段 1：摘要树
    if config.SUMMARY_TREE_ENABLED:
        if start_from <= 1:
            summary_docs, summary_meta_map = _stage_summary(chunk_nodes, log_list)
        else:
            summary_docs, summary_meta_map = _load_summary(log_list)
    else:
        _log("摘要树未启用，跳过")

    # 构建 all_nodes
    all_nodes = list(chunk_nodes) if chunk_nodes else []
    if summary_docs:
        all_nodes = all_nodes + list(summary_docs)

    # 阶段 2：BM25 索引
    if start_from <= 2:
        bm25_retriever = _stage_bm25(all_nodes, log_list)
    else:
        bm25_retriever = _load_bm25(log_list)

    # 阶段 3：向量索引
    if start_from <= 3:
        vector_index = _stage_vector(all_nodes, log_list)
    else:
        vector_index = _load_vector(log_list)

    # 阶段 4：知识图谱（使用原始文档独立分块，chunk 更大）
    if config.GRAPH_ENABLED:
        if start_from <= 4:
            raw_documents = load_documents()
            graph_index = build_graph(raw_documents, Settings.llm, log_list)
        else:
            graph_index = load_graph(log_list)
    else:
        _log("知识图谱未启用，跳过")

    return vector_index, bm25_retriever, summary_meta_map, graph_index