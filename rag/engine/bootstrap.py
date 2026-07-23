"""
入口装配模块 —— CLI 与 Streamlit 共用的 Settings 初始化、查询引擎组装、引用格式化。

此前两个入口各自复制约 50 行装配逻辑，已出现行为漂移（ui 缓存 bug）。
现统一收敛到本模块，入口只保留展示层代码。
"""

import logging
from typing import Optional

from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding

from rag import config
from rag.indexing.staged_indexer import get_or_build_index
from rag.retrieval.reranker import Reranker
from rag.retrieval.hybrid_retriever import HybridRetriever, _safe_text
from rag.graph.graph_retriever import GraphRetriever
from rag.engine.query_engine import create_query_engine
from rag.retrieval.query_rewriter import QueryRewriter
from rag.retrieval.query_decomposer import QueryDecomposer
from rag.llm.factory import create_answer_llm, create_rewrite_llm

logger = logging.getLogger(__name__)


def init_settings() -> None:
    """初始化全局 Settings（embedding + 回答 LLM）。"""
    Settings.embed_model = OllamaEmbedding(
        model_name=config.EMBED_MODEL_NAME,
        base_url=config.EMBED_OLLAMA_BASE_URL,
        request_timeout=config.EMBED_TIMEOUT,
        embed_batch_size=config.EMBED_BATCH_SIZE,
    )
    Settings.llm = create_answer_llm()


def build_query_engine() -> RetrieverQueryEngine:
    """构建/加载索引并组装完整查询引擎（检索器 + 重排 + 图增强）。"""
    index, bm25_retriever, summary_meta_map, graph_index = get_or_build_index()

    # 动态兜底：确保 top_k 不超过实际文档数，避免空语料时报错
    total_docs = len(index.docstore.docs)
    vec_top_k = min(config.RETRIEVAL_TOP_K, total_docs)
    vector_retriever = index.as_retriever(similarity_top_k=vec_top_k)

    bm25_corpus_size = len(bm25_retriever._nodes) if hasattr(bm25_retriever, "_nodes") else total_docs
    bm25_retriever.similarity_top_k = min(config.RETRIEVAL_TOP_K, bm25_corpus_size)

    reranker = Reranker() if config.RERANK_ENABLED else None
    rewrite_llm = create_rewrite_llm()
    query_rewriter = QueryRewriter(
        enabled=config.REWRITE_ENABLED,
        llm=rewrite_llm,
    )
    decomposer = QueryDecomposer(
        llm=rewrite_llm,
        enabled=config.DECOMPOSE_ENABLED,
    )

    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        query_rewriter=query_rewriter,
        summary_meta_map=summary_meta_map,
        decomposer=decomposer,
    )

    # 图检索器（独立于三路检索，结果注入到 LLM 上下文）
    graph_retriever = GraphRetriever(
        graph_index=graph_index,
        llm=Settings.llm,
    )

    return create_query_engine(hybrid_retriever, graph_retriever=graph_retriever)


def format_source_nodes(source_nodes) -> list[tuple[str, str]]:
    """格式化参考文献列表，返回 [(标题行, 文本预览), ...]。"""
    results: list[tuple[str, str]] = []
    for i, node in enumerate(source_nodes, start=1):
        # 图谱上下文节点没有 file_name/section，单独标注（避免显示为"未知"）
        if node.metadata.get("is_graph_context"):
            results.append((f"[{i}] 知识图谱关联信息", _safe_text(node)[:100].replace("\n", " ")))
            continue

        fname = node.metadata.get("file_name", "未知")
        section = node.metadata.get("section_path", "") or node.metadata.get("section", "")
        is_summary = node.metadata.get("is_summary", False)
        level = node.metadata.get("summary_level", 0)
        chunk_range = node.metadata.get("summary_chunk_range", [])
        text_preview = _safe_text(node)[:100].replace("\n", " ")

        # 构建摘要标注
        summary_label = ""
        if is_summary:
            if level == 1 and chunk_range:
                summary_label = f" [叶子摘要，覆盖 chunk #{chunk_range[0]+1}]"
            elif chunk_range and len(chunk_range) == 2:
                summary_label = f" [L{level}摘要，覆盖 chunk #{chunk_range[0]+1} ~ #{chunk_range[1]+1}]"

        results.append((f"[{i}] {fname} | {section}{summary_label}", text_preview))
    return results
