"""
Streamlit Web 入口 —— 完整的 RAG 检索问答流程，用于客户端使用。

调用链:
  config → indexer.get_or_build_index → reranker + retriever → query_engine → 查询输出
"""

import streamlit as st
import logging

from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding

from rag import config
from indexer import get_or_build_index
from rag.retrieval.reranker import Reranker
from rag.retrieval.hybrid_retriever import HybridRetriever, _safe_text
from rag.graph.graph_retriever import GraphRetriever
from rag.engine.query_engine import create_query_engine
from rag.retrieval.query_rewriter import QueryRewriter
from rag.retrieval.query_decomposer import QueryDecomposer
from rag.llm.factory import create_answer_llm, create_rewrite_llm

logger = logging.getLogger(__name__)

st.set_page_config(page_title="知识库问答", page_icon="📚")
st.title("📚 内部知识库问答")


# ---------- 步骤 0 ----
st.write("步骤 0：开始运行")

# ---------- 步骤 1：初始化全局设置 ----
Settings.embed_model = OllamaEmbedding(
    model_name=config.EMBED_MODEL_NAME,
    base_url=config.EMBED_OLLAMA_BASE_URL,
    request_timeout=config.ANSWER_OLLAMA_TIMEOUT,
    embed_batch_size=config.EMBED_BATCH_SIZE,
)
Settings.llm = create_answer_llm()


# ---------- 加载索引（自动判断构建/加载） ----------
@st.cache_resource
def load_index_and_engine(log_list: list):
    """
    加载或构建索引，并组装检索 + 查询引擎。

    使用 st.cache_resource 缓存，避免每次查询都重新加载。
    """
    # 大阶段 A：索引构建/加载
    index, bm25_retriever, summary_meta_map, graph_index = get_or_build_index(log_list)

    # 动态兜底：确保 top_k 不超过实际文档数，避免空语料时报错
    total_docs = len(index.docstore.docs)
    vec_top_k = min(config.RETRIEVAL_TOP_K, total_docs)
    vector_retriever = index.as_retriever(similarity_top_k=vec_top_k)

    bm25_corpus_size = len(bm25_retriever._nodes) if hasattr(bm25_retriever, '_nodes') else total_docs
    bm25_top_k = min(config.RETRIEVAL_TOP_K, bm25_corpus_size)
    bm25_retriever.similarity_top_k = bm25_top_k

    reranker = Reranker(log_list=log_list) if config.RERANK_ENABLED else None
    rewrite_llm = create_rewrite_llm()
    query_rewriter = QueryRewriter(
        enabled=config.REWRITE_ENABLED,
        log_list=log_list,
        llm=rewrite_llm,
    )

    decomposer = QueryDecomposer(
        llm=rewrite_llm,
        enabled=config.DECOMPOSE_ENABLED,
        log_list=log_list,
    )

    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        query_rewriter=query_rewriter,
        log_list=log_list,
        summary_meta_map=summary_meta_map,
        decomposer=decomposer,
    )

    # 图检索器（独立于三路检索，结果注入到 LLM 上下文）
    graph_retriever = GraphRetriever(
        graph_index=graph_index,
        llm=Settings.llm,
        log_list=log_list,
    )

    # 组装查询引擎（统一 prompt 模板 + response_mode + 图增强）
    query_engine = create_query_engine(hybrid_retriever, graph_retriever=graph_retriever)
    return query_engine


# ---------- 对话管理 ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- 查询入口 ----------
if prompt := st.chat_input("请输入你的问题"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 每次查询新建日志列表（reranker 和 retriever 持有的是同一引用）
    _log_list: list[str] = []
    query_engine = load_index_and_engine(_log_list)

    # ---- 步骤 2.1-2.7 的日志先输出，然后清空 ----
    # （Streamlit 中这些日志已经在 expander 中，但为保持与 app.py 一致，清空 log_list）
    _log_list.clear()

    # ---- 步骤 3：执行查询 ----
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            response = query_engine.query(prompt)
            answer = str(response)

        # ---- 步骤 4：LLM 生成回答 ----
        st.markdown("**步骤 4：LLM 生成回答**")
        st.markdown(answer)

        # ---- 步骤 5：输出参考文献 ----
        st.markdown("**步骤 5：输出参考文献**")
        if response.source_nodes:
            for i, node in enumerate(response.source_nodes, start=1):
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

                st.caption(f"[{i}] {fname} | {section}{summary_label}")
                st.text(text_preview + "...")
        else:
            st.caption("（无参考文献）")

        # 步骤 2.1-3.5 的运行日志（retriever 模块内部产生）
        if _log_list:
            with st.expander("运行流程"):
                for line in _log_list:
                    st.text(line)

    st.session_state.messages.append({"role": "assistant", "content": answer})