"""
CLI 入口 —— 完整的 RAG 检索问答流程。

流程：
  步骤 0：开始运行
  步骤 1：初始化全局设置（embedding + LLM）
  步骤 2：构建查询引擎（索引 → 检索器 → 查询引擎）
      步骤 2.1-2.7：索引构建/加载（由 indexer 模块输出）
      步骤 2.8：组装检索器与查询引擎
  步骤 3：执行查询（由 retriever 模块输出步骤 3.1-3.4）
  步骤 4：LLM 生成回答
  步骤 5：输出参考文献
"""

from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding

from rag import config
from rag.retrieval.hybrid_retriever import _safe_text

from indexer import get_or_build_index
from rag.retrieval.reranker import Reranker
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.graph.graph_retriever import GraphRetriever
from rag.engine.query_engine import create_query_engine
from rag.retrieval.query_rewriter import QueryRewriter
from rag.retrieval.query_decomposer import QueryDecomposer
from rag.llm.factory import create_answer_llm, create_rewrite_llm


def print_step(msg: str):
    """统一打印步骤日志。"""
    print(msg, flush=True)


def run_query(question: str) -> None:
    """执行一次完整查询并打印结果。"""
    log_list: list[str] = []

    # ---- 步骤 0 ----
    print_step("步骤 0：开始运行")

    # ---- 步骤 1：初始化全局设置 ----
    print_step("步骤 1：初始化全局设置 (embedding + LLM)")
    Settings.embed_model = OllamaEmbedding(
        model_name=config.EMBED_MODEL_NAME,
        base_url=config.EMBED_OLLAMA_BASE_URL,
        request_timeout=config.ANSWER_OLLAMA_TIMEOUT,
        embed_batch_size=config.EMBED_BATCH_SIZE,
    )
    Settings.llm = create_answer_llm()

    # ---- 步骤 2：构建查询引擎 ----
    print_step("步骤 2：构建查询引擎")
    print("  正在进入索引构建/加载流程，请稍候...", flush=True)

    index, bm25_retriever, summary_meta_map, graph_index = get_or_build_index(log_list)

    # 步骤 2.1-2.7 的日志立即输出，然后清空给步骤 3 使用
    for line in log_list:
        print(f"  {line}")
    log_list.clear()

    print_step("步骤 2.8：组装检索器与查询引擎")
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

    query_engine = create_query_engine(hybrid_retriever, graph_retriever=graph_retriever)
    print_step("  组装完成")

    # ---- 步骤 3：执行查询 ----
    print_step("步骤 3：执行查询")
    response = query_engine.query(question)

    # ---- 打印检索日志 ----
    print("=" * 60, flush=True)
    for line in log_list:
        print(f"  {line}", flush=True)
    print("=" * 60, flush=True)

    # ---- 步骤 4：LLM 生成回答 ----
    print_step("步骤 4：LLM 生成回答")
    print(f"  {response}")

    # ---- 步骤 5：输出参考文献 ----
    print_step("步骤 5：输出参考文献")
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

            print(f"  [{i}] {fname} | {section}{summary_label} | {text_preview}...")
    else:
        print("  （无参考文献）")


# ======================== 主入口 ========================
if __name__ == "__main__":
    run_query("欧阳雪为什么为难丁元英")