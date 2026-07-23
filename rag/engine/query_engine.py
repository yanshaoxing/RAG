"""
查询引擎工厂模块 —— 统一组装 Prompt 模板 + RetrieverQueryEngine。

支持可选的图检索增强：在图检索结果可用时，将其作为附加上下文注入到参考资料中。

供两个入口（app/cli.py 与 app/ui.py，经 rag/engine/bootstrap.py 装配）统一使用，
确保 CLI 与 Web 使用完全相同的查询配置。
"""

from typing import Optional, List

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import PromptTemplate
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle

from rag import prompts
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.utils.concurrency import run_parallel_captured


class GraphAugmentedQueryEngine(RetrieverQueryEngine):
    """在 RetrieverQueryEngine 基础上，将图检索结果作为附加上下文注入。

    管道：
      三路检索 → RRF → Rerank → rerank 后的 chunks
      图检索 → 三元组文本
      → 合并后一起送入 LLM
    """

    def __init__(self, graph_retriever=None, **kwargs):
        super().__init__(**kwargs)
        self._graph_retriever = graph_retriever

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if self._graph_retriever is None or not self._graph_retriever.is_available:
            return super().retrieve(query_bundle)

        # 图检索（实体提取 LLM 调用 + Kuzu 查询）与主检索管线相互独立，
        # 并发执行以隐藏图检索延迟（此前串行排在主检索之后）
        parent_retrieve = super().retrieve
        nodes, graph_text = run_parallel_captured(
            [
                lambda: parent_retrieve(query_bundle),
                lambda: self._graph_retriever.retrieve(query_bundle.query_str),
            ],
            max_workers=2,
        )

        if graph_text:
            # score 取当前结果的最低分（而非固定 1.0）——
            # 图谱上下文是补充信息，不应在任何按分数排序/截断的
            # 下游逻辑中永远压过 rerank 结果
            min_score = min((n.score or 0.0) for n in nodes) if nodes else 0.0
            graph_node = NodeWithScore(
                node=TextNode(
                    text=f"【知识图谱关联信息】\n{graph_text}",
                    metadata={"is_graph_context": True},
                ),
                score=min_score,
            )
            nodes = list(nodes) + [graph_node]

        return nodes


def create_query_engine(
    retriever: HybridRetriever,
    graph_retriever=None,
) -> RetrieverQueryEngine:
    """
    创建配置好 Prompt 模板和响应模式的查询引擎（可选图增强）。

    Args:
        retriever: HybridRetriever 实例
        graph_retriever: 可选的 GraphRetriever 实例，用于注入图检索结果

    Returns:
        配置完成的 RetrieverQueryEngine（或 GraphAugmentedQueryEngine）
    """
    from llama_index.core.response_synthesizers import get_response_synthesizer

    qa_template = PromptTemplate(prompts.QA_TEMPLATE_STR)

    response_synthesizer = get_response_synthesizer(
        text_qa_template=qa_template,
        response_mode="compact",
    )

    return GraphAugmentedQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        graph_retriever=graph_retriever,
    )