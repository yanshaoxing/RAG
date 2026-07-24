"""
查询引擎工厂模块 —— 统一组装 Prompt 模板 + RetrieverQueryEngine。

支持可选的图检索增强：在图检索结果可用时，将其作为附加上下文注入到参考资料中。

供两个入口（app/cli.py 与 app/ui.py，经 rag/engine/bootstrap.py 装配）统一使用，
确保 CLI 与 Web 使用完全相同的查询配置。
"""

import logging
from typing import Optional, List, Sequence

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import PromptTemplate, Settings
from llama_index.core.base.response.schema import StreamingResponse
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle, MetadataMode
from llama_index.core.callbacks.schema import CBEventType, EventPayload

from rag import config, prompts
from rag.retrieval.hybrid_retriever import HybridRetriever, _safe_text
from rag.utils.concurrency import run_parallel_captured

logger = logging.getLogger(__name__)


class GraphAugmentedQueryEngine(RetrieverQueryEngine):
    """在 RetrieverQueryEngine 基础上，将图检索结果作为附加上下文注入。

    管道：
      三路检索 → RRF → Rerank → rerank 后的 chunks
      图检索 → 三元组文本
      → 合并后一起送入 LLM
    """

    def __init__(self, graph_retriever=None, streaming: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._graph_retriever = graph_retriever
        self._stream_answer = streaming

    def _query(self, query_bundle: QueryBundle):
        """覆盖父类 _query：父类直接调 self._response_synthesizer.synthesize，会绕过
        本类覆盖的 self.synthesize（真流式路径在那里）。改为经 self.synthesize 分发。
        """
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            nodes = self.retrieve(query_bundle)
            response = self.synthesize(query_bundle, nodes)
            query_event.on_end(payload={EventPayload.RESPONSE: response})
        return response

    def synthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ):
        """回答合成。流式时绕开 llama_index 的 Refine 合成器直接逐 token 流式，
        否则沿用父类成熟的 compact 合成。

        为何绕开：llama_index 0.14 的 DefaultRefineProgram.stream_call 会把整个
        流累积成字符串后只 yield 一次（上游有意为之），使真流式失效（P1-6）。
        因 ALIYUN_CONTEXT_WINDOW 已如实申报（1M），单本查询的参考资料必落入单块、
        compact 本就等价于「一次 QA 调用」，故这里用 QA_TEMPLATE 直接拼接后
        调 stream_chat，行为与 compact 一致，只是真正逐 token 产出。
        """
        if not self._stream_answer:
            return super().synthesize(query_bundle, nodes, additional_source_nodes)

        source_nodes = list(nodes)
        if additional_source_nodes:
            source_nodes = source_nodes + list(additional_source_nodes)

        # 用 LLM 元数据模式取内容：file_name/section 等未排除的元数据一并带上，
        # 供模型「注明文件名称」（与父类 compact 的取文本方式一致）
        context_str = "\n\n".join(
            n.node.get_content(metadata_mode=MetadataMode.LLM) for n in source_nodes
        )
        prompt = prompts.QA_TEMPLATE_STR.format(
            context_str=context_str, query_str=query_bundle.query_str
        )
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]

        def response_gen():
            for chunk in Settings.llm.stream_chat(messages):
                if chunk.delta:
                    yield chunk.delta

        return StreamingResponse(response_gen=response_gen(), source_nodes=source_nodes)

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
                    # 标记键只供入口层识别，不进 LLM 上下文
                    excluded_llm_metadata_keys=["is_graph_context"],
                    excluded_embed_metadata_keys=["is_graph_context"],
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

    # 非流式路径仍用成熟的 compact 合成器；流式路径由 GraphAugmentedQueryEngine
    # .synthesize 绕开 Refine 直接逐 token 产出（见该方法注释，P1-6）。
    # synthesizer 本身不再开 streaming —— 上游 Refine.stream_call 会累积整流后
    # 只 yield 一次，那条路不是真流式。
    response_synthesizer = get_response_synthesizer(
        text_qa_template=qa_template,
        response_mode="compact",
    )

    return GraphAugmentedQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        graph_retriever=graph_retriever,
        streaming=config.ANSWER_STREAM_ENABLED,
    )