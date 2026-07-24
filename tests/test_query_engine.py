"""rag/engine/query_engine.py 单测 —— 真流式回答路径（P1-6 回归）。

背景：llama_index 0.14 的 Refine 合成器在 streaming 模式下会把整个流累积成字符串
后只 yield 一次，使真流式失效。GraphAugmentedQueryEngine 覆盖 synthesize，在
streaming 时绕开合成器直接调 llm.stream_chat 逐 token 产出。这里锁定：
  - 流式路径确实逐 token yield（不是攒完只 yield 一次）
  - 非流式路径仍走父类合成器
  - _query 经 self.synthesize 分发（否则父类直接调 _response_synthesizer 会绕过覆盖）
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle

from rag.engine.query_engine import GraphAugmentedQueryEngine


def _make_nodes():
    return [
        NodeWithScore(node=TextNode(text="陈伯说机器炒的茶火候是死的。",
                                    metadata={"file_name": "WuLingChaShi.txt"}), score=0.9),
        NodeWithScore(node=TextNode(text="顾明澜想投三百万上机器扩产。",
                                    metadata={"file_name": "WuLingChaShi.txt"}), score=0.8),
    ]


class _FakeStreamLLM:
    """逐 token yield 的假 LLM，记录收到的 prompt。"""

    def __init__(self):
        self.received_prompt = None
        self.tokens = ["机", "器", "炒", "的", "茶", "火候是死的。"]

    def stream_chat(self, messages, **kwargs):
        self.received_prompt = messages[0].content
        for t in self.tokens:
            yield ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content=t),
                delta=t,
            )


@pytest.fixture
def engine():
    """构造一个只测 synthesize 的引擎（retriever/synthesizer 用桩，不触发真检索）。"""
    eng = GraphAugmentedQueryEngine.__new__(GraphAugmentedQueryEngine)
    eng._graph_retriever = None
    eng._stream_answer = True
    eng._response_synthesizer = MagicMock()
    return eng


class TestStreamingSynthesize:
    def test_流式逐token产出而非攒完只yield一次(self, engine, monkeypatch):
        fake = _FakeStreamLLM()
        monkeypatch.setattr("rag.engine.query_engine.Settings", SimpleNamespace(llm=fake))
        resp = engine.synthesize(QueryBundle("陈伯为什么反对上机器"), _make_nodes())

        # 是流式响应，且 response_gen 的产出次数 == LLM 的 token 数（逐 token）
        assert hasattr(resp, "response_gen")
        chunks = list(resp.response_gen)
        assert chunks == fake.tokens
        assert len(chunks) == 6      # 关键：不是 1（攒完只 yield 一次就退化成非流式）

    def test_参考资料与问题都进了prompt(self, engine, monkeypatch):
        fake = _FakeStreamLLM()
        monkeypatch.setattr("rag.engine.query_engine.Settings", SimpleNamespace(llm=fake))
        resp = engine.synthesize(QueryBundle("陈伯为什么反对上机器"), _make_nodes())
        list(resp.response_gen)   # 触发 stream_chat
        assert "陈伯说机器炒的茶火候是死的。" in fake.received_prompt
        assert "顾明澜想投三百万上机器扩产。" in fake.received_prompt
        assert "陈伯为什么反对上机器" in fake.received_prompt
        # QA 模板要求注明文件名 → 元数据须带上
        assert "WuLingChaShi.txt" in fake.received_prompt

    def test_source_nodes_挂到响应上(self, engine, monkeypatch):
        monkeypatch.setattr("rag.engine.query_engine.Settings", SimpleNamespace(llm=_FakeStreamLLM()))
        nodes = _make_nodes()
        resp = engine.synthesize(QueryBundle("问题"), nodes)
        assert len(resp.source_nodes) == 2

    def test_附加节点并入source_nodes(self, engine, monkeypatch):
        monkeypatch.setattr("rag.engine.query_engine.Settings", SimpleNamespace(llm=_FakeStreamLLM()))
        extra = [NodeWithScore(node=TextNode(text="附加"), score=0.1)]
        resp = engine.synthesize(QueryBundle("问题"), _make_nodes(),
                                 additional_source_nodes=extra)
        assert len(resp.source_nodes) == 3


class TestNonStreaming:
    def test_非流式走父类合成器(self, engine):
        engine._stream_answer = False
        qb = QueryBundle("问题")
        nodes = _make_nodes()
        engine.synthesize(qb, nodes)
        # 父类 synthesize 内部调 _response_synthesizer.synthesize
        engine._response_synthesizer.synthesize.assert_called_once()


class TestQueryDispatch:
    def test_query经self_synthesize分发(self, engine, monkeypatch):
        """_query 必须调 self.synthesize（被覆盖的），而非父类直接调 _response_synthesizer。"""
        monkeypatch.setattr("rag.engine.query_engine.Settings", SimpleNamespace(llm=_FakeStreamLLM()))
        engine.retrieve = MagicMock(return_value=_make_nodes())
        # 用真实 callback_manager 以支持 event 上下文
        from llama_index.core.callbacks import CallbackManager
        engine.callback_manager = CallbackManager([])

        resp = engine._query(QueryBundle("问题"))
        engine.retrieve.assert_called_once()
        assert hasattr(resp, "response_gen")   # 走到了流式 synthesize
