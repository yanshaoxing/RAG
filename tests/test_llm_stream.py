"""rag/llm/factory.py 流式单测 —— ThinkStreamFilter 增量剥离与 stream_chat 真流式。"""

import json

import pytest

from rag.llm.factory import DavyLLM, ThinkStreamFilter


def _run_filter(text: str, chunk_size: int) -> str:
    """按 chunk_size 切分喂入过滤器，拼接全部输出。"""
    f = ThinkStreamFilter()
    out = []
    for i in range(0, len(text), chunk_size):
        out.append(f.feed(text[i:i + chunk_size]))
    out.append(f.finalize())
    return "".join(out)


class TestThinkStreamFilter:
    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 1000])
    def test_parity_with_batch_strip(self, chunk_size):
        # 任意切分粒度下与批量版 _strip_thinking 结果一致（闭合块场景）
        cases = [
            "没有思考块的普通回答。",
            "<think>推理过程</think>这是答案。",
            "<thinking>推理过程</thinking>这是答案。",
            "<think>第一段</think>答案甲<think>第二段</think>答案乙",
            "前置文本<think>中间思考</think>后置文本",
        ]
        for text in cases:
            assert _run_filter(text, chunk_size) == DavyLLM._strip_thinking(text), (
                f"chunk_size={chunk_size}, text={text!r}"
            )

    def test_tag_split_across_chunks(self):
        f = ThinkStreamFilter()
        out = f.feed("<thi")
        out += f.feed("nk>内部推理</th")
        out += f.feed("ink>最终答案")
        out += f.finalize()
        assert out == "最终答案"

    def test_angle_bracket_not_a_tag(self):
        # "<" 开头但不是思考标签的文本必须原样输出
        assert _run_filter("数学上 a<b 且 b<c。", 3) == "数学上 a<b 且 b<c。"

    def test_partial_tag_at_stream_end_emitted(self):
        f = ThinkStreamFilter()
        out = f.feed("答案<th")
        out += f.finalize()
        assert out == "答案<th"

    def test_unclosed_think_dropped(self):
        # 未闭合思考块：流式版丢弃（UI 不应显示思考内容）
        f = ThinkStreamFilter()
        out = f.feed("<think>没有闭合的思考")
        out += f.finalize()
        assert out == ""

    def test_whitespace_after_think_stripped(self):
        assert _run_filter("<think>推理</think>\n\n  答案", 4) == "答案"


class _FakeSSEResponse:
    """模拟 requests 流式响应。"""

    def __init__(self, deltas):
        self._deltas = deltas
        self.closed = False

    def iter_lines(self):
        for d in self._deltas:
            payload = {"choices": [{"delta": {"content": d}}]}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}".encode("utf-8")
        yield b"data: [DONE]"

    def close(self):
        self.closed = True


def _stream(monkeypatch, deltas):
    llm = DavyLLM()
    fake = _FakeSSEResponse(deltas)
    monkeypatch.setattr(DavyLLM, "_post_with_retry", lambda self, body, stream=False: fake)
    responses = list(llm.stream_chat([]))
    return responses, fake


class TestStreamChat:
    def test_true_streaming_multiple_yields(self, monkeypatch):
        # 真流式回归：多个增量应产生多次 yield（旧实现全量读完只 yield 一次）
        responses, fake = _stream(monkeypatch, ["第一块", "第二块", "第三块"])
        assert len(responses) == 3
        assert [r.delta for r in responses] == ["第一块", "第二块", "第三块"]
        # message.content 是累积文本
        assert responses[-1].message.content == "第一块第二块第三块"
        assert fake.closed

    def test_think_block_stripped_in_stream(self, monkeypatch):
        responses, _ = _stream(monkeypatch, ["<think>推理", "过程</think>", "真实答案"])
        full = responses[-1].message.content
        assert full == "真实答案"
        assert all("推理" not in (r.delta or "") for r in responses)

    def test_only_thinking_yields_empty_response(self, monkeypatch):
        responses, _ = _stream(monkeypatch, ["<think>只有思考</think>"])
        assert len(responses) == 1
        assert responses[-1].message.content == ""
