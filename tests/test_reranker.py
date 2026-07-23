"""rag/retrieval/reranker.py 单测 —— 响应解析、降级路径（锁定缺陷 #4）与快速重试。"""

import pytest
import requests

import rag.retrieval.reranker as reranker_mod
from rag.retrieval.reranker import Reranker


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """重试等待不真正 sleep，保持测试速度。"""
    monkeypatch.setattr(reranker_mod.time, "sleep", lambda s: None)


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _patch_post(monkeypatch, response=None, exc=None):
    def _fake_post(*args, **kwargs):
        if exc is not None:
            raise exc
        return response

    monkeypatch.setattr(requests, "post", _fake_post)


def _rerank(monkeypatch, response=None, exc=None, docs=None, top_k=5):
    _patch_post(monkeypatch, response=response, exc=exc)
    r = Reranker(base_url="http://fake", model_name="test", timeout=1)
    return r.rerank("查询", docs if docs is not None else ["文档甲", "文档乙", "文档丙"], top_k)


class TestRerankSuccess:
    def test_normal_response_sorted_desc(self, monkeypatch):
        payload = {"results": [
            {"index": 0, "relevance_score": 0.2},
            {"index": 2, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.5},
        ]}
        out = _rerank(monkeypatch, _FakeResponse(payload=payload))
        assert out == [(2, 0.9), (1, 0.5), (0, 0.2)]

    def test_top_k_truncation(self, monkeypatch):
        payload = {"results": [{"index": i, "relevance_score": 1.0 - i * 0.1} for i in range(3)]}
        out = _rerank(monkeypatch, _FakeResponse(payload=payload), top_k=2)
        assert len(out) == 2

    def test_empty_documents_returns_empty(self, monkeypatch):
        # 不应发起网络请求
        _patch_post(monkeypatch, exc=AssertionError("不应调用网络"))
        r = Reranker(base_url="http://fake", model_name="test", timeout=1)
        assert r.rerank("查询", [], 5) == []


class TestRerankDegradation:
    """缺陷 #4 回归：一切降级路径必须返回 None（保留 RRF 分数），不能返回全 0。"""

    def test_http_error_returns_none(self, monkeypatch):
        assert _rerank(monkeypatch, _FakeResponse(status_code=500, text="err")) is None

    def test_network_exception_returns_none(self, monkeypatch):
        assert _rerank(monkeypatch, exc=requests.ConnectionError("refused")) is None

    def test_empty_results_returns_none(self, monkeypatch):
        assert _rerank(monkeypatch, _FakeResponse(payload={"results": []})) is None

    def test_missing_index_entries_skipped(self, monkeypatch):
        # 缺 index 的条目跳过（不能默认 0 导致 0 号文档重复）
        payload = {"results": [
            {"relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.5},
        ]}
        out = _rerank(monkeypatch, _FakeResponse(payload=payload))
        assert out == [(1, 0.5)]

    def test_all_invalid_index_returns_none(self, monkeypatch):
        payload = {"results": [{"relevance_score": 0.9}, {"index": 99, "relevance_score": 0.5}]}
        assert _rerank(monkeypatch, _FakeResponse(payload=payload)) is None


class TestRerankRetry:
    """瞬时故障先快速重试，重试成功不降级；重试仍失败才返回 None。"""

    def _patch_sequence(self, monkeypatch, outcomes):
        """按顺序返回/抛出 outcomes 中的元素（Exception 实例则抛出）。"""
        calls = []

        def _fake_post(*args, **kwargs):
            outcome = outcomes[min(len(calls), len(outcomes) - 1)]
            calls.append(outcome)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(requests, "post", _fake_post)
        return calls

    def test_exception_then_success(self, monkeypatch):
        good = _FakeResponse(payload={"results": [{"index": 0, "relevance_score": 0.9}]})
        calls = self._patch_sequence(monkeypatch, [requests.ConnectionError("refused"), good])
        r = Reranker(base_url="http://fake", model_name="test", timeout=1)
        out = r.rerank("查询", ["文档甲"], 5)
        assert out == [(0, 0.9)]
        assert len(calls) == 2

    def test_5xx_then_success(self, monkeypatch):
        good = _FakeResponse(payload={"results": [{"index": 0, "relevance_score": 0.9}]})
        calls = self._patch_sequence(monkeypatch, [_FakeResponse(status_code=503, text="busy"), good])
        r = Reranker(base_url="http://fake", model_name="test", timeout=1)
        assert r.rerank("查询", ["文档甲"], 5) == [(0, 0.9)]
        assert len(calls) == 2

    def test_4xx_not_retried(self, monkeypatch):
        calls = self._patch_sequence(monkeypatch, [_FakeResponse(status_code=400, text="bad")])
        r = Reranker(base_url="http://fake", model_name="test", timeout=1)
        assert r.rerank("查询", ["文档甲"], 5) is None
        assert len(calls) == 1  # 配置类错误不重试
