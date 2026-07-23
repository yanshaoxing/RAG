"""rag/retrieval/reranker.py 单测 —— 响应解析与降级路径（锁定缺陷 #4：降级返回 None 而非全 0）。"""

import requests

from rag.retrieval.reranker import Reranker


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
