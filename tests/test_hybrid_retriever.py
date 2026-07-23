"""rag/retrieval/hybrid_retriever.py 单测 —— gap 过滤、摘要冗余过滤、RRF 三路融合。"""

import pytest

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from rag import config
from rag.retrieval.hybrid_retriever import HybridRetriever


def _node(nid: str, score: float, text: str = "正文", **metadata) -> NodeWithScore:
    return NodeWithScore(node=TextNode(id_=nid, text=text, metadata=metadata), score=score)


class _FakeRetriever(BaseRetriever):
    """返回固定节点列表的假检索器。"""

    def __init__(self, nodes: list[NodeWithScore]):
        super().__init__()
        self._nodes = nodes

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        # 返回副本列表，模拟每次检索独立返回
        return list(self._nodes)


# ---------- gap 过滤 ----------

class TestGapFilter:
    def test_min_score_floor(self):
        nodes = [_node("a", 0.9), _node("b", 0.5), _node("c", 0.05)]
        out = HybridRetriever._gap_filter(nodes, min_score=0.1, gap_threshold=0.9,
                                          max_candidates=10, min_candidates=0)
        assert [n.node.node_id for n in out] == ["a", "b"]

    def test_gap_cut(self):
        # 0.9 → 0.2 相邻降幅 78% > 阈值 20% → 截断
        nodes = [_node("a", 0.9), _node("b", 0.2), _node("c", 0.19)]
        out = HybridRetriever._gap_filter(nodes, min_score=0.0, gap_threshold=0.2,
                                          max_candidates=10, min_candidates=0)
        assert [n.node.node_id for n in out] == ["a"]

    def test_min_candidates_prevents_early_cut(self):
        nodes = [_node("a", 0.9), _node("b", 0.2), _node("c", 0.19)]
        out = HybridRetriever._gap_filter(nodes, min_score=0.0, gap_threshold=0.2,
                                          max_candidates=10, min_candidates=2)
        assert len(out) == 2

    def test_no_gap_takes_max_candidates(self):
        nodes = [_node(f"n{i}", 1.0 - i * 0.01) for i in range(10)]
        out = HybridRetriever._gap_filter(nodes, min_score=0.0, gap_threshold=0.5,
                                          max_candidates=5, min_candidates=0)
        assert len(out) == 5

    def test_all_below_floor(self):
        nodes = [_node("a", 0.01)]
        assert HybridRetriever._gap_filter(nodes, min_score=0.1, gap_threshold=0.2,
                                           max_candidates=10) == []


# ---------- 摘要冗余过滤 ----------

class TestFilterRedundantSummaries:
    def _run(self, nodes, all_ids, threshold, monkeypatch):
        monkeypatch.setattr(config, "SUMMARY_REDUNDANCY_THRESHOLD", threshold)
        retriever = object.__new__(HybridRetriever)
        return HybridRetriever._filter_redundant_summaries(retriever, nodes, all_ids)

    def test_summary_covered_is_removed(self, monkeypatch):
        summary = _node("s1", 0.8, is_summary=True, summary_child_ids=["c1", "c2"])
        chunk = _node("c1", 0.9)
        kept, removed = self._run([summary, chunk], {"s1", "c1", "c2"}, 0.5, monkeypatch)
        assert removed == 1
        assert [n.node.node_id for n in kept] == ["c1"]

    def test_summary_below_threshold_kept(self, monkeypatch):
        summary = _node("s1", 0.8, is_summary=True, summary_child_ids=["c1", "c2", "c3", "c4"])
        kept, removed = self._run([summary], {"s1", "c1"}, 0.5, monkeypatch)
        assert removed == 0
        assert len(kept) == 1

    def test_plain_chunks_untouched(self, monkeypatch):
        nodes = [_node("c1", 0.9), _node("c2", 0.8)]
        kept, removed = self._run(nodes, {"c1", "c2"}, 0.5, monkeypatch)
        assert removed == 0
        assert len(kept) == 2


# ---------- RRF 三路融合（经 _single_query_retrieve 端到端，不含 LLM/网络） ----------

@pytest.fixture
def _rrf_config(monkeypatch):
    """固定融合相关参数，使测试不受调参影响。"""
    monkeypatch.setattr(config, "RRF_K", 60.0)
    monkeypatch.setattr(config, "FINAL_TOP_K", 10)
    monkeypatch.setattr(config, "VECTOR_MIN_SCORE", 0.0)
    monkeypatch.setattr(config, "BM25_MIN_SCORE", 0.0)
    monkeypatch.setattr(config, "GAP_THRESHOLD", 1.0)   # 实际上禁用 gap 截断
    monkeypatch.setattr(config, "MAX_CANDIDATES", 30)
    monkeypatch.setattr(config, "MIN_CANDIDATES", 0)
    monkeypatch.setattr(config, "DEBUG", False)


def _build_retriever(vec_nodes, bm25_nodes) -> HybridRetriever:
    return HybridRetriever(
        vector_retriever=_FakeRetriever(vec_nodes),
        bm25_retriever=_FakeRetriever(bm25_nodes),
        reranker=None,
        query_rewriter=None,
        summary_meta_map={},
        decomposer=None,
    )


class TestRRFFusion:
    def test_multi_route_hit_ranks_first(self, _rrf_config):
        # "both" 在向量与 BM25 都命中 → RRF 累计分应排第一
        vec = [_node("only_vec", 0.9), _node("both", 0.8)]
        bm25 = [_node("both", 5.0), _node("only_bm25", 4.0)]
        out = _build_retriever(vec, bm25)._single_query_retrieve("测试查询")
        assert out[0].node.node_id == "both"
        assert {n.node.node_id for n in out} == {"only_vec", "both", "only_bm25"}

    def test_scores_are_rrf_not_route_scores(self, _rrf_config):
        # 无 reranker 时返回的是 RRF 分数：向量检索器被 NL/HyDE 两路各调一次
        # + BM25 一路，三路 rank=1 → 3/(K+1)
        vec = [_node("both", 0.8)]
        bm25 = [_node("both", 5.0)]
        out = _build_retriever(vec, bm25)._single_query_retrieve("测试查询")
        assert out[0].score == pytest.approx(3.0 / 61.0)

    def test_bm25_original_text_restored(self, _rrf_config):
        # BM25 节点存分词后文本，检索后应恢复 original_text，且不改写共享节点对象
        tokenized = TextNode(id_="b1", text="分词 后 文本",
                             metadata={"original_text": "分词后文本原文"})
        bm25_node = NodeWithScore(node=tokenized, score=3.0)
        out = _build_retriever([], [bm25_node])._single_query_retrieve("测试查询")
        restored = [n for n in out if n.node.node_id == "b1"]
        assert restored and restored[0].node.text == "分词后文本原文"
        # docstore 里的原始节点不能被就地改写（Streamlit 跨会话共享）
        assert tokenized.text == "分词 后 文本"

    def test_final_top_k_truncation(self, _rrf_config, monkeypatch):
        monkeypatch.setattr(config, "FINAL_TOP_K", 3)
        vec = [_node(f"v{i}", 0.9 - i * 0.01) for i in range(8)]
        out = _build_retriever(vec, [])._single_query_retrieve("测试查询")
        assert len(out) == 3


# ---------- 子查询并行检索 ----------

class _FakeDecomposer:
    def __init__(self, sub_queries):
        self._subs = sub_queries

    def decompose(self, query):
        return True, list(self._subs)


class TestDecomposedRetrieve:
    def test_parallel_subqueries_merged_dedup(self, _rrf_config, monkeypatch):
        monkeypatch.setattr(config, "SUBQUERY_MAX_CONCURRENCY", 2)
        vec = [_node("shared", 0.9), _node("v_only", 0.8)]
        bm25 = [_node("shared", 5.0), _node("b_only", 4.0)]
        retriever = HybridRetriever(
            vector_retriever=_FakeRetriever(vec),
            bm25_retriever=_FakeRetriever(bm25),
            reranker=None,
            query_rewriter=None,
            summary_meta_map={},
            decomposer=_FakeDecomposer(["子查询甲内容", "子查询乙内容"]),
        )
        out = retriever._retrieve(QueryBundle("复杂查询"))
        ids = [n.node.node_id for n in out]
        # 两个子查询命中相同节点 → 去重后每个 id 只出现一次
        assert len(ids) == len(set(ids))
        assert set(ids) == {"shared", "v_only", "b_only"}
        # 合并后按分数降序
        scores = [n.score for n in out]
        assert scores == sorted(scores, reverse=True)
