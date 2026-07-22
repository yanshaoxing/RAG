"""
混合检索模块 —— 三路策略检索，RRF 融合，可选重排序。

流程：
  步骤 3.0 — 查询分解（可选，复杂查询拆分为子查询）
  步骤 3.1 — 三路查询改写（NL改写 / HyDE / BM25关键词扩展）
  步骤 3.2 — 分路检索（NL+HyDE → 向量，关键词 → BM25）
  步骤 3.2b — 摘要冗余过滤（三路 node_id 取并集判断）
  步骤 3.2c — gap 过滤（向量/BM25 各自独立过滤）
  步骤 3.3 — RRF 融合（三路统一融合）
  步骤 3.4 — 重排序（融合后 rerank 一次）
"""

import logging
from typing import Optional

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.retrievers.bm25 import BM25Retriever

from rag import config
from rag.retrieval.reranker import Reranker
from rag.retrieval.query_rewriter import QueryRewriter

logger = logging.getLogger(__name__)


def _safe_text(node) -> str:
    """安全获取任意节点类型的文本内容（非 TextNode 如 ImageNode/IndexNode 返回空字符串）。"""
    try:
        return node.text or ""
    except ValueError:
        return ""


class HybridRetriever(BaseRetriever):
    """三路策略混合检索器：NL改写+HyDE → 向量检索，关键词扩展 → BM25检索，RRF融合。"""

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        bm25_retriever: BM25Retriever,
        reranker: Optional[Reranker] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        summary_meta_map: Optional[dict] = None,
        log_list: Optional[list] = None,
        decomposer=None,
    ):
        super().__init__()
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        self._reranker = reranker
        self._query_rewriter = query_rewriter
        self._summary_meta_map = summary_meta_map or {}
        self._log_list = log_list
        self._decomposer = decomposer

    def _log(self, msg: str):
        if self._log_list is not None:
            self._log_list.append(msg)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        original_query = query_bundle.query_str

        # ---- 步骤 3.0：查询分解 ----
        if self._decomposer is not None:
            is_complex, sub_queries = self._decomposer.decompose(original_query)
            if is_complex:
                return self._decomposed_retrieve(sub_queries)

        # ---- 步骤 3.1~3.4：单查询检索 ----
        return self._single_query_retrieve(original_query)

    # ---------- 单查询完整检索管线 ----------

    def _single_query_retrieve(self, query: str) -> list[NodeWithScore]:
        """执行单条查询的完整检索管线：改写 → 三路检索 → 过滤 → RRF → rerank。"""

        # ---- 步骤 3.1：三路查询改写 ----
        if self._query_rewriter is not None:
            nl_query, hyde_passage, kw_string = self._query_rewriter.rewrite(query)
        else:
            nl_query = hyde_passage = kw_string = query
            self._log("步骤 3.1：查询重写 — 未启用")

        # ---- 步骤 3.2：分路检索 ----
        # 路径 A：NL改写 → 向量检索
        vec_nl = self._vector_retriever.retrieve(QueryBundle(nl_query))
        # 路径 B：HyDE → 向量检索
        vec_hyde = self._vector_retriever.retrieve(QueryBundle(hyde_passage))
        # 路径 C：关键词 → BM25 检索（关键词已是分词后形式，直接送入）
        bm25_kw = self._bm25_retriever.retrieve(QueryBundle(kw_string))

        # BM25 索引文本是分词后的，检索后恢复原始文本。
        # 注意：docstore 中的节点是共享对象（Streamlit 下跨会话复用），
        # 必须用副本替换而非就地改写。
        for n in bm25_kw:
            orig = n.metadata.get("original_text")
            if orig and n.node.text != orig:
                n.node = n.node.model_copy(update={"text": orig})

        self._log(
            f"步骤 3.2：三路检索 — 向量(NL) {len(vec_nl)} 条, "
            f"向量(HyDE) {len(vec_hyde)} 条, BM25(关键词) {len(bm25_kw)} 条"
        )

        if config.DEBUG:
            self._log_debug_top3("向量(NL)", vec_nl)
            self._log_debug_top3("向量(HyDE)", vec_hyde)
            self._log_debug_top3("BM25(关键词)", bm25_kw)

        # ---- 步骤 3.2b：摘要冗余过滤（三路 node_id 取并集判断） ----
        all_node_ids = (
            {n.node.node_id for n in vec_nl}
            | {n.node.node_id for n in vec_hyde}
            | {n.node.node_id for n in bm25_kw}
        )
        v_nl_orig = len(vec_nl)
        v_hyde_orig = len(vec_hyde)
        b_kw_orig = len(bm25_kw)

        vec_nl, v_nl_removed = self._filter_redundant_summaries(vec_nl, all_node_ids)
        vec_hyde, v_hyde_removed = self._filter_redundant_summaries(vec_hyde, all_node_ids)
        bm25_kw, b_kw_removed = self._filter_redundant_summaries(bm25_kw, all_node_ids)

        self._log(
            f"步骤 3.2b：摘要冗余过滤 — 向量(NL) {v_nl_orig}→{len(vec_nl)}"
            f"（移除 {v_nl_removed} 摘要）, 向量(HyDE) {v_hyde_orig}→{len(vec_hyde)}"
            f"（移除 {v_hyde_removed} 摘要）, BM25 {b_kw_orig}→{len(bm25_kw)}"
            f"（移除 {b_kw_removed} 摘要）"
        )

        # ---- 步骤 3.2c：gap 过滤（向量/BM25 各自独立过滤） ----
        vec_nl = self._gap_filter(
            vec_nl, config.VECTOR_MIN_SCORE, config.GAP_THRESHOLD,
            config.MAX_CANDIDATES, config.MIN_CANDIDATES,
        )
        vec_hyde = self._gap_filter(
            vec_hyde, config.VECTOR_MIN_SCORE, config.GAP_THRESHOLD,
            config.MAX_CANDIDATES, config.MIN_CANDIDATES,
        )
        bm25_kw = self._gap_filter(
            bm25_kw, config.BM25_MIN_SCORE, config.GAP_THRESHOLD,
            config.MAX_CANDIDATES, config.MIN_CANDIDATES,
        )
        self._log(
            f"步骤 3.2c：gap 过滤 — 向量(NL) {len(vec_nl)} 条, "
            f"向量(HyDE) {len(vec_hyde)} 条, BM25 {len(bm25_kw)} 条"
        )

        # ---- 步骤 3.3：RRF 三路融合 ----
        node_id_to_score: dict[str, float] = {}
        node_id_to_node: dict[str, NodeWithScore] = {}
        node_id_vec_nl: dict[str, float] = {}
        node_id_vec_hyde: dict[str, float] = {}
        node_id_bm25: dict[str, float] = {}

        for rank, node in enumerate(vec_nl, start=1):
            nid = node.node.node_id
            contrib = 1.0 / (config.RRF_K + rank)
            node_id_to_score[nid] = contrib
            node_id_vec_nl[nid] = contrib
            node_id_to_node[nid] = node

        for rank, node in enumerate(vec_hyde, start=1):
            nid = node.node.node_id
            contrib = 1.0 / (config.RRF_K + rank)
            node_id_to_score[nid] = node_id_to_score.get(nid, 0.0) + contrib
            node_id_vec_hyde[nid] = node_id_vec_hyde.get(nid, 0.0) + contrib
            if nid not in node_id_to_node:
                node_id_to_node[nid] = node

        for rank, node in enumerate(bm25_kw, start=1):
            nid = node.node.node_id
            contrib = 1.0 / (config.RRF_K + rank)
            node_id_to_score[nid] = node_id_to_score.get(nid, 0.0) + contrib
            node_id_bm25[nid] = node_id_bm25.get(nid, 0.0) + contrib
            if nid not in node_id_to_node:
                node_id_to_node[nid] = node

        sorted_ids = sorted(node_id_to_score, key=node_id_to_score.get, reverse=True)
        self._log(f"步骤 3.3：RRF 三路融合 → {len(sorted_ids)} 个唯一文档")

        if config.DEBUG:
            self._log_debug_rrf_three(sorted_ids, node_id_vec_nl, node_id_vec_hyde,
                                      node_id_bm25, node_id_to_score, node_id_to_node)

        # ---- 步骤 3.4：重排序 ----
        reranked = self._do_rerank(query, sorted_ids, node_id_to_node, node_id_to_score)
        if reranked is not None:
            return reranked

        # 无 reranker：直接取 RRF top-k
        top_ids = sorted_ids[:config.FINAL_TOP_K]
        if config.DEBUG:
            self._log_debug_final(top_ids, node_id_to_node, node_id_to_score)
        return [self._build_result(nid, node_id_to_node[nid], node_id_to_score[nid]) for nid in top_ids]

    # ---------- 多子查询分解检索 ----------

    def _decomposed_retrieve(self, sub_queries: list[str]) -> list[NodeWithScore]:
        """对每个子查询独立执行完整检索管线，合并去重后返回。"""

        all_nodes: dict[str, NodeWithScore] = {}
        sub_results: list[list[NodeWithScore]] = []

        for i, sq in enumerate(sub_queries, start=1):
            self._log(f"步骤 3.0b：子查询 {i}/{len(sub_queries)} 检索 → {sq}")
            nodes = self._single_query_retrieve(sq)
            sub_results.append(nodes)
            for n in nodes:
                nid = n.node.node_id
                if nid not in all_nodes or n.score > all_nodes[nid].score:
                    all_nodes[nid] = n

        # 按分数降序排列
        merged = sorted(all_nodes.values(), key=lambda n: n.score, reverse=True)
        self._log(
            f"步骤 3.0c：子查询结果合并 — "
            f"{sum(len(r) for r in sub_results)} 条 → 去重 {len(merged)} 个唯一文档"
        )

        if config.DEBUG:
            self._log(f"  [子查询贡献]")
            for i, nodes in enumerate(sub_results):
                self._log(f"    子查询 {i+1}: {len(nodes)} 条")

        return merged[:config.FINAL_TOP_K]

    # ---------- 摘要冗余过滤 ----------

    def _filter_redundant_summaries(
        self, nodes: list[NodeWithScore],
        all_node_ids: Optional[set] = None,
    ) -> tuple[list[NodeWithScore], int]:
        """
        删除冗余摘要节点：若某摘要节点的 child_ids 中，
        有 ≥ SUMMARY_REDUNDANCY_THRESHOLD 比例的 chunk 出现在 all_node_ids 中，
        则认为原文已覆盖该摘要的语义范围，删除摘要节点。
        """
        threshold = config.SUMMARY_REDUNDANCY_THRESHOLD
        if all_node_ids is None:
            all_node_ids = {n.node.node_id for n in nodes}

        kept: list[NodeWithScore] = []
        removed = 0
        for n in nodes:
            meta = n.metadata
            if meta.get("is_summary", False):
                child_ids: list[str] = meta.get("summary_child_ids", [])
                if child_ids:
                    matched = sum(1 for cid in child_ids if cid in all_node_ids)
                    if matched / len(child_ids) >= threshold:
                        removed += 1
                        continue
            kept.append(n)
        return kept, removed

    # ---------- 相邻分数比过滤 ----------

    @staticmethod
    def _gap_filter(
        nodes: list[NodeWithScore],
        min_score: float,
        gap_threshold: float,
        max_candidates: int,
        min_candidates: int = 0,
    ) -> list[NodeWithScore]:
        """
        相邻分数比过滤：
        1. 过滤 score < min_score
        2. 扫描 gap，gap > threshold 时截断（不少于 min_candidates）
        3. 无大 gap 取前 max_candidates
        """
        filtered = [n for n in nodes if n.score >= min_score]
        if not filtered:
            return []

        for i in range(len(filtered) - 1):
            s_i = filtered[i].score
            s_next = filtered[i + 1].score
            if s_i > 0:
                gap = (s_i - s_next) / s_i
                if gap > gap_threshold:
                    cut_pos = max(i + 1, min_candidates)
                    return filtered[:cut_pos]

        return filtered[:max_candidates]

    # ---------- 重排序 ----------

    def _do_rerank(
        self, query: str, sorted_ids: list[str],
        node_id_to_node: dict, node_id_to_score: dict,
    ) -> Optional[list[NodeWithScore]]:
        """执行重排序，返回 None 表示跳过或降级（由调用方回退到 RRF top-k）。"""
        if self._reranker is None:
            return None

        candidate_ids = sorted_ids[:min(config.RERANK_CANDIDATE_POOL_SIZE, len(sorted_ids))]
        candidate_nodes = [node_id_to_node[nid] for nid in candidate_ids]
        candidate_texts = [_safe_text(n)[: config.RERANK_TEXT_MAX_LENGTH] for n in candidate_nodes]

        reranked = self._reranker.rerank(query=query, documents=candidate_texts, top_k=config.FINAL_TOP_K)
        if reranked is None:
            # reranker 降级：保留 RRF 分数，走无 reranker 的 top-k 路径
            return None
        self._log(
            f"步骤 3.4：重排序 — RRF 融合 {len(sorted_ids)} 个 → "
            f"候选池 {len(candidate_ids)} 条 → 最终取 top-{len(reranked)}"
        )

        if config.DEBUG:
            for idx, score in reranked:
                node = candidate_nodes[idx]
                nid = node.node.node_id
                rrf = node_id_to_score.get(nid, 0.0)
                fname = node.metadata.get("file_name", "?")
                section = node.metadata.get("section", "?")
                preview = _safe_text(node)[:60].replace("\n", " ")
                self._log(f"    rerank: nid={nid[:20]}..., RRF={rrf:.4f} → score={score:.4f}, "
                          f"{fname} | {section}: \"{preview}\"")

        results: list[NodeWithScore] = []
        for idx, score in reranked:
            node = candidate_nodes[idx]
            node.score = score
            results.append(node)
        return results

    @staticmethod
    def _build_result(nid: str, node: NodeWithScore, score: float) -> NodeWithScore:
        node.score = score
        return node

    # ---------- DEBUG 日志 ----------

    def _log_debug_top3(self, label: str, nodes: list[NodeWithScore]):
        if not nodes:
            return
        self._log(f"  [{label} top-3]")
        for i, n in enumerate(nodes[:3], start=1):
            fname = n.metadata.get("file_name", "?")
            section = n.metadata.get("section", "?")
            preview = _safe_text(n)[:80].replace("\n", " ")
            self._log(f"    [{i}] nid={n.node.node_id[:20]}..., score={n.score:.4f}, "
                      f"{fname} | {section}: \"{preview}\"")

    def _log_debug_rrf_three(self, sorted_ids, v_nl_map, v_hyde_map, bm25_map, score_map, node_map):
        n = min(5, len(sorted_ids))
        self._log(f"  [RRF 三路融合 top-{n} 分数拆解]")
        for i, nid in enumerate(sorted_ids[:n], start=1):
            v_nl = v_nl_map.get(nid, 0.0)
            v_hyde = v_hyde_map.get(nid, 0.0)
            b = bm25_map.get(nid, 0.0)
            total = score_map.get(nid, 0.0)
            node = node_map[nid]
            fname = node.metadata.get("file_name", "?")
            section = node.metadata.get("section", "?")
            is_summary = node.metadata.get("is_summary", False)
            preview = _safe_text(node)[:60].replace("\n", " ")
            self._log(f"    [{i}] nid={nid[:20]}..., total={total:.4f} "
                      f"(v_nl={v_nl:.4f}, v_hyde={v_hyde:.4f}, bm25={b:.4f})"
                      f"{' [摘要]' if is_summary else ''}, "
                      f"{fname} | {section}: \"{preview}\"")

    def _log_debug_final(self, top_ids, node_map, score_map):
        self._log(f"  [最终 top-{len(top_ids)} 结果]")
        for i, nid in enumerate(top_ids, start=1):
            node = node_map[nid]
            fname = node.metadata.get("file_name", "?")
            section = node.metadata.get("section", "?")
            self._log(f"    [{i}] nid={nid[:20]}..., RRF={score_map[nid]:.4f}, "
                      f"{fname} | {section}")