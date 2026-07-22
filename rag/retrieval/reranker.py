"""
重排序模块 —— 通过 vLLM /v1/rerank 端点调用 bge-reranker-v2-m3。

bge-reranker-v2-m3 是 cross-encoder 模型，通过 vLLM 标准 /v1/rerank 端点调用，
返回 relevance_score（分数越高越相关）。

并发策略：支持批量 /v1/rerank 单次请求（放入所有文档），失败时自动降级为原始顺序。
"""

import logging
from typing import Optional

import requests

from rag import config

logger = logging.getLogger(__name__)


class Reranker:
    """通过 vLLM /v1/rerank 端点调用 bge-reranker-v2-m3 进行重排序。"""

    def __init__(
        self,
        base_url: str = config.RERANK_BASE_URL,
        model_name: str = config.RERANK_MODEL_NAME,
        timeout: float = config.RERANK_TIMEOUT,
        log_list: Optional[list] = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._timeout = timeout
        self._log_list = log_list

    def _log(self, msg: str):
        if self._log_list is not None:
            self._log_list.append(msg)

    # ---------- 批量重排序 ----------

    def rerank(self, query: str, documents: list[str], top_k: int) -> Optional[list[tuple[int, float]]]:
        """对文档列表重排序，返回 [(原始索引, 分数), ...]，按分数降序。

        调用失败时返回 None（降级信号），由调用方回退到 RRF 顺序，
        不再返回全 0 分数（全 0 会破坏下游按分数排序/去重的逻辑）。
        """
        if not documents:
            return []

        try:
            resp = requests.post(
                f"{self._base_url}/v1/rerank",
                json={
                    "model": self._model_name,
                    "query": query,
                    "documents": documents,
                    "top_n": top_k,
                },
                timeout=self._timeout,
            )

            if resp.status_code != 200:
                detail = resp.text[:500]
                try:
                    err_json = resp.json()
                    detail = err_json.get("error", detail)
                except Exception:
                    pass
                logger.warning(f"Reranker HTTP {resp.status_code}: {detail}")
                self._log(f"步骤 3.4：重排序 — HTTP {resp.status_code}，降级为 RRF 顺序")
                return None

            data = resp.json()
            results = data.get("results", [])
            if not results:
                logger.warning("Reranker 返回空 results，降级")
                self._log("步骤 3.4：重排序 — 返回空 results，降级为 RRF 顺序")
                return None

            scores: list[tuple[int, float]] = []
            for item in results:
                idx = item.get("index")
                # 响应缺 index 字段时跳过该条，避免默认 0 导致 0 号文档被重复返回
                if idx is None or not (0 <= idx < len(documents)):
                    continue
                score = item.get("relevance_score", 0.0)
                scores.append((idx, score))

            if not scores:
                logger.warning("Reranker 返回结果均无有效 index，降级")
                self._log("步骤 3.4：重排序 — 返回结果无有效 index，降级为 RRF 顺序")
                return None

            scores.sort(key=lambda x: x[1], reverse=True)
            self._log(
                f"步骤 3.4：重排序（候选池 {len(documents)} 条 → top-{min(top_k, len(scores))}）"
            )
            return scores[:top_k]

        except Exception as e:
            logger.warning(f"Reranker 调用失败: {e}，降级")
            self._log(f"步骤 3.4：重排序 — 调用失败 ({e})，降级为 RRF 顺序")
            return None