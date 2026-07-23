"""
重排序模块 —— 支持两种 provider（config.RERANK_PROVIDER）：
  - "vllm"：内网 vLLM /v1/rerank 端点调用 bge-reranker-v2-m3（无鉴权）
  - "aliyun"：阿里云 MaaS rerank 端点调用 qwen3-rerank（Bearer 鉴权）

两者均为 cross-encoder 重排，响应格式相同：results[].index + relevance_score
（分数越高越相关），解析逻辑共用。

并发策略：支持批量 /v1/rerank 单次请求（放入所有文档），失败时自动降级为原始顺序。
瞬时故障（网络异常 / 429 / 5xx）先快速重试 RERANK_MAX_RETRIES 次，重试仍失败才降级 ——
避免一次超时就放弃整包 rerank。
"""

import logging
import time
from typing import Optional

import requests

from rag import config

logger = logging.getLogger(__name__)

# 可快速重试的 HTTP 状态码（4xx 配置类错误重试无意义，不重试）
_RETRYABLE_STATUS = (429, 500, 502, 503, 504)
_RETRY_DELAY = 0.5


class Reranker:
    """cross-encoder 重排序器（vLLM bge-reranker-v2-m3 / 阿里云 qwen3-rerank）。"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = config.RERANK_TIMEOUT,
    ):
        # 显式传入 base_url 时视为 vLLM 风格端点（测试/内网直连均走此路径）
        if base_url is None and config.RERANK_PROVIDER == "aliyun":
            self._url = config.ALIYUN_RERANK_URL
            self._model_name = model_name or config.ALIYUN_RERANK_MODEL
            self._headers = {"Authorization": f"Bearer {config.ALIYUN_RERANK_API_KEY}"}
        else:
            self._url = (base_url or config.RERANK_BASE_URL).rstrip("/") + "/v1/rerank"
            self._model_name = model_name or config.RERANK_MODEL_NAME
            self._headers = {}
        self._timeout = timeout

    @staticmethod
    def _log(msg: str):
        logger.info(msg)

    # ---------- 批量重排序 ----------

    def _post_rerank(self, body: dict) -> requests.Response:
        """带快速重试的 POST：网络异常 / 429 / 5xx 重试后仍失败则抛出或返回失败响应。"""
        max_retries = config.RERANK_MAX_RETRIES
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    self._url, json=body, headers=self._headers, timeout=self._timeout,
                )
            except Exception as e:
                if attempt >= max_retries:
                    raise
                logger.warning(f"Reranker 请求异常（{e.__class__.__name__}），"
                               f"{_RETRY_DELAY}s 后重试 ({attempt + 1}/{max_retries})")
                time.sleep(_RETRY_DELAY)
                continue

            if resp.status_code in _RETRYABLE_STATUS and attempt < max_retries:
                logger.warning(f"Reranker HTTP {resp.status_code}，"
                               f"{_RETRY_DELAY}s 后重试 ({attempt + 1}/{max_retries})")
                time.sleep(_RETRY_DELAY)
                continue

            return resp

        raise RuntimeError("Reranker 重试逻辑异常退出")  # 理论上不可达

    def rerank(self, query: str, documents: list[str], top_k: int) -> Optional[list[tuple[int, float]]]:
        """对文档列表重排序，返回 [(原始索引, 分数), ...]，按分数降序。

        调用失败时返回 None（降级信号），由调用方回退到 RRF 顺序，
        不再返回全 0 分数（全 0 会破坏下游按分数排序/去重的逻辑）。
        """
        if not documents:
            return []

        try:
            resp = self._post_rerank({
                "model": self._model_name,
                "query": query,
                "documents": documents,
                "top_n": top_k,
            })

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