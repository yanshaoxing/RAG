"""
进度日志 Embedding 包装类 —— 在批量编码时输出进度。

替代标准 OllamaEmbedding，每完成一个 batch 输出 "已处理 X/Y (百分比%)"。
"""

import logging
from typing import Optional

from llama_index.embeddings.ollama import OllamaEmbedding

logger = logging.getLogger(__name__)


class ProgressOllamaEmbedding(OllamaEmbedding):
    """包装 OllamaEmbedding，批量编码时输出进度日志。"""

    def __init__(
        self,
        total_nodes: int,
        log_list: Optional[list] = None,
        label: str = "Embedding",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._processed = 0
        self._total = total_nodes
        self._log_list = log_list
        self._label = label

    def _log(self, msg: str):
        print(f"    {msg}", flush=True)
        logger.info(msg)
        if self._log_list is not None:
            self._log_list.append(msg)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        embeddings = super()._get_text_embeddings(texts)
        self._processed += len(texts)
        pct = int(self._processed / self._total * 100)
        self._log(f"    已处理 {self._processed}/{self._total} ({pct}%)")
        return embeddings