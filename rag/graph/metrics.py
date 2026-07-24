"""
统计模块 —— 收集构建过程中的各项指标。
"""

import logging

from .models import BuildMetrics

logger = logging.getLogger(__name__)


class MetricsCollector:
    """收集构建统计指标。"""

    def __init__(self):
        self._metrics = BuildMetrics()

    def record_chunk_success(self):
        self._metrics.success_chunks += 1
        self._metrics.total_chunks += 1

    def record_chunk_failed(self):
        self._metrics.failed_chunks += 1
        self._metrics.total_chunks += 1

    def record_entity(self, count: int = 1):
        self._metrics.total_entities += count

    def record_relation(self, count: int = 1):
        self._metrics.total_relations += count

    def record_filtered_by_rules(self, count: int = 1):
        self._metrics.filtered_by_rules += count

    def record_filtered_by_llm(self, count: int = 1):
        self._metrics.filtered_by_llm += count

    def record_corrected_by_llm(self, count: int = 1):
        self._metrics.corrected_by_llm += count

    def record_merged_description(self, count: int = 1):
        self._metrics.merged_descriptions += count

    def record_canonicalized(self, count: int = 1):
        self._metrics.canonicalized_entities += count

    @property
    def summary(self) -> str:
        return self._metrics.summary()

    def log_summary(self):
        for line in self.summary.split("\n"):
            logger.info(line)