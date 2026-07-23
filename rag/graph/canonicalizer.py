"""
规范化模块 —— 使用 LLM 将实体别名映射到 Canonical Name。

例如：
  丁先生 → 丁元英
  元英 → 丁元英
  韩大哥 → 韩楚风
"""

import logging
from typing import Optional

from rag import prompts
from rag.utils.json_parse import parse_json_obj

logger = logging.getLogger(__name__)


class Canonicalizer:
    """实体名称规范化。"""

    def __init__(self, llm, model_name: str = ""):
        self._llm = llm
        self._model_name = model_name

        # 本地快速映射：不需要 LLM 的简单规则
        self._local_map: dict[str, str] = {}

    def canonicalize(
        self,
        candidate: str,
        known_names: list[str],
    ) -> Optional[str]:
        """将候选名称映射到已知实体名称。

        Args:
            candidate: 候选名称
            known_names: 已知实体名称列表

        Returns:
            Canonical name，如果候选是一个全新实体则返回 None
        """
        if not candidate or not known_names:
            return None

        # 精确匹配：已经是已知名称
        if candidate in known_names:
            return candidate

        # 本地映射缓存
        if candidate in self._local_map:
            return self._local_map[candidate]

        # 快速规则：已知名称包含候选名称
        for name in known_names:
            if len(candidate) >= 2 and candidate in name:
                self._local_map[candidate] = name
                return name

        # 快速规则：候选名称包含已知名称
        for name in known_names:
            if len(name) >= 2 and name in candidate and len(candidate) <= len(name) + 2:
                self._local_map[candidate] = name
                return name

        # 太多已知名称时，只取前 30 个做 LLM 判断
        if len(known_names) > 30:
            known_names = known_names[:30]

        # 调用 LLM
        prompt = prompts.CANONICALIZE_PROMPT.format(
            candidate=candidate,
            known_names="\n".join(f"- {n}" for n in known_names),
        )

        try:
            response = self._llm.complete(prompt)
            text = response.text.strip()

            data = parse_json_obj(text)
            if not data:
                return None

            canonical = data.get("canonical")
            if canonical and canonical in known_names:
                self._local_map[candidate] = canonical
                logger.info(f"  🏷️ 实体规范化: '{candidate}' → '{canonical}'")
                return canonical

        except Exception as e:
            logger.debug(f"Canonicalize 失败 [{candidate}]: {e}")

        return None

    def add_mapping(self, alias: str, canonical: str):
        """手动添加别名映射。"""
        self._local_map[alias] = canonical