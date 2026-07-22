"""
规范化模块 —— 使用 LLM 将实体别名映射到 Canonical Name。

例如：
  丁先生 → 丁元英
  元英 → 丁元英
  韩大哥 → 韩楚风
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

CANONICALIZE_PROMPT = """你是一个实体名称规范化助手。给定一个「候选名称」和一个「已知实体名称列表」，判断候选名称是否是某个已知实体的别名或简称。

候选名称：{candidate}

已知实体名称：
{known_names}

判断规则：
1. 如果候选名称是某个已知实体的简称、尊称、昵称、别称，返回该已知实体名称
2. 如果候选名称和某个已知实体只差"先生"、"女士"、"总"、"哥"、"姐"等后缀，视为同一实体
3. 如果候选名称和某个已知实体只差姓氏/名字的部分，如"丁元英"和"元英"，视为同一实体
4. 如果候选名称是一个全新的实体，返回 null

输出格式（严格 JSON）：
{{"canonical": "已知实体名称"}}

或

{{"canonical": null}}

仅输出 JSON，不要输出其他内容。"""


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
        prompt = CANONICALIZE_PROMPT.format(
            candidate=candidate,
            known_names="\n".join(f"- {n}" for n in known_names),
        )

        try:
            response = self._llm.complete(prompt)
            text = response.text.strip()

            import json
            import re

            try:
                import json_repair
                data = json_repair.repair_json(text, return_objects=True)
            except (ImportError, Exception):
                json_match = re.search(r"\{.*\}", text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
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