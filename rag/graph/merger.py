"""
合并模块 —— 增量合并实体描述，始终维护一条 Canonical Description。

与旧版的关键区别：
  旧版：取第一次出现的描述，后续全部忽略
  新版：每次新描述到来时，与已有描述合并，始终保持一条高质量描述
"""

import logging
from typing import Optional

from rag import prompts

logger = logging.getLogger(__name__)


class DescriptionMerger:
    """增量合并实体描述。"""

    def __init__(self, llm, model_name: str = ""):
        self._llm = llm
        self._model_name = model_name

    def merge(
        self,
        existing_desc: str,
        new_desc: str,
        entity_name: str,
    ) -> str:
        """将新描述合并到已有描述中。

        Args:
            existing_desc: 已有（已合并的）描述
            new_desc: 新抽取的描述
            entity_name: 实体名称（用于日志）

        Returns:
            合并后的描述
        """
        # 已有描述为空，直接用新描述
        if not existing_desc or len(existing_desc) < 5:
            return new_desc

        # 新描述为空，保留已有
        if not new_desc or len(new_desc) < 5:
            return existing_desc

        # 新描述是已有描述的子串，保留已有
        if new_desc in existing_desc:
            return existing_desc

        # 已有描述是新描述的子串，用新描述
        if existing_desc in new_desc:
            return new_desc

        # 两者完全相同
        if existing_desc.strip() == new_desc.strip():
            return existing_desc

        # 调用 LLM 合并
        prompt = prompts.MERGE_PROMPT.format(
            existing_desc=existing_desc,
            new_desc=new_desc,
        )

        try:
            response = self._llm.complete(prompt)
            merged = response.text.strip()
            if len(merged) >= 5:
                logger.debug(
                    f"  📝 Description 合并 [{entity_name}]: "
                    f"\"{existing_desc[:30]}...\"  +  \"{new_desc[:30]}...\"  →  \"{merged[:40]}...\""
                )
                return merged
        except Exception as e:
            logger.warning(f"Description 合并失败 [{entity_name}]: {e}")

        # 回退：保留较长的描述
        return existing_desc if len(existing_desc) >= len(new_desc) else new_desc