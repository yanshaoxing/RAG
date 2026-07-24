"""
查询分解模块 —— 将复杂查询拆分为多个子查询，分别检索后合并结果。

流程：
  步骤 3.0a — 复杂度判断（启发式 + LLM 兜底）
  步骤 3.0b — LLM 拆分子查询
  步骤 3.0c — 各子查询独立检索，合并去重
"""

import logging
import re

from llama_index.core.llms import ChatMessage, CustomLLM, MessageRole

from rag import config, prompts

logger = logging.getLogger(__name__)

# 多部件关键词：触发快速复杂度判断
_MULTI_PART_KEYWORDS = [
    "和", "以及", "还有", "另外", "分别", "各自",
    "对比", "比较", "区别", "不同", "异同",
    "为什么", "怎么", "如何", "影响", "导致",
    "先", "后", "然后", "之后", "接着",
    "第一", "第二", "首先", "其次",
    "同时", "另一方面", "不仅如此",
    "原因", "结果", "后果", "后来",
]


class QueryDecomposer:
    """查询分解器：判断复杂度 + 拆分子查询。"""

    def __init__(
        self,
        llm: CustomLLM | None = None,
        enabled: bool = True,
    ):
        self._llm = llm
        self._enabled = enabled

    @staticmethod
    def _log(msg: str):
        logger.info(msg)

    def decompose(self, query: str) -> tuple[bool, list[str]]:
        """
        返回 (是否复杂, [子查询列表])。
        不复杂时返回 (False, [原始查询])。
        """
        if not self._enabled:
            return False, [query]

        if not self._is_complex(query):
            return False, [query]

        sub_queries = self._do_decompose(query)
        if not sub_queries or len(sub_queries) <= 1:
            return False, [query]

        self._log(f"步骤 3.0a：查询分解 — 拆分为 {len(sub_queries)} 个子查询")
        for i, sq in enumerate(sub_queries, start=1):
            self._log(f"  子查询 {i}: {sq}")

        return True, sub_queries

    # ---------- 复杂度判断 ----------

    def _is_complex(self, query: str) -> bool:
        """两级判断：启发式快速路径 → LLM 兜底。"""
        if self._heuristic_check(query):
            return True
        return self._llm_classify(query)

    @staticmethod
    def _heuristic_check(query: str) -> bool:
        """启发式检查：多部件关键词数量 + 多问号。"""
        keyword_count = sum(1 for kw in _MULTI_PART_KEYWORDS if kw in query)
        if keyword_count >= 2:
            return True
        question_marks = query.count("？") + query.count("?")
        if question_marks >= 2:
            return True
        if len(query) > 80 and keyword_count >= 1:
            return True
        return False

    def _llm_classify(self, query: str) -> bool:
        """用 LLM 判断查询是否复杂（轻量 prompt）。"""
        prompt = prompts.DECOMPOSE_CLASSIFY_PROMPT.format(query=query)
        try:
            response = self._call_llm(prompt)
            # 前缀匹配：LLM 回答"不是"时也包含"是"字，不能用子串判断。
            # 无法识别时保守返回 False（不拆解），避免误拆放大延迟。
            answer = response.strip().lstrip("：:。.\"'“”「」 ")
            if answer.startswith("否") or answer.startswith("不"):
                return False
            return answer.startswith("是") or answer.lower().startswith("yes")
        except Exception as e:
            logger.warning(f"复杂度分类 LLM 调用失败: {e}")
            return False

    # ---------- 子查询拆解 ----------

    def _do_decompose(self, query: str) -> list[str]:
        """LLM 拆分子查询。"""
        prompt = prompts.DECOMPOSE_PROMPT.format(
            query=query,
            max_sub=config.DECOMPOSE_MAX_SUB_QUERIES,
        )
        try:
            response = self._call_llm(prompt)
            return self._parse_sub_queries(response)
        except Exception as e:
            logger.warning(f"查询分解 LLM 调用失败: {e}")
            return [query]

    @staticmethod
    def _parse_sub_queries(response: str) -> list[str]:
        """解析 LLM 返回的子查询列表。"""
        queries: list[str] = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line or len(line) < 3:
                continue
            # 移除编号前缀：1. 1) 1、 Q1: - *
            line = re.sub(r"^[\d]+[\.\)、\s:：]+\s*", "", line)
            line = re.sub(r"^[-•*]\s*", "", line)
            line = re.sub(r"^[Qq]uestion\s*\d+[:：]\s*", "", line)
            line = re.sub(r"^子问题\s*\d+[:：]\s*", "", line)
            line = re.sub(r"^子查询\s*\d+[:：]\s*", "", line)
            if line and len(line) > 3:
                queries.append(line)
        return queries

    # ---------- LLM 调用 ----------

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        from llama_index.core import Settings
        return Settings.llm

    def _call_llm(self, prompt: str) -> str:
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        llm = self._get_llm()
        llm_response = llm.chat(messages=messages, temperature=config.DECOMPOSE_LLM_TEMPERATURE)
        return str(llm_response.message.content).strip()