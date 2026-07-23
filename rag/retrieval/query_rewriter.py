"""
查询重写模块 —— 术语映射 + 三路策略改写（NL改写 / HyDE / BM25关键词扩展）。

阶段 0：从 terminology.json 加载通俗名→原文术语映射，做最长匹配字符串替换。
阶段 1：三路并行 LLM 改写：
  - 路径 A：自然语言改写（贴近原著措辞，用于向量检索）
  - 路径 B：HyDE 假设性回答段落（用于向量检索）
  - 路径 C：BM25 关键词扩展（用于 BM25 检索）

通过 config.REWRITE_ENABLED 控制开关，失败时自动回退。
"""

import json
import logging
import re
from typing import Optional

from llama_index.core.llms import ChatMessage, MessageRole, CustomLLM

from rag import config, prompts
from rag.utils.concurrency import run_parallel_captured

logger = logging.getLogger(__name__)


class QueryRewriter:
    """三路查询重写：NL改写 + HyDE + BM25关键词扩展。"""

    def __init__(self, enabled: bool = True, llm: Optional[CustomLLM] = None):
        self._enabled = enabled
        self._llm = llm
        self._term_map: dict[str, str] = self._load_term_map()

    @staticmethod
    def _log(msg: str):
        logger.info(msg)

    # ---------- 阶段 0：术语映射 ----------

    @staticmethod
    def _load_term_map() -> dict[str, str]:
        """从 terminology.json 加载术语映射，key 按长度降序排序（最长匹配优先）。"""
        try:
            with open(config.TERM_MAP_PATH, "r", encoding="utf-8") as f:
                raw: dict[str, str] = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("加载术语映射文件失败 (%s): %s，跳过术语替换",
                           config.TERM_MAP_PATH, e)
            return {}

        return dict(sorted(raw.items(), key=lambda kv: len(kv[0]), reverse=True))

    def _apply_term_map(self, query: str) -> tuple[str, list[str]]:
        """对查询做术语映射替换（单遍、最长匹配优先），返回 (替换后查询, 替换日志列表)。

        用正则交替一次性替换：只匹配原始查询中的词条，替换结果不参与再次匹配
        （逐词条 str.replace 会级联——长词条的替换结果被短词条二次替换）。
        """
        if not self._term_map:
            return query, []

        # _load_term_map 已按 key 长度降序排序，交替分支顺序即最长匹配优先
        pattern = re.compile("|".join(re.escape(k) for k in self._term_map))
        replaced: list[str] = []

        def _sub(m: re.Match) -> str:
            slang = m.group()
            original = self._term_map[slang]
            entry = f"{slang} → {original}"
            if entry not in replaced:
                replaced.append(entry)
            return original

        result = pattern.sub(_sub, query)
        return result, replaced

    # ---------- 阶段 1：三路 LLM 改写 ----------

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        from llama_index.core import Settings
        return Settings.llm

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM 并返回原始文本，失败时返回空字符串。"""
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        llm = self._get_llm()
        llm_response = llm.chat(messages=messages, temperature=config.REWRITE_OLLAMA_TEMPERATURE)
        return str(llm_response.message.content).strip()

    def _generate_nl(self, mapped_query: str) -> str:
        """路径 A：自然语言改写（贴近原著措辞）。"""
        prompt = prompts.REWRITE_NL_PROMPT.format(query=mapped_query)
        try:
            return self._call_llm(prompt)
        except Exception as e:
            logger.warning(f"NL改写失败: {e}")
            return mapped_query

    def _generate_hyde(self, mapped_query: str) -> str:
        """路径 B：HyDE 假设性回答段落。"""
        prompt = prompts.REWRITE_HYDE_PROMPT.format(query=mapped_query)
        try:
            return self._call_llm(prompt)
        except Exception as e:
            logger.warning(f"HyDE生成失败: {e}")
            return mapped_query

    def _generate_kw(self, mapped_query: str) -> str:
        """路径 C：BM25 关键词扩展。"""
        prompt = prompts.REWRITE_KW_PROMPT.format(query=mapped_query)
        try:
            result = self._call_llm(prompt)
            return self._dedup_kw(result)
        except Exception as e:
            logger.warning(f"关键词扩展失败: {e}")
            return mapped_query

    @staticmethod
    def _dedup_kw(kw_string: str, max_kw: int = 50) -> str:
        """关键词去重 + 截断：保留顺序去重，超过 max_kw 个则截断。"""
        tokens = kw_string.split()
        seen: set[str] = set()
        unique: list[str] = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        if len(unique) > max_kw:
            unique = unique[:max_kw]
        return " ".join(unique)

    def rewrite(self, query: str) -> tuple[str, str, str]:
        """三路改写查询，返回 (nl_query, hyde_passage, kw_string)。

        禁用或全部失败时返回 (原始查询, 原始查询, 原始查询)。
        """
        # ---- 阶段 0：术语映射（纯字符串替换，不依赖 LLM，禁用改写时也执行） ----
        mapped, term_log = self._apply_term_map(query)

        if not self._enabled:
            if term_log:
                self._log(f"步骤 3.1：查询重写 — 未启用（术语映射: {', '.join(term_log)}）")
            return mapped, mapped, mapped

        # ---- 阶段 1：三路 LLM 改写（三路彼此独立，并行执行） ----
        nl_query, hyde_passage, kw_string = run_parallel_captured(
            [
                lambda: self._generate_nl(mapped),
                lambda: self._generate_hyde(mapped),
                lambda: self._generate_kw(mapped),
            ],
            max_workers=config.QUERY_REWRITE_MAX_CONCURRENCY,
        )

        self._log("步骤 3.1：三路查询改写（并行）")
        self._log(f"  原始查询: {query}")
        if term_log:
            self._log(f"  术语映射: {', '.join(term_log)}")
        self._log(f"  [A] NL改写: {nl_query}")
        self._log(f"  [B] HyDE:   {hyde_passage}")
        self._log(f"  [C] 关键词: {kw_string}")

        return nl_query, hyde_passage, kw_string