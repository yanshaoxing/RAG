"""
章节结构 LLM 检测 —— 内置章节正则（「一、」/「第X回」）零命中的新书，采样送 LLM 判断结构。

流程（由 preprocessor.load_documents 触发）：
  语料档案无 chapter_pattern 字段 且 内置正则全文零命中
  → 采样（开头 + 全文 25%/50%/75% 三处切片）送 LLM
  → LLM 返回章节/小节正则 → 确定性校验（可编译 + 切分章节数与标题长度合理）
  → 通过则写回 corpora/<slug>/corpus.json（之后构建直接读档案，不再调 LLM）
  → 任一环节失败：记 warning 后静默回退内置检测，不中断构建。

校验是安全边界：LLM 产出的正则只有通过「应用到全文后切分结果合理」的
确定性检查才会被采用，坏正则不会进入分块流程。
"""

import logging
import re

from rag import config
from rag.utils.json_parse import parse_json_obj

logger = logging.getLogger(__name__)


# ============================================================
# 采样
# ============================================================

def sample_text(text: str) -> str:
    """采样：开头 HEAD_CHARS 字符 + 全文 25%/50%/75% 三处各 SLICE_CHARS 字符。

    多点采样的目的：避免超长序言/出版说明误导判断，并确认章节格式全书一致。
    切片位置落在开头样本范围内时跳过（短文本只保留开头样本）。
    """
    head_chars = config.STRUCTURE_SAMPLE_HEAD_CHARS
    slice_chars = config.STRUCTURE_SAMPLE_SLICE_CHARS
    parts = [f"【样本：开头】\n{text[:head_chars]}"]
    n = len(text)
    for ratio in (0.25, 0.5, 0.75):
        start = int(n * ratio)
        if start <= head_chars:
            continue
        parts.append(f"【样本：全文{int(ratio * 100)}%处】\n{text[start:start + slice_chars]}")
    return "\n\n".join(parts)


# ============================================================
# 确定性校验
# ============================================================

def validate_chapter_pattern(pattern: str, text: str) -> bool:
    """校验 LLM 给出的章节正则：可编译、不匹配空串、应用到全文后切分结果合理。

    合理性标准（config.STRUCTURE_*）：
      - 标题行命中数在 [MIN_SECTIONS, MAX_SECTIONS] 区间（过少=无效，过多=正则过宽）
      - 命中的标题行长度 ≤ MAX_TITLE_CHARS（超长说明匹配到了正文句子）
    """
    if not pattern:
        return False
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        logger.warning("章节正则无法编译，拒绝：%r（%s）", pattern, e)
        return False
    if compiled.match(""):
        logger.warning("章节正则匹配空字符串，拒绝：%r", pattern)
        return False

    # 与 preprocessor.split_by_section 相同的切分方式（行首 lookahead）
    try:
        parts = re.split(rf"\n[ 　]*(?={pattern})", text)
    except re.error as e:
        logger.warning("章节正则无法用作 lookahead，拒绝：%r（%s）", pattern, e)
        return False

    matched_titles = 0
    for part in parts:
        first_line = part.strip().split("\n", 1)[0].strip()
        if not compiled.match(first_line):
            continue
        if len(first_line) > config.STRUCTURE_MAX_TITLE_CHARS:
            logger.warning("章节正则命中超长标题行（%d 字符），拒绝：%r",
                           len(first_line), pattern)
            return False
        matched_titles += 1

    if not (config.STRUCTURE_MIN_SECTIONS <= matched_titles <= config.STRUCTURE_MAX_SECTIONS):
        logger.warning("章节正则切分出 %d 个章节（合理区间 [%d, %d]），拒绝：%r",
                       matched_titles, config.STRUCTURE_MIN_SECTIONS,
                       config.STRUCTURE_MAX_SECTIONS, pattern)
        return False

    logger.info("章节正则校验通过：%r（切分出 %d 个章节）", pattern, matched_titles)
    return True


def validate_subsection_pattern(pattern: str) -> bool:
    """校验小节标记正则：可编译（MULTILINE 语境）且不匹配空字符串。"""
    if not pattern:
        return False
    try:
        compiled = re.compile(pattern, re.MULTILINE)
    except re.error as e:
        logger.warning("小节正则无法编译，拒绝：%r（%s）", pattern, e)
        return False
    if compiled.match(""):
        logger.warning("小节正则匹配空字符串，拒绝：%r", pattern)
        return False
    return True


# ============================================================
# LLM 检测
# ============================================================

def detect_structure(text: str, llm) -> dict:
    """采样送 LLM 判断章节结构，返回校验通过的正则。

    Returns:
        {"chapter_pattern": str, "subsection_pattern": str}，仅含校验通过的键；
        LLM 失败 / 输出无法解析 / 校验不通过时返回空 dict（调用方回退内置检测）。
    """
    from rag import prompts

    prompt = prompts.STRUCTURE_DETECT_PROMPT.format(samples=sample_text(text))
    try:
        raw = llm.complete(prompt).text
    except Exception as e:
        logger.warning("结构检测 LLM 调用失败：%s", e)
        return {}

    data = parse_json_obj(raw)
    if data is None:
        logger.warning("结构检测 LLM 输出无法解析为 JSON：%s", (raw or "")[:200])
        return {}

    result: dict = {}
    chapter = data.get("chapter_pattern")
    if isinstance(chapter, str) and validate_chapter_pattern(chapter, text):
        result["chapter_pattern"] = chapter

    subsection = data.get("subsection_pattern")
    if isinstance(subsection, str) and validate_subsection_pattern(subsection):
        result["subsection_pattern"] = subsection

    return result


def detect_and_persist(slug: str, text: str) -> dict:
    """检测章节结构并把通过校验的正则写回语料档案。

    仅在新书首次入库（档案无 chapter_pattern 且内置正则零命中）时被调用；
    章节正则校验通过才写回，小节正则单独存在时不写（无章节切分则小节无意义）。
    """
    from rag import corpus as corpus_mod
    from rag.llm.factory import create_summary_llm

    logger.info("步骤 1.0 内置章节正则零命中，采样送 LLM 检测章节结构……")
    result = detect_structure(text, create_summary_llm())
    if result.get("chapter_pattern"):
        corpus_mod.save_structure_patterns(
            slug,
            result["chapter_pattern"],
            result.get("subsection_pattern", ""),
        )
    else:
        logger.warning("LLM 结构检测未得到有效章节正则，回退整书单章处理")
    return result
