"""
摘要树构建模块 —— 层次化摘要树。

流程：
  1. L1 叶子摘要：每个 chunk → ≤100 字一句话摘要
  2. L2 小节摘要：按 (section, subsection) 分组，读出原文 → 20% 摘要
  3. L3 章节摘要：按 section 分组，读出原文 → 15% 摘要
  4. L4 全书摘要：合并所有 L3 摘要 → ≤500 字

所有层级的摘要节点作为 Document 混入主索引。
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from llama_index.core import Document

from rag import config
from rag.llm.factory import create_summary_llm

logger = logging.getLogger(__name__)


def _log(msg: str) -> None:
    """管线日志：经标准 logging 输出，入口层用 capture_pipeline_logs 捕获。"""
    logger.info(msg)


# ======================== 数据结构 ========================

@dataclass
class SummaryNode:
    """摘要树中的一个节点（中间数据结构，非最终输出 Document）。"""
    text: str                              # 摘要文本
    node_id: str                           # 唯一标识
    level: int                             # 0=chunk, 1=叶子摘要, 2=小节, 3=章节, 4=全书
    child_ids: list[str] = field(default_factory=list)  # 直接子节点的 id 列表
    file_name: str = ""                    # 所属文件名
    section: str = ""                      # 所属章节
    subsection: str = ""                   # 所属小节编号
    chunk_range: tuple[int, int] = (0, 0)  # 覆盖的 chunk 索引范围 [start, end]（闭区间）
    original_text: str = ""                # 原始文本（L1=chunk原文, L2=小节原文, L3=章节原文）
    is_fallback: bool = False              # True=LLM 失败后用原文截断冒充的降级摘要


@dataclass
class SummaryGroup:
    """一个待合并的摘要分组。"""
    children: list[SummaryNode]             # 该组内的子摘要节点
    file_name: str = ""
    section: str = ""
    subsection: str = ""


# ======================== 叶子摘要生成（L1）=======================


def _parse_batch_leaf_response(response: str, expected_count: int) -> list[str]:
    """
    解析批量摘要响应，提取每条摘要。

    Args:
        response: LLM 返回的批量摘要文本
        expected_count: 期望的摘要条数

    Returns:
        长度为 expected_count 的摘要列表，解析失败的位置填入 fallback
    """
    results = [""] * expected_count
    pattern = re.compile(r"\[(\d+)\]\s*(.+?)(?=\[\d+\]|$)", re.DOTALL)
    for m in pattern.finditer(response):
        idx = int(m.group(1)) - 1  # 转为 0-based
        content = m.group(2).strip()
        content = re.sub(r"\s+", " ", content)
        if 0 <= idx < expected_count:
            results[idx] = content
    # 未匹配到的置空
    for i in range(expected_count):
        if not results[i]:
            results[i] = ""
    return results


def _generate_batch_leaves(batch_nodes: list, batch_start_idx: int) -> list[SummaryNode]:
    """
    为一个批次的 chunk 批量生成叶子摘要（一次 LLM 调用）。
    供线程池调用，每个线程创建独立的 LLM 实例。

    Args:
        batch_nodes: 该批次的节点列表
        batch_start_idx: 批次起始的全局 chunk 索引

    Returns:
        SummaryNode 列表，level=1，顺序与 batch_nodes 一致
    """
    llm = create_summary_llm()
    batch_count = len(batch_nodes)

    # 构建批量 prompt
    chunks_parts = []
    for i, node in enumerate(batch_nodes, start=1):
        chunks_parts.append(f"[{i}] {node.text[:2048]}")
    chunks_str = "\n\n".join(chunks_parts)
    prompt = config.SUMMARY_LEAF_BATCH_PROMPT.format(chunks=chunks_str)

    try:
        resp = llm.complete(prompt)
        leaf_texts = _parse_batch_leaf_response(resp.text.strip(), batch_count)
    except Exception as e:
        logger.warning("批量生成叶子摘要失败 (batch_start=%d, count=%d): %s", batch_start_idx, batch_count, e)
        leaf_texts = [""] * batch_count

    results: list[SummaryNode] = []
    for i, node in enumerate(batch_nodes):
        chunk_idx = batch_start_idx + i
        fname = node.metadata.get("file_name", "")
        section = node.metadata.get("section", "")
        subsection = node.metadata.get("subsection", "")
        # LLM 失败/漏答时用原文截断兜底，并显式标记为降级摘要（可统计、可事后定位）
        is_fallback = not leaf_texts[i]
        leaf_text = leaf_texts[i] if leaf_texts[i] else (node.text[:100] + "...")

        results.append(SummaryNode(
            text=leaf_text,
            node_id=f"summary_leaf_{node.node_id}",
            level=1,
            child_ids=[node.node_id],
            file_name=fname,
            section=section,
            subsection=subsection,
            chunk_range=(chunk_idx, chunk_idx),
            original_text=node.text,  # L1 保留原始 chunk 文本
            is_fallback=is_fallback,
        ))
    return results


def _fallback_batch_leaves(batch_nodes: list, batch_start_idx: int) -> list[SummaryNode]:
    """批次任务整体失败时的兜底：全部用原文截断生成降级叶子摘要（不调 LLM）。"""
    results: list[SummaryNode] = []
    for i, node in enumerate(batch_nodes):
        chunk_idx = batch_start_idx + i
        results.append(SummaryNode(
            text=node.text[:100] + "...",
            node_id=f"summary_leaf_{node.node_id}",
            level=1,
            child_ids=[node.node_id],
            file_name=node.metadata.get("file_name", ""),
            section=node.metadata.get("section", ""),
            subsection=node.metadata.get("subsection", ""),
            chunk_range=(chunk_idx, chunk_idx),
            original_text=node.text,
            is_fallback=True,
        ))
    return results


def _generate_leaves(nodes: list) -> list[SummaryNode]:
    """
    为每个 chunk 并发生成叶子摘要（批量模式：每批 SUMMARY_LEAF_BATCH_SIZE 个 chunk
    一次 LLM 调用，批次间通过 ThreadPoolExecutor 并发）。

    Args:
        nodes: 分块后的节点列表

    Returns:
        SummaryNode 列表，level=1
    """
    total = len(nodes)
    batch_size = config.SUMMARY_LEAF_BATCH_SIZE

    # 将节点按 batch_size 分组，每组是一次 LLM 调用的任务
    batch_tasks: list[tuple[list, int]] = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_tasks.append((nodes[start:end], start))

    num_batches = len(batch_tasks)
    max_workers = min(config.SUMMARY_MAX_CONCURRENCY, num_batches)
    print(f"    开始生成 L1 叶子摘要（共 {total} 个 chunk，{num_batches} 批，"
          f"每批 {batch_size} 个，并发数={max_workers}）...", flush=True)
    _log(f"  L1 叶子摘要：共 {total} 个 chunk，{num_batches} 批，每批 {batch_size} 个，并发数={max_workers}")

    summary_nodes: list[SummaryNode] = [None] * total

    if max_workers <= 1:
        # 串行执行各批次
        completed = 0
        for batch_nodes, batch_start in batch_tasks:
            batch_results = _generate_batch_leaves(batch_nodes, batch_start)
            for i, sn in enumerate(batch_results):
                summary_nodes[batch_start + i] = sn
            completed += len(batch_results)
            bs = config.SUMMARY_BATCH_SIZE
            if completed % bs == 0 or completed >= total:
                pct = int(completed / total * 100)
                msg = f"  已处理 {completed}/{total} ({pct}%)"
                print(f"    {msg}", flush=True)
                _log(msg)
    else:
        # 并发执行各批次
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(_generate_batch_leaves, batch_nodes, batch_start): batch_start
                for batch_nodes, batch_start in batch_tasks
            }

            completed_batches = 0
            for future in as_completed(future_to_batch):
                batch_start = future_to_batch[future]
                try:
                    batch_results = future.result()
                except Exception as e:
                    # 单个 worker 崩溃不中止整个摘要阶段：该批次降级为原文截断
                    logger.exception("叶子摘要批次任务异常 (batch_start=%d): %s", batch_start, e)
                    batch_nodes = next(bn for bn, bs in batch_tasks if bs == batch_start)
                    batch_results = _fallback_batch_leaves(batch_nodes, batch_start)
                for i, sn in enumerate(batch_results):
                    summary_nodes[batch_start + i] = sn
                completed_batches += 1
                completed = completed_batches * batch_size
                if completed > total:
                    completed = total
                bs = config.SUMMARY_BATCH_SIZE
                if completed_batches % max(1, bs // batch_size) == 0 or completed >= total:
                    pct = int(completed / total * 100)
                    msg = f"  已处理 {completed}/{total} ({pct}%)"
                    print(f"    {msg}", flush=True)
                    _log(msg)

    _log(f"  L1 叶子摘要生成完成，共 {len(summary_nodes)} 条")
    return summary_nodes


# ======================== 分组策略 ========================


def _group_by_subsection(
    leaves: list[SummaryNode],
) -> list[SummaryGroup]:
    """按 (file_name, section, subsection) 分组。"""
    groups: dict[tuple[str, str, str], list[SummaryNode]] = {}
    for leaf in leaves:
        key = (leaf.file_name, leaf.section, leaf.subsection)
        groups.setdefault(key, []).append(leaf)

    result: list[SummaryGroup] = []
    for (fname, section, subsection), children in groups.items():
        result.append(SummaryGroup(
            children=children, file_name=fname, section=section, subsection=subsection,
        ))
    return result


def _group_by_section(
    nodes: list[SummaryNode],
) -> list[SummaryGroup]:
    """按 (file_name, section) 分组，用于 L2→L3。"""
    groups: dict[tuple[str, str], list[SummaryNode]] = {}
    for node in nodes:
        key = (node.file_name, node.section)
        groups.setdefault(key, []).append(node)

    result: list[SummaryGroup] = []
    for (fname, section), children in groups.items():
        result.append(SummaryGroup(
            children=children, file_name=fname, section=section,
        ))
    return result


# ======================== 父级摘要生成 ========================


def _generate_parent_summary(
    group: SummaryGroup,
    level: int,
    use_original: bool = False,
    max_chars: Optional[int] = None,
    ratio: Optional[float] = None,
) -> SummaryNode:
    """
    将一个分组的子节点合并为一个父级摘要。

    Args:
        group: 摘要分组
        level: 父级摘要的层级
        use_original: True=用子节点 original_text，False=用子节点 text
        max_chars: 显式指定摘要字数上限（优先于 ratio）
        ratio: 摘要字数比例（相对于 source_text 长度），默认 SUMMARY_PARENT_RATIO

    Returns:
        父级 SummaryNode
    """
    # 收集所有孙节点 id 和 chunk_range
    all_child_ids: list[str] = []
    min_chunk = float("inf")
    max_chunk = -1
    fname = group.file_name
    section = group.section
    subsection = group.subsection

    for child in group.children:
        all_child_ids.extend(child.child_ids)
        c_start, c_end = child.chunk_range
        if c_start < min_chunk:
            min_chunk = c_start
        if c_end > max_chunk:
            max_chunk = c_end
        if not fname and child.file_name:
            fname = child.file_name
        if not section and child.section:
            section = child.section
        if not subsection and child.subsection:
            subsection = child.subsection

    # 构建输入源文本
    if use_original:
        source_parts = []
        for child in group.children:
            if child.original_text:
                source_parts.append(child.original_text)
        source_text = "\n\n".join(source_parts)
    else:
        child_parts = []
        for idx, child in enumerate(group.children, start=1):
            child_parts.append(f"[{idx}] {child.text}")
        source_text = "\n".join(child_parts)

    # 动态字数上限
    if max_chars is not None:
        chars_limit = max_chars
    elif ratio is not None:
        chars_limit = max(200, int(len(source_text) * ratio))
    else:
        chars_limit = max(200, int(len(source_text) * config.SUMMARY_PARENT_RATIO))

    # 生成父级摘要（LLM 失败时用原文截断兜底，并显式标记为降级摘要）
    parent_text = ""
    try:
        llm = create_summary_llm()
        if use_original:
            prompt = config.SUMMARY_PARENT_FROM_RAW_PROMPT.format(
                max_chars=chars_limit,
                section_name=f"{section}" + (f" §{subsection}" if subsection else ""),
                chapter_text=source_text,
            )
        else:
            prompt = config.SUMMARY_PARENT_PROMPT.format(
                max_chars=chars_limit,
                child_summaries=source_text,
            )
        resp = llm.complete(prompt)
        parent_text = resp.text.strip()
    except Exception as e:
        logger.warning("生成父级摘要失败 (level=%d): %s", level, e)

    is_fallback = not parent_text
    if is_fallback:
        parent_text = source_text[:200] + "..."

    parent_id = f"summary_L{level}_{int(min_chunk)}_{int(max_chunk)}"
    return SummaryNode(
        text=parent_text,
        node_id=parent_id,
        level=level,
        child_ids=all_child_ids,
        file_name=fname,
        section=section,
        subsection=subsection,
        chunk_range=(int(min_chunk), int(max_chunk)),
        original_text=source_text,
        is_fallback=is_fallback,
    )


# ======================== L2: 小节摘要 ========================


def _generate_subsection_summaries(leaves: list[SummaryNode]) -> list[SummaryNode]:
    """
    L1 → L2：按 (section, subsection) 分组，从原文直接生成小节摘要。

    对于无小节的章节（subsection=""），L2 即章节摘要。
    """
    groups = _group_by_subsection(leaves)
    num_groups = len(groups)
    print(f"    L2 小节摘要：{len(leaves)} 个叶子 → {num_groups} 个小节", flush=True)
    _log(f"  L2 小节摘要：{len(leaves)} 个叶子 → {num_groups} 个小节")

    if num_groups == 0:
        return []

    max_workers = min(config.SUMMARY_MAX_CONCURRENCY, num_groups)
    l2_nodes: list[SummaryNode] = []

    if max_workers <= 1:
        for g in groups:
            l2_nodes.append(_generate_parent_summary(
                g, level=2, use_original=True, ratio=config.SUMMARY_PARENT_RATIO,
            ))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_group = {
                executor.submit(
                    _generate_parent_summary, g, 2, True,
                    None, config.SUMMARY_PARENT_RATIO,
                ): g
                for g in groups
            }
            for future in as_completed(future_to_group):
                try:
                    l2_nodes.append(future.result())
                except Exception as e:
                    g = future_to_group[future]
                    logger.exception("L2 小节摘要任务异常 (section=%s §%s): %s",
                                     g.section, g.subsection, e)

    _log(f"  L2 小节摘要生成完成，共 {len(l2_nodes)} 条")
    return l2_nodes


# ======================== L3: 章节摘要 ========================


def _generate_chapter_summaries(l2_nodes: list[SummaryNode]) -> list[SummaryNode]:
    """
    L2 → L3：按 section 分组，从 L2 的 original_text 重构章节原文生成章节摘要。

    无小节章节（该章仅 1 个 L2 且 subsection=""）：
    L2 已覆盖整章，直接返回空列表（L2 本身就是章节级摘要）。
    """
    # 区分有/无小节章节
    chapter_l2_map: dict[str, list[SummaryNode]] = {}
    for node in l2_nodes:
        key = f"{node.file_name}|{node.section}"
        chapter_l2_map.setdefault(key, []).append(node)

    # 需要生成 L3 的章节：L2 节点数 > 1 或 subsection != ""
    chapters_for_l3: list[SummaryGroup] = []
    no_sub_chapters = 0
    for key, nodes in chapter_l2_map.items():
        if len(nodes) == 1 and nodes[0].subsection == "":
            no_sub_chapters += 1
        else:
            chapters_for_l3.append(SummaryGroup(
                children=nodes,
                file_name=nodes[0].file_name,
                section=nodes[0].section,
            ))

    if no_sub_chapters > 0:
        _log(f"  L3 章节摘要：{no_sub_chapters} 个无小节的章节使用 L2 节点作为章节摘要")

    if not chapters_for_l3:
        _log("  L3 章节摘要：无需额外生成")
        return []

    num_l3 = len(chapters_for_l3)
    print(f"    L3 章节摘要：{num_l3} 个章节（从原文提取）", flush=True)
    _log(f"  L3 章节摘要：{num_l3} 个章节（{config.SUMMARY_CHAPTER_RATIO*100:.0f}% 压缩率）")

    max_workers = min(config.SUMMARY_MAX_CONCURRENCY, num_l3)
    l3_nodes: list[SummaryNode] = []

    if max_workers <= 1:
        for g in chapters_for_l3:
            l3_nodes.append(_generate_parent_summary(
                g, level=3, use_original=True, ratio=config.SUMMARY_CHAPTER_RATIO,
            ))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_group = {
                executor.submit(
                    _generate_parent_summary, g, 3, True,
                    None, config.SUMMARY_CHAPTER_RATIO,
                ): g
                for g in chapters_for_l3
            }
            for future in as_completed(future_to_group):
                try:
                    l3_nodes.append(future.result())
                except Exception as e:
                    g = future_to_group[future]
                    logger.exception("L3 章节摘要任务异常 (section=%s): %s", g.section, e)

    _log(f"  L3 章节摘要生成完成，共 {len(l3_nodes)} 条")
    return l3_nodes


# ======================== 章节排序 ========================

_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}


def _cn_numeral_to_int(text: str) -> int:
    """把中文数字（如 "十二"、"二十三"、"一百零五"）转为整数。无法解析返回 0。"""
    result = 0
    current = 0
    for ch in text:
        if ch in _CN_DIGITS:
            current = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            unit = _CN_UNITS[ch]
            result += (current or 1) * unit   # "十二" 的 "十" 前无数字 → 按 1 处理
            current = 0
        else:
            return 0
    return result + current


def _section_order(node: SummaryNode) -> tuple[int, int]:
    """提取章节序号用于排序，兼容"第十二回"等中文数字与阿拉伯数字。

    返回 (章节序号, 起始 chunk) 二元组，chunk 起点兜底保证排序确定。
    """
    order = 0
    m = re.search(r"第([零一二两三四五六七八九十百千]+)[回章节篇]", node.section)
    if m:
        order = _cn_numeral_to_int(m.group(1))
    if order == 0:
        m = re.search(r"^([一二两三四五六七八九十]{1,3})、", node.section)
        if m:
            order = _cn_numeral_to_int(m.group(1))
    if order == 0:
        m = re.search(r"(\d+)", node.section)
        if m:
            order = int(m.group(1))
    return (order, node.chunk_range[0])


# ======================== L4: 全书摘要 ========================


def _get_chapter_level_nodes(
    l2_nodes: list[SummaryNode],
    l3_nodes: list[SummaryNode],
) -> list[SummaryNode]:
    """
    收集所有"章节级"摘要节点，用于生成 L4 全书摘要。

    规则：
      - 有 L3 节点的章节：用 L3 节点
      - 无 L3 节点的章节（无小节，L2 即章节）：用 L2 节点
    """
    # 哪些章节已有 L3
    sections_with_l3: set[str] = set()
    for node in l3_nodes:
        sections_with_l3.add(f"{node.file_name}|{node.section}")

    chapter_nodes: list[SummaryNode] = list(l3_nodes)
    for node in l2_nodes:
        if node.subsection == "":
            # 无小节章节，L2 即章节级
            chapter_nodes.append(node)
            continue
        key = f"{node.file_name}|{node.section}"
        if key not in sections_with_l3:
            # 该章节有多个小节但没生成 L3（理论上不应该，做防御）
            chapter_nodes.append(node)

    return chapter_nodes


def _generate_book_summary(
    l2_nodes: list[SummaryNode],
    l3_nodes: list[SummaryNode],
) -> list[SummaryNode]:
    """
    L3 → L4：合并所有章节级摘要，生成全书摘要。
    """
    chapter_nodes = _get_chapter_level_nodes(l2_nodes, l3_nodes)
    if len(chapter_nodes) <= 1:
        if len(chapter_nodes) == 1:
            _log("  L4 全书摘要：仅 1 个章节级节点，跳过")
        return []

    # 按章节顺序排序（as_completed 完成顺序不定，必须显式排序保证两次构建产物一致）
    chapter_nodes.sort(key=_section_order)

    group = SummaryGroup(
        children=chapter_nodes,
        file_name=chapter_nodes[0].file_name,
    )

    print(f"    L4 全书摘要：{len(chapter_nodes)} 个章节级节点 → 1 个全书摘要", flush=True)
    _log(f"  L4 全书摘要：{len(chapter_nodes)} 个章节级节点，{config.SUMMARY_BOOK_CHARS} 字上限")

    l4_node = _generate_parent_summary(
        group, level=4, use_original=False,
        max_chars=config.SUMMARY_BOOK_CHARS,
    )
    _log(f"  L4 全书摘要生成完成")
    return [l4_node]


# ======================== 统一入口 ========================


def build_summary_tree(nodes: list) -> tuple[list[Document], dict[str, dict]]:
    """
    构建完整的摘要树并返回可混入主索引的 Document 列表。

    四级结构：
      L1: 叶子摘要（chunk → ≤100字）
      L2: 小节摘要（小节原文 → 20%）
      L3: 章节摘要（章节原文 → 15%）
      L4: 全书摘要（合并L3 → ≤500字）

    Args:
        nodes: 分块后的原始节点列表

    Returns:
        (summary_docs, summary_meta_map)
        - summary_docs: 所有层级摘要节点对应的 Document 列表
        - summary_meta_map: node_id → 摘要元信息（供参考文献阶段使用）
    """
    total_chunks = len(nodes)
    if total_chunks == 0:
        _log("  无 chunk，跳过摘要树构建")
        return [], {}

    # 统计 section 和 subsection 数量
    sections_set = set()
    subsections_set = set()
    for node in nodes:
        sec = node.metadata.get("section", "")
        sub = node.metadata.get("subsection", "")
        sections_set.add(sec)
        if sub:
            subsections_set.add(f"{sec}|{sub}")

    print(f"  步骤 2.4：构建摘要树（共 {total_chunks} 个 chunk，"
          f"{len(sections_set)} 个 section，{len(subsections_set)} 个小节）...", flush=True)
    _log(f"步骤 2.6：构建摘要树（共 {total_chunks} 个 chunk，{len(sections_set)} 个 section）")

    # ---- 第 1 步：L1 叶子摘要 ----
    leaves = _generate_leaves(nodes)

    # ---- 第 2 步：L2 小节摘要 ----
    if total_chunks > 1:
        l2_nodes = _generate_subsection_summaries(leaves)
    else:
        l2_nodes = []
        _log("  仅 1 个 chunk，跳过上层摘要构建")

    # ---- 第 3 步：L3 章节摘要 ----
    if l2_nodes:
        l3_nodes = _generate_chapter_summaries(l2_nodes)
    else:
        l3_nodes = []

    # ---- 第 4 步：L4 全书摘要 ----
    chapter_nodes = _get_chapter_level_nodes(l2_nodes, l3_nodes)
    if len(chapter_nodes) > 1:
        l4_nodes = _generate_book_summary(l2_nodes, l3_nodes)
    else:
        l4_nodes = []

    all_summary_nodes = leaves + l2_nodes + l3_nodes + l4_nodes
    _log(f"  摘要树构建完成：L1 {len(leaves)} + L2 {len(l2_nodes)} + L3 {len(l3_nodes)} "
         f"+ L4 {len(l4_nodes)} = 共 {len(all_summary_nodes)} 个摘要节点")

    # 统计降级摘要（LLM 失败后用原文截断冒充的），显式暴露而非静默污染索引
    fallback_count = sum(1 for sn in all_summary_nodes if sn.is_fallback)
    if fallback_count > 0:
        fallback_ids = [sn.node_id for sn in all_summary_nodes if sn.is_fallback]
        logger.warning("有 %d 个摘要为降级产物（原文截断），建议检查后重建: %s",
                       fallback_count, fallback_ids[:10])
        _log(f"  ⚠️ 其中 {fallback_count} 个为降级摘要（LLM 失败，用原文截断代替，"
             f"已在 metadata 中标记 summary_fallback）")

    # ---- 第 5 步：转为 Document 列表 + 元信息映射 ----
    summary_docs: list[Document] = []
    summary_meta_map: dict[str, dict] = {}

    for sn in all_summary_nodes:
        # 获取 chunk_range 对应的原始文本
        if sn.original_text:
            original_text = sn.original_text[:config.RERANK_TEXT_MAX_LENGTH]
        elif sn.level == 1 and len(sn.child_ids) == 1:
            # 叶子：直接使用对应 chunk 的原文
            chunk_node = None
            for n in nodes:
                if n.node_id == sn.child_ids[0]:
                    chunk_node = n
                    break
            original_text = (chunk_node.text if chunk_node else "")[:config.RERANK_TEXT_MAX_LENGTH]
        else:
            # 父级 fallback：使用覆盖范围内 chunk 的拼接原文
            start, end = sn.chunk_range
            original_texts = []
            for n in nodes[start:end + 1]:
                original_texts.append(n.text)
            original_text = "\n\n".join(original_texts)[:config.RERANK_TEXT_MAX_LENGTH]

        doc = Document(
            text=sn.text,
            metadata={
                "summary_level": sn.level,
                "summary_child_ids": sn.child_ids,
                "summary_chunk_range": list(sn.chunk_range),
                "original_text": original_text,
                "file_name": sn.file_name,
                "section": sn.section,
                "subsection": sn.subsection,
                "is_summary": True,
                "summary_fallback": sn.is_fallback,
            },
            doc_id=sn.node_id,
        )
        summary_docs.append(doc)

        summary_meta_map[sn.node_id] = {
            "level": sn.level,
            "child_ids": sn.child_ids,
            "chunk_range": list(sn.chunk_range),
            "file_name": sn.file_name,
            "section": sn.section,
            "subsection": sn.subsection,
        }

    return summary_docs, summary_meta_map