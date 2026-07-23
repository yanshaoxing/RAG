"""
文档预处理模块 —— 章节感知分割 + 文档加载 + 分块管道。

供 rag/indexing/staged_indexer.py 调用，两个入口（app/cli.py / app/ui.py）不直接使用本模块。
"""

import re
import logging
from pathlib import Path
from typing import Optional

from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import BaseNode, TextNode, TransformComponent

import docx2txt

from rag import config

logger = logging.getLogger(__name__)

# chunk 节点中不进入 embedding 输入 / LLM 上下文的元数据键：
# section_path 恒等于 section（重复注入无意义）。file_name/section/subsection
# 保留 —— 少量前缀上下文对嵌入与溯源都有益。
# staged_indexer 反序列化 chunk 节点时必须重新应用（序列化不保存排除键）。
CHUNK_EXCLUDED_META_KEYS = ["section_path"]


# ============================================================
# 章节 / 子章节模式

# 小节标记：全角空格 + 1~2 位数字独占一行（如 "　　1"、"　　2"），
# 或 "第X节"（阿拉伯数字）独占一行
_SUB_PATTERN = re.compile(r"^(?:　+\d{1,2}|第\d{1,2}节)\s*$", re.MULTILINE)
# ============================================================

# 一级标题：一、二、三…十、…二十、三十…
_L1_PATTERN_DUN = re.compile(r"[一二三四五六七八九十]{1,3}、")

# 一级标题（回/章/节/篇）：第一回、第二十二回、第三章、第一篇 …
_L1_PATTERN_HUI = re.compile(r"第[一二三四五六七八九十百千]+[回章节篇]")

# 一级标题统一匹配（用于 title 校验，合并两种模式）
_L1_PATTERN = re.compile(r"[一二三四五六七八九十]{1,3}、|第[一二三四五六七八九十百千]+[回章节篇]")

def _detect_l1_format(text: str) -> str:
    """
    自动检测一级标题格式。

    扫描文本前 3000 字符，比较「一、」和「第X回/章/节/篇」两种模式的命中数，
    返回 "dun" 或 "hui"。
    """
    sample = text[:3000]
    dun_count = len(_L1_PATTERN_DUN.findall(sample))
    hui_count = len(_L1_PATTERN_HUI.findall(sample))
    return "hui" if hui_count > dun_count else "dun"


# ============================================================
# 章节分割
# ============================================================

def split_by_section(text: str) -> list[dict]:
    """
    按章节标题将全文拆分为带元数据的段落列表。

    自动检测一级标题格式，支持：
      - 「一、二、三、…」（中文数字 + 顿号）
      - 「第一回、第二回、…」（第X回/章/节/篇）

    返回:
        [{"section": "第一回 灵根育孕源流出...",
          "section_path": "第一回 灵根育孕源流出 > （二）贾府兴衰",
          "content": "该章节完整文本"}, ...]
    """
    l1_fmt = _detect_l1_format(text)
    if l1_fmt == "hui":
        pattern = r"\n[ 　]*(?=第[一二三四五六七八九十百千]+[回章节篇])"
    else:
        pattern = r"\n[ 　]*(?=[一二三四五六七八九十]{1,3}、)"
    parts = re.split(pattern, text)

    sections = []
    for part in parts:
        part = part.strip()
        if not part:
            continue

        first_line_end = part.find("\n")
        if first_line_end == -1:
            title = part
            body = ""
        else:
            title = part[:first_line_end].strip()
            body = part[first_line_end:].strip()

        # 如果第一行是章节标题，正文从第二行开始（标题已在 section 元数据中）
        if _L1_PATTERN.match(title):
            sections.append({"section": title, "content": body, "section_path": title})
        else:
            sections.append({"section": "概述", "content": part, "section_path": "概述"})

    return sections


# ============================================================
# 小节分割
# ============================================================

def _split_body_by_subsections(body: str) -> list[dict]:
    """
    将章节正文按小节标记（"　　1"、"　　2"等）拆分为小节列表。

    小节标记：以全角空格开头 + 1~2 位数字 + 可选空白组成的行。

    返回:
        [{"subsection": "1", "content": "该小节正文..."}, ...]
        若无小节标记，返回 [{"subsection": "", "content": body}]
    """
    sub_matches = list(_SUB_PATTERN.finditer(body))
    if not sub_matches:
        return [{"subsection": "", "content": body}]

    subs = []
    # 章节标题与第一个小节标记之间的正文不能丢弃（否则该部分不可检索、不进摘要和图谱）
    preamble = body[: sub_matches[0].start()].strip()
    if preamble:
        subs.append({"subsection": "", "content": preamble})

    for i, sm in enumerate(sub_matches):
        sub_label = re.search(r"\d+", sm.group()).group()
        sub_start = sm.end()
        sub_end = sub_matches[i + 1].start() if i + 1 < len(sub_matches) else len(body)
        sub_text = body[sub_start:sub_end].strip()
        if sub_text:
            subs.append({"subsection": sub_label, "content": sub_text})
    return subs


# ============================================================
# 文档加载
# ============================================================

def load_documents(data_dir: Optional[str] = None) -> list[Document]:
    """
    从 data 目录加载所有 .docx / .txt 文件，按章节→小节拆分为 Document 列表。

    Args:
        data_dir: 数据目录路径，默认使用 config.DATA_DIR

    Returns:
        Document 列表，每个 Document 带 file_name、section、subsection 和 section_path 元数据
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    raw_documents = []
    for fp in sorted(Path(data_dir).glob("*")):
        suffix = fp.suffix.lower()
        if suffix == ".docx":
            text = docx2txt.process(str(fp))
        elif suffix == ".txt":
            with open(fp, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            continue

        sections = split_by_section(text)
        for sec in sections:
            subs = _split_body_by_subsections(sec["content"])
            for sub in subs:
                doc = Document(
                    text=sub["content"],
                    metadata={
                        "file_name": fp.name,
                        "section": sec["section"],
                        "subsection": sub["subsection"],
                        "section_path": sec["section_path"],
                    },
                    # HierarchicalChunker._make_node 会把排除键继承给所有 chunk 节点
                    excluded_embed_metadata_keys=list(CHUNK_EXCLUDED_META_KEYS),
                    excluded_llm_metadata_keys=list(CHUNK_EXCLUDED_META_KEYS),
                )
                raw_documents.append(doc)

    return raw_documents


# ============================================================
# 分块器 TransformComponent
# ============================================================

# ============================================================
# 语义边界检测（模块级工具函数）
# ============================================================

# 汉字 Unicode 范围
_HANZI_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]')

# 句子结束标点
_SENT_PUNCT = frozenset(['\u3002', '\uff01', '\uff1f'])  # 。！？

# 省略号（两个 U+2026）
_ELLIPSIS = '\u2026\u2026'  # ……

# 中文右双引号
_RIGHT_DQUOTE = '\u201d'  # "

# 条件 ii 双标点组合（不含 ……"）
_DOUBLE_PUNCT_PAIRS = frozenset([
    '\u3002' + _RIGHT_DQUOTE,   # 。"
    '\uff01' + _RIGHT_DQUOTE,   # ！"
    '\uff1f' + _RIGHT_DQUOTE,   # ？"
    _RIGHT_DQUOTE + '\u3002',   # "。
    '\uff09' + '\u3002',        # ）。
])

# 条件 ii 三字符组合：……"
_TRIPLE_PUNCT = _ELLIPSIS + _RIGHT_DQUOTE


def _is_chinese_char(ch: str) -> bool:
    """判断字符是否为汉字。"""
    return _HANZI_RE.match(ch) is not None


def _find_semantic_boundary_right(text: str, start_pos: int) -> int:
    """
    从 start_pos 向右逐字符扫描，找到第一个符合语义规则的切分点。

    检查优先级（先 ii 后 i，避免双标点被从中间切断）：
      ii. ……" + 汉字 → 切在 +3 处（第二个标点右边）
      ii. 。" / ！" / ？" / "。 / ）。 + 汉字 → 切在 +2 处
      i.  …… + 汉字 → 切在 +2 处（标点右边）
      i.  。 / ！ / ？ + 汉字 → 切在 +1 处

    未找到匹配则返回 len(text)，表示从此处起到末尾无合适切分点。
    """
    n = len(text)
    pos = max(0, start_pos)
    while pos < n:
        # ---- 条件 ii（三字符）: ……" + 汉字 ----
        if pos + 3 < n and text[pos:pos + 3] == _TRIPLE_PUNCT and _is_chinese_char(text[pos + 3]):
            return pos + 3

        # ---- 条件 ii（双字符）: 。" / ！" / ？" / "。 / ）。 + 汉字 ----
        if pos + 2 < n and text[pos:pos + 2] in _DOUBLE_PUNCT_PAIRS and _is_chinese_char(text[pos + 2]):
            return pos + 2

        # ---- 条件 i: …… + 汉字 ----
        if pos + 2 < n and text[pos:pos + 2] == _ELLIPSIS and _is_chinese_char(text[pos + 2]):
            return pos + 2

        # ---- 条件 i: 。/！/？ + 汉字 ----
        if pos + 1 < n and text[pos] in _SENT_PUNCT and _is_chinese_char(text[pos + 1]):
            return pos + 1

        pos += 1

    return n  # 未找到，返回文本末尾作为兜底


class HierarchicalChunker(TransformComponent):
    """
    语义边界向右扫描分块器。

    所有边界查找一律向右扫描，逐字符检测是否满足语义切分条件：
      ii. 双标点 + 汉字：。" / ！" / ？" / "。 / ）。 / ……"
      i.  单标点 + 汉字：。 / ！ / ？ / ……
    先检测 ii 再检测 i，避免连续标点被从中间切开。

    保证每个 chunk 大小约等于 chunk_size、相邻块之间有 overlap，且边界语义完整。
    不跨自然段。
    """

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        chunk_size = config.CHUNK_SIZE
        overlap = config.CHUNK_OVERLAP
        results: list[BaseNode] = []
        stats = {"total_chars": 0, "chunks": 0}

        for node in nodes:
            text = node.text or ""
            if not text.strip():
                continue
            stats["total_chars"] += len(text)

            # 直接对章节全文做终点驱动的滑动窗口分块
            chunks_info = self._sliding_window(text, chunk_size, overlap)
            stats["chunks"] += len(chunks_info)

            for chunk_info in chunks_info:
                results.append(self._make_node(chunk_info["text"], node))

        logger.info(
            "终点驱动分块: chars=%d → chunks=%d",
            stats["total_chars"], stats["chunks"],
        )
        return results

    # ---------- 语义边界向右扫描分块核心逻辑 ----------

    @staticmethod
    def _do_tail_balance(text: str, L2_L: int, L1_R: int,
                         chunk_size: int, overlap_max: int) -> list[dict]:
        """
        尾块均衡：将 text[L2_L:L1_R] 拆成 2 个大小接近且有 overlap 的 chunk。

        L2_R 从 mid + half_ov 向右扫描语义边界；
        L1_L 从 L2_R - overlap_max 向右扫描语义边界。
        """
        half_ov = overlap_max // 2
        mid = (L2_L + L1_R) // 2

        # L2_R：从 mid + half_ov 向右扫描
        L2_R = _find_semantic_boundary_right(text, mid + half_ov)
        if L2_R >= L1_R:
            L2_R = L1_R - 1
        if L2_R <= L2_L:
            L2_R = min(L2_L + chunk_size, L1_R)

        # L1_L：从 L2_R - overlap_max 向右扫描
        L1_L_search_start = max(L2_L + 1, L2_R - overlap_max)
        L1_L = _find_semantic_boundary_right(text, L1_L_search_start)
        if L1_L >= L1_R:
            L1_L = L1_R - 1
        # 确保 L1_L ≤ L2_R（无间隙），若扫描越过了则回退
        if L1_L > L2_R:
            L1_L = L2_R

        result: list[dict] = []
        seg_L2 = text[L2_L:L2_R].strip()
        seg_L1 = text[L1_L:L1_R].strip()
        if seg_L2:
            result.append({"text": seg_L2, "left": L2_L, "right": L2_R})
        if seg_L1:
            result.append({"text": seg_L1, "left": L1_L, "right": L1_R})
        return result

    @staticmethod
    def _sliding_window(
        text: str, chunk_size: int, overlap_max: int
    ) -> list[dict]:
        """
        语义边界向右扫描 + 动态重叠分块算法。

        chunk_size=1024, overlap_max=102。

        0. text_len ≤ chunk_size → 1 个 chunk，直接返回。
        1. chunk_size < text_len ≤ 2*chunk_size → 直接尾块均衡（L2_L=0）。
        2. text_len > 2*chunk_size → 正常流程：
           - 首块 Chunk1：L=0, R 从 chunk_size 向右扫描
           - 循环后续块：L 从 prev_R - overlap 向右扫描, R 从 L+chunk_size 向右扫描
           - 尾块均衡：remaining ≤ 2*chunk_size 时触发
        """
        text_len = len(text)

        # ====== 情况 0：文本很短，直接整段作为 1 个 chunk ======
        if text_len <= chunk_size:
            segment = text.strip()
            if segment:
                return [{"text": segment, "left": 0, "right": text_len}]
            return []

        chunks: list[dict] = []

        # ====== 情况 1：中等长度，直接尾块均衡 ======
        if text_len <= 2 * chunk_size:
            return HierarchicalChunker._do_tail_balance(
                text, L2_L=0, L1_R=text_len,
                chunk_size=chunk_size, overlap_max=overlap_max,
            )

        # ====== 情况 2：长文本，正常流程 ======

        # ----- Chunk 1 -----
        R1 = _find_semantic_boundary_right(text, chunk_size)
        if R1 > text_len:
            R1 = text_len
        segment = text[0:R1].strip()
        if segment:
            chunks.append({"text": segment, "left": 0, "right": R1})

        prev_right = R1
        if prev_right >= text_len:
            return chunks

        # ----- 后续 Chunk -----
        while prev_right < text_len:
            remaining = text_len - prev_right

            # --- 尾块均衡 ---
            if remaining <= 2 * chunk_size:
                tail = HierarchicalChunker._do_tail_balance(
                    text, L2_L=prev_right, L1_R=text_len,
                    chunk_size=chunk_size, overlap_max=overlap_max,
                )
                chunks.extend(tail)
                break

            # --- 正常分块 ---
            # 左边界：从 prev_right - overlap_max 向右扫描，但不能越过 prev_right
            cur_L_search_start = max(0, prev_right - overlap_max)
            cur_L = _find_semantic_boundary_right(text, cur_L_search_start)
            if cur_L > prev_right:
                cur_L = prev_right

            # 右边界：从 cur_L + chunk_size 向右扫描
            cur_R = _find_semantic_boundary_right(text, cur_L + chunk_size)

            if cur_L >= cur_R:
                cur_R = min(cur_L + chunk_size, text_len)

            segment = text[cur_L:cur_R].strip()
            if segment:
                chunks.append({"text": segment, "left": cur_L, "right": cur_R})

            prev_right = cur_R
            if prev_right >= text_len:
                break

        return chunks


    @staticmethod
    def _make_node(text: str, template: BaseNode) -> BaseNode:
        """创建带继承元数据的 TextNode。"""
        return TextNode(
            text=text,
            metadata=dict(template.metadata),
            excluded_embed_metadata_keys=template.excluded_embed_metadata_keys,
            excluded_llm_metadata_keys=template.excluded_llm_metadata_keys,
        )


# ============================================================
# 分块管道
# ============================================================

def create_chunking_pipeline() -> IngestionPipeline:
    """
    创建分块管道。

    章节分割已在 load_documents 阶段通过 split_by_section 完成，
    HierarchicalChunker 直接对章节全文做终点驱动 + 语义边界对齐重叠分块。
    """
    return IngestionPipeline(transformations=[HierarchicalChunker()])
