"""
索引管理 —— 委托给分阶段管线 staged_indexer。

所有构建/加载逻辑已迁移到 staged_indexer.py，此模块仅为兼容性包装。
"""

import logging
from typing import Optional, Tuple

from llama_index.core import VectorStoreIndex
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.retrievers.bm25 import BM25Retriever

from rag.indexing.staged_indexer import get_or_build_index as _staged_get_or_build

logger = logging.getLogger(__name__)


# ======================== 分词工具（retriever.py 可能直接引用） ========================

def tokenize_for_bm25(text: str) -> str:
    """中文分词后空格连接，供 BM25 索引构建和检索使用。"""
    import jieba
    return " ".join(jieba.cut(text))


# ======================== 统一入口 ========================

def get_or_build_index(
    log_list: Optional[list] = None,
) -> Tuple[VectorStoreIndex, BM25Retriever, Optional[dict], Optional[PropertyGraphIndex]]:
    """分阶段构建/加载索引，返回 (向量索引, BM25检索器, 摘要树元数据映射, 知识图谱索引)。

    5 个阶段产物独立持久化：
      - chunks/       分块节点
      - summary_tree/ 摘要树元数据 + 摘要节点
      - bm25/         BM25 索引
      - vector/       向量索引
      - data/graph_db/  知识图谱（Kuzu）

    若中途失败，只需删除未完成阶段的文件夹即可从中断处继续。
    """
    return _staged_get_or_build(log_list)