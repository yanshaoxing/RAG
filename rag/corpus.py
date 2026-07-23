"""语料档案（Corpus Profile）—— 多书 RAG 的每本书配置。

每本书一个目录 corpora/<slug>/：
  corpus.json        必需：title / context（注入 prompt 的原著背景），可选 author / description
  raw/               原文（txt / docx）
  terminology.json   可选：通俗名 → 原文术语映射（缺省时术语映射跳过）
  graph_rules.json   可选：图谱规则补充（与 rag/graph/rules.json 基础规则合并）
  data/              5 阶段索引持久化目录（重建时整体删除即可）

激活语料由环境变量 RAG_CORPUS 选择（默认 config.DEFAULT_CORPUS），
config.py 的持久化路径常量全部随之派生；prompts.py 的模板在访问时
注入激活语料的 title / context（见 prompts.__getattr__）。
"""

import json
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache

from rag import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorpusProfile:
    """一本书的档案：prompt 注入所需的背景信息 + 语料目录路径。"""

    slug: str                       # 目录名，即 corpora/<slug>/
    title: str                      # 书名（注入 prompt 的 {book_title}）
    context: str                    # 原著背景块（注入 prompt 的 {corpus_context}）
    author: str = ""
    description: str = ""
    corpus_dir: str = field(default="", repr=False)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.corpus_dir, "raw")

    @property
    def terminology_path(self) -> str:
        return os.path.join(self.corpus_dir, "terminology.json")

    @property
    def graph_rules_path(self) -> str:
        return os.path.join(self.corpus_dir, "graph_rules.json")

    @property
    def data_dir(self) -> str:
        return os.path.join(self.corpus_dir, "data")


def load_profile(slug: str) -> CorpusProfile:
    """加载 corpora/<slug>/corpus.json 为 CorpusProfile。

    缺目录 / 缺 corpus.json / 缺必需字段（title、context）时抛出带修复提示的异常。
    """
    corpus_dir = os.path.join(config.CORPORA_ROOT, slug)
    profile_path = os.path.join(corpus_dir, "corpus.json")
    if not os.path.exists(profile_path):
        raise FileNotFoundError(
            f"语料档案不存在：{profile_path}。"
            f"请确认 RAG_CORPUS={slug} 正确，且 corpora/{slug}/corpus.json 已创建"
            f"（必需字段：title、context）。"
        )
    with open(profile_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    missing = [k for k in ("title", "context") if not data.get(k)]
    if missing:
        raise ValueError(f"语料档案 {profile_path} 缺少必需字段：{missing}")
    return CorpusProfile(
        slug=slug,
        title=data["title"],
        context=data["context"],
        author=data.get("author", ""),
        description=data.get("description", ""),
        corpus_dir=corpus_dir,
    )


@lru_cache(maxsize=None)
def get_active_profile() -> CorpusProfile:
    """激活语料的档案（进程内缓存；激活语料由 RAG_CORPUS / config.ACTIVE_CORPUS 决定）。"""
    profile = load_profile(config.ACTIVE_CORPUS)
    logger.info("激活语料：%s（《%s》）", profile.slug, profile.title)
    return profile
