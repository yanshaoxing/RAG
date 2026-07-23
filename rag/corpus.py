"""语料档案（Corpus Profile）—— 多书 RAG 的每本书配置。

每本书一个目录 corpora/<slug>/：
  corpus.json        必需：title / context（注入 prompt 的原著背景），可选 author / description
  raw/               原文（txt / docx）
  terminology.json   可选：通俗名 → 原文术语映射（缺省时术语映射跳过）
  graph_rules.json   可选：图谱规则补充（与 rag/graph/rules.json 基础规则合并）
  data/              5 阶段索引持久化目录（重建时整体删除即可）

激活语料启动时由环境变量 RAG_CORPUS 选择（默认 config.DEFAULT_CORPUS），
运行期可用 set_active_corpus() 切换；config.py 的持久化路径常量（动态属性）
与 prompts.py 的模板注入均实时跟随激活语料。

多语料并存约定：查询期依赖语料的状态（索引 / 图存储 / QueryRewriter 的
prompt 与术语表）全部在引擎构建时绑定，切换激活语料不影响已构建的引擎；
构建期组件（摘要树 / 图谱 prompt、图规则）读取激活语料，因此构建必须在
bootstrap 的构建锁内、切换激活语料后进行。
"""

import json
import logging
import os
import threading
from dataclasses import dataclass, field

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


# ---------- 激活语料（进程级状态，启动默认 = RAG_CORPUS / config.DEFAULT_CORPUS） ----------

_active_lock = threading.Lock()
_active_slug: str = config.ACTIVE_CORPUS
_profile_cache: dict[str, CorpusProfile] = {}


def get_active_slug() -> str:
    """当前激活语料的 slug。"""
    return _active_slug


def set_active_corpus(slug: str) -> CorpusProfile:
    """切换激活语料（先校验档案可加载，失败不改变现状态）。

    切换只影响之后的构建/prompt 渲染；已构建的引擎绑定各自语料，不受影响。
    """
    global _active_slug
    profile = load_profile(slug)  # 校验：目录/corpus.json/必需字段
    with _active_lock:
        if slug != _active_slug:
            logger.info("切换激活语料：%s → %s（《%s》）", _active_slug, slug, profile.title)
            _active_slug = slug
        _profile_cache[slug] = profile
    return profile


def get_active_profile() -> CorpusProfile:
    """激活语料的档案（按 slug 缓存，进程内同一 slug 只读一次盘）。"""
    slug = _active_slug
    profile = _profile_cache.get(slug)
    if profile is None:
        profile = load_profile(slug)
        with _active_lock:
            _profile_cache[slug] = profile
        logger.info("激活语料：%s（《%s》）", profile.slug, profile.title)
    return profile


def list_corpora() -> list[CorpusProfile]:
    """扫描 corpora/ 下所有含 corpus.json 的语料目录，按 slug 排序返回档案列表。

    档案损坏的目录记 warning 后跳过，不影响其他语料。
    """
    root = config.CORPORA_ROOT
    if not os.path.isdir(root):
        return []
    profiles = []
    for name in sorted(os.listdir(root)):
        if not os.path.exists(os.path.join(root, name, "corpus.json")):
            continue
        try:
            profiles.append(load_profile(name))
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning("语料 %s 的档案无效，已跳过：%s", name, e)
    return profiles
