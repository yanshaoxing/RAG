"""
文本工具 —— BM25 分词。

BM25 语料在索引期用 jieba 分词后以空格连接（见 rag/indexing/staged_indexer.py），
查询期送入 BM25 的字符串也必须是同一分词形式：bm25s 的默认 tokenizer 不切中文，
原始整句中文会成为单个 token，与分词后语料几乎无法匹配（实测得分全 0）。
统一从本模块取 tokenize_for_bm25，保证索引期与查询期分词一致。
"""

import jieba


def tokenize_for_bm25(text: str) -> str:
    """中文分词后空格连接，供 BM25 索引构建和检索使用。

    对已是空格分隔的关键词串同样适用：jieba 会把关键词进一步切成与
    语料一致的词粒度（语料 token 即 jieba token，粒度对齐才能命中）。
    """
    return " ".join(t for t in jieba.cut(text) if t.strip())
