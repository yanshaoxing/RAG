"""新缺陷 1 回归 —— 摘要/chunk 节点的元数据不得泄漏进 embedding 输入与 LLM 上下文。

LlamaIndex 默认把全部 metadata 以 "key: value" 拼在正文前；
original_text（≤8192 字）与 summary_child_ids（UUID 列表）一旦进入
嵌入输入会主导摘要向量、进入 QA prompt 会注入 ~8KB 冗余。
"""

from llama_index.core import Document
from llama_index.core.schema import MetadataMode, TextNode

from rag import config
from rag.indexing.staged_indexer import (
    _deserialize_nodes,
    _deserialize_summary_docs,
    _stage_bm25,
)
from rag.ingestion.preprocessor import CHUNK_EXCLUDED_META_KEYS, HierarchicalChunker
from rag.summarization.summary_tree import SUMMARY_EXCLUDED_META_KEYS

_SUMMARY_ITEM = {
    "doc_id": "summary_L2_0_3",
    "text": "这是摘要正文",
    "metadata": {
        "summary_level": 2,
        "summary_child_ids": ["uuid-aaaa", "uuid-bbbb"],
        "summary_chunk_range": [0, 3],
        "original_text": "很长的原文内容" * 100,
        "file_name": "book.txt",
        "section": "第一章",
        "subsection": "",
        "is_summary": True,
        "summary_fallback": False,
    },
}


class TestSummaryDocExclusion:
    def test_deserialized_summary_embed_content_clean(self):
        doc = _deserialize_summary_docs([_SUMMARY_ITEM])[0]
        embed_text = doc.get_content(metadata_mode=MetadataMode.EMBED)
        assert "这是摘要正文" in embed_text
        assert "很长的原文内容" not in embed_text
        assert "uuid-aaaa" not in embed_text
        assert "summary_chunk_range" not in embed_text

    def test_deserialized_summary_llm_content_clean(self):
        doc = _deserialize_summary_docs([_SUMMARY_ITEM])[0]
        llm_text = doc.get_content(metadata_mode=MetadataMode.LLM)
        assert "这是摘要正文" in llm_text
        assert "很长的原文内容" not in llm_text
        assert "uuid-aaaa" not in llm_text
        # 溯源信息保留
        assert "第一章" in llm_text

    def test_exclusion_keys_cover_all_bulk_metadata(self):
        # 常量本身的回归：大体积/结构性键必须全部在排除列表中
        for key in ("original_text", "summary_child_ids", "summary_chunk_range"):
            assert key in SUMMARY_EXCLUDED_META_KEYS


class TestChunkNodeExclusion:
    def test_deserialized_chunk_excludes_section_path(self):
        item = {
            "node_id": "c1",
            "text": "chunk 正文",
            "metadata": {"file_name": "book.txt", "section": "第一章",
                         "subsection": "1", "section_path": "第一章"},
        }
        node = _deserialize_nodes([item])[0]
        for mode in (MetadataMode.EMBED, MetadataMode.LLM):
            content = node.get_content(metadata_mode=mode)
            assert "section_path" not in content
            assert "第一章" in content  # section 本身保留

    def test_chunker_propagates_exclusions(self):
        template = Document(
            text="正文",
            metadata={"section_path": "第一章", "section": "第一章"},
            excluded_embed_metadata_keys=list(CHUNK_EXCLUDED_META_KEYS),
            excluded_llm_metadata_keys=list(CHUNK_EXCLUDED_META_KEYS),
        )
        node = HierarchicalChunker._make_node("切出的 chunk", template)
        assert "section_path" in node.excluded_embed_metadata_keys
        assert "section_path" in node.excluded_llm_metadata_keys


class TestBM25NodeExclusion:
    def test_stage_bm25_excludes_original_text(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "BM25_DIR", str(tmp_path / "bm25"))
        chunk = TextNode(id_="c1", text="丁元英 创办 私募基金",
                         metadata={"section": "第一章"})
        retriever = _stage_bm25([chunk])
        retriever.similarity_top_k = 1  # 语料只有 1 条
        # 经真实检索路径取回节点（含序列化/反序列化），验证排除键随节点保留
        results = retriever.retrieve("丁元英")
        assert len(results) == 1
        n = results[0].node
        # original_text 已写入 metadata 供检索后恢复……
        assert n.metadata["original_text"].endswith("丁元英 创办 私募基金")
        # ……但绝不能进入 LLM 上下文（BM25 命中节点直接进 QA prompt）
        llm_text = n.get_content(metadata_mode=MetadataMode.LLM)
        assert "original_text" not in llm_text
