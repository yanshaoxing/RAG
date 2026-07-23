"""rag/prompts.py 单测 —— 每个模板都能用其运行时占位符成功 .format（守护 JSON 花括号转义）。"""

import pytest

from rag import prompts

# 模板 → 运行时 .format 关键字（与各调用点一致）
_TEMPLATE_KWARGS = {
    "REWRITE_NL_PROMPT": {"query": "问"},
    "REWRITE_HYDE_PROMPT": {"query": "问"},
    "REWRITE_KW_PROMPT": {"query": "问"},
    "DECOMPOSE_CLASSIFY_PROMPT": {"query": "问"},
    "DECOMPOSE_PROMPT": {"query": "问", "max_sub": 5},
    "SUMMARY_LEAF_BATCH_PROMPT": {"chunks": "片段"},
    "SUMMARY_PARENT_FROM_RAW_PROMPT": {"max_chars": 200, "section_name": "第一回", "chapter_text": "文"},
    "SUMMARY_PARENT_PROMPT": {"max_chars": 200, "child_summaries": "摘要"},
    "GRAPH_EXTRACT_PROMPT": {"chunk_text": "文"},
    "GRAPH_VALIDATE_PROMPT": {"chunk_text": "文", "triples_text": "三元组"},
    "CANONICALIZE_PROMPT": {"candidate": "元英", "known_names": "- 丁元英"},
    "MERGE_PROMPT": {"existing_desc": "旧", "new_desc": "新"},
    "ENTITY_EXTRACT_FROM_QUERY_PROMPT": {"query": "问"},
    "QA_TEMPLATE_STR": {"context_str": "资料", "query_str": "问"},
}


@pytest.mark.parametrize("name", sorted(_TEMPLATE_KWARGS))
def test_template_formats_cleanly(name):
    template = getattr(prompts, name)
    kwargs = _TEMPLATE_KWARGS[name]
    result = template.format(**kwargs)  # 占位符错误会抛 KeyError/IndexError
    for v in kwargs.values():
        assert str(v) in result


def test_novel_context_injected_into_rewrite_prompts():
    # {novel_context} 标记应已被替换为原著背景，不残留
    for name in ("REWRITE_NL_PROMPT", "REWRITE_HYDE_PROMPT", "REWRITE_KW_PROMPT"):
        template = getattr(prompts, name)
        assert "{novel_context}" not in template
        assert "遥远的救世主" in template
        assert "丁元英" in template


def test_no_double_escaped_query_placeholder():
    # 旧写法的 {{query}} 双层转义已消除，运行时占位符是单层 {query}
    for name in ("REWRITE_NL_PROMPT", "REWRITE_HYDE_PROMPT", "REWRITE_KW_PROMPT"):
        assert "{{query}}" not in getattr(prompts, name)
