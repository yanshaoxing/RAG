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
    "EVAL_JUDGE_TEMPLATE_STR": {
        "question": "问", "num_points": 2, "gold_points": "- 要点", "answer": "答",
    },
}


@pytest.mark.parametrize("name", sorted(_TEMPLATE_KWARGS))
def test_template_formats_cleanly(name):
    template = getattr(prompts, name)
    kwargs = _TEMPLATE_KWARGS[name]
    result = template.format(**kwargs)  # 占位符错误会抛 KeyError/IndexError
    for v in kwargs.values():
        assert str(v) in result


def test_corpus_markers_injected():
    # {book_title} / {corpus_context} 标记应已被替换为激活语料的书名/背景，不残留
    from rag.corpus import get_active_profile

    profile = get_active_profile()
    for name in sorted(prompts._RAW_TEMPLATES):
        template = getattr(prompts, name)
        assert "{book_title}" not in template, name
        assert "{corpus_context}" not in template, name
        assert profile.title in template, name
    # 三路改写模板还应包含完整背景块
    for name in ("REWRITE_NL_PROMPT", "REWRITE_HYDE_PROMPT", "REWRITE_KW_PROMPT"):
        assert profile.context in getattr(prompts, name)


def test_raw_templates_have_no_book_hardcoding():
    # 原始模板不得硬编码具体书名/人物（多书 RAG：语料信息只经档案注入）
    for name, raw in prompts._RAW_TEMPLATES.items():
        assert "遥远的救世主" not in raw, name
        assert "丁元英" not in raw, name


def test_no_double_escaped_query_placeholder():
    # 旧写法的 {{query}} 双层转义已消除，运行时占位符是单层 {query}
    for name in ("REWRITE_NL_PROMPT", "REWRITE_HYDE_PROMPT", "REWRITE_KW_PROMPT"):
        assert "{{query}}" not in getattr(prompts, name)
