"""rag/utils/json_parse.py 单测 —— LLM 输出解析的返回类型保证。"""

from rag.utils.json_parse import parse_json_obj, parse_json_list, coerce_index_set


class TestParseJsonObj:
    def test_clean_json(self):
        assert parse_json_obj('{"a": 1}') == {"a": 1}

    def test_json_with_surrounding_text(self):
        text = '以下是结果：\n{"valid": [0, 2]}\n以上。'
        assert parse_json_obj(text) == {"valid": [0, 2]}

    def test_markdown_fence(self):
        text = '```json\n{"a": "值"}\n```'
        assert parse_json_obj(text) == {"a": "值"}

    def test_trailing_comma_repaired(self):
        # json_repair 应能修复尾逗号
        assert parse_json_obj('{"a": 1,}') == {"a": 1}

    def test_list_input_returns_none(self):
        # LLM 返回数组时不能当成 dict（曾导致 data.get() 抛 AttributeError）
        assert parse_json_obj('[1, 2, 3]') is None

    def test_plain_text_returns_none(self):
        assert parse_json_obj("这不是 JSON") is None

    def test_empty_and_none(self):
        assert parse_json_obj("") is None
        assert parse_json_obj(None) is None


class TestParseJsonList:
    def test_clean_list(self):
        assert parse_json_list('[{"x": 1}, {"x": 2}]') == [{"x": 1}, {"x": 2}]

    def test_list_with_surrounding_text(self):
        assert parse_json_list('结果：["a", "b"]') == ["a", "b"]

    def test_dict_input_returns_none(self):
        assert parse_json_list('{"a": 1}') is None

    def test_empty_and_none(self):
        assert parse_json_list("") is None
        assert parse_json_list(None) is None


class TestCoerceIndexSet:
    def test_int_indices(self):
        assert coerce_index_set([0, 2, 5]) == {0, 2, 5}

    def test_string_indices(self):
        # LLM 常返回字符串下标（缺陷 #6：曾导致匹配全部失败、关系被整体误删）
        assert coerce_index_set(["0", "2"]) == {0, 2}

    def test_mixed_and_garbage(self):
        assert coerce_index_set([0, "1", "abc", None, 2.0]) == {0, 1, 2}

    def test_non_list_input(self):
        assert coerce_index_set("012") == set()
        assert coerce_index_set(None) == set()
        assert coerce_index_set({"a": 1}) == set()
