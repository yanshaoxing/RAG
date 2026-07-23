"""rag/retrieval/query_decomposer.py 单测 —— 复杂度判断（锁定缺陷 #3）与子查询解析。"""

from rag.retrieval.query_decomposer import QueryDecomposer


class _FakeLLM:
    """chat() 返回固定文本的假 LLM。"""

    def __init__(self, reply: str):
        self._reply = reply
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1

        class _Msg:
            content = self._reply

        class _Resp:
            message = _Msg()

        return _Resp()


class TestHeuristicCheck:
    def test_simple_query_not_complex(self):
        assert not QueryDecomposer._heuristic_check("丁元英是谁？")

    def test_two_keywords_complex(self):
        assert QueryDecomposer._heuristic_check("丁元英和芮小丹分别是什么样的人？")

    def test_double_question_marks_complex(self):
        assert QueryDecomposer._heuristic_check("丁元英是谁？他做了什么？")


class TestLlmClassify:
    """缺陷 #3 回归：'不是' 包含 '是' 字，必须前缀匹配而非子串匹配。"""

    def _classify(self, reply: str) -> bool:
        d = QueryDecomposer(llm=_FakeLLM(reply))
        return d._llm_classify("测试查询")

    def test_bushi_is_not_complex(self):
        assert self._classify("不是") is False

    def test_fou_is_not_complex(self):
        assert self._classify("否") is False

    def test_shi_is_complex(self):
        assert self._classify("是") is True

    def test_shi_with_punct_prefix(self):
        assert self._classify("：是的，这是复杂查询") is True

    def test_unrecognized_defaults_false(self):
        # 无法识别时保守不拆解，避免误拆放大延迟
        assert self._classify("这个问题很有意思") is False

    def test_llm_failure_defaults_false(self):
        class _BrokenLLM:
            def chat(self, messages, **kwargs):
                raise RuntimeError("network down")

        d = QueryDecomposer(llm=_BrokenLLM())
        assert d._llm_classify("测试查询") is False


class TestParseSubQueries:
    def test_numbered_lines(self):
        resp = "1. 丁元英是谁？\n2. 芮小丹是谁？"
        assert QueryDecomposer._parse_sub_queries(resp) == ["丁元英是谁？", "芮小丹是谁？"]

    def test_various_prefixes(self):
        resp = "- 第一个子查询内容\n* 第二个子查询内容\n子问题3：第三个子查询内容"
        parsed = QueryDecomposer._parse_sub_queries(resp)
        assert parsed == ["第一个子查询内容", "第二个子查询内容", "第三个子查询内容"]

    def test_short_lines_dropped(self):
        resp = "1. 有效的子查询问题\n2. 短"
        assert QueryDecomposer._parse_sub_queries(resp) == ["有效的子查询问题"]


class TestDecompose:
    def test_disabled_passthrough(self):
        d = QueryDecomposer(llm=_FakeLLM("是"), enabled=False)
        assert d.decompose("任何查询") == (False, ["任何查询"])

    def test_simple_query_no_llm_split(self):
        # 启发式判简单 + LLM 判"否" → 不拆解
        d = QueryDecomposer(llm=_FakeLLM("否"))
        is_complex, subs = d.decompose("丁元英是谁")
        assert is_complex is False
        assert subs == ["丁元英是谁"]
