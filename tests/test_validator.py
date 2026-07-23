"""rag/graph/validator.py 单测 —— 锁定缺陷 #6（阈值语义 / 单关系放行 / 字符串下标）。"""

from rag.graph.models import Relation
from rag.graph.validator import Validator


class _FakeLLM:
    """complete() 返回固定文本的假 LLM，记录调用次数。"""

    def __init__(self, reply: str):
        self._reply = reply
        self.calls = 0

    def complete(self, prompt: str):
        self.calls += 1

        class _Resp:
            text = self._reply

        return _Resp()


class _BrokenLLM:
    def complete(self, prompt: str):
        raise RuntimeError("network down")


def _rel(subject: str, obj: str, confidence: float) -> Relation:
    return Relation(subject=subject, predicate="认识", object=obj, confidence=confidence)


def _validator(llm, threshold: float = 2.0) -> Validator:
    return Validator(
        validate_llm=llm,
        validate_prompt="原文：{chunk_text}\n三元组：{triples_text}",
        model_name="test-model",
        confidence_threshold=threshold,
        enabled=True,
    )


class TestValidateGating:
    def test_disabled_passthrough(self):
        llm = _FakeLLM('{"valid": []}')
        v = Validator(validate_llm=llm, validate_prompt="{chunk_text}{triples_text}",
                      enabled=False)
        rels = [_rel("丁元英", "芮小丹", 0.5)]
        assert v.validate(rels, "原文") == rels
        assert llm.calls == 0

    def test_high_confidence_skips_llm(self):
        llm = _FakeLLM('{"valid": []}')
        v = _validator(llm, threshold=0.7)
        rels = [_rel("丁元英", "芮小丹", 0.9)]
        assert v.validate(rels, "原文") == rels
        assert llm.calls == 0

    def test_single_low_conf_relation_is_validated(self):
        # 缺陷 #6 回归：单条低置信度关系也必须送 LLM 校验（此前直接放行）
        llm = _FakeLLM('{"valid": [0]}')
        v = _validator(llm)
        out = v.validate([_rel("丁元英", "芮小丹", 0.5)], "原文")
        assert llm.calls == 1
        assert len(out) == 1
        assert out[0].validated is True
        assert out[0].validate_model == "test-model"


class TestIndexCoercion:
    def test_string_indices_accepted(self):
        # 缺陷 #6 回归：LLM 返回 ["0","2"] 字符串下标时不能整体误删
        llm = _FakeLLM('{"valid": ["0", "2"], "invalid": ["1"]}')
        v = _validator(llm)
        rels = [_rel("甲某", "乙某", 0.5), _rel("丙某", "丁某", 0.5), _rel("戊某", "己某", 0.5)]
        out = v.validate(rels, "原文")
        assert {(r.subject, r.object) for r in out} == {("甲某", "乙某"), ("戊某", "己某")}

    def test_invalid_filtered(self):
        llm = _FakeLLM('{"valid": [1], "invalid": [0]}')
        v = _validator(llm)
        rels = [_rel("甲某", "乙某", 0.5), _rel("丙某", "丁某", 0.5)]
        out = v.validate(rels, "原文")
        assert len(out) == 1
        assert out[0].subject == "丙某"


class TestFallbacks:
    def test_llm_exception_keeps_all(self):
        v = _validator(_BrokenLLM())
        rels = [_rel("甲某", "乙某", 0.5), _rel("丙某", "丁某", 0.5)]
        out = v.validate(rels, "原文")
        assert len(out) == 2
        assert all(r.validated for r in out)

    def test_invalid_json_keeps_all(self):
        v = _validator(_FakeLLM("这不是 JSON 输出"))
        rels = [_rel("甲某", "乙某", 0.5)]
        assert len(v.validate(rels, "原文")) == 1

    def test_empty_judgment_keeps_all(self):
        # LLM 返回空判定时保留全部（防御性放行）
        v = _validator(_FakeLLM('{"valid": [], "invalid": [], "corrected": []}'))
        rels = [_rel("甲某", "乙某", 0.5), _rel("丙某", "丁某", 0.5)]
        assert len(v.validate(rels, "原文")) == 2


class TestCorrection:
    def test_corrected_relation_replaces_original(self):
        reply = ('{"valid": [], "invalid": [0], '
                 '"corrected": [{"index": 0, "subject": "丁元英", '
                 '"predicate": "帮助", "object": "王庙村"}]}')
        v = _validator(_FakeLLM(reply))
        orig = _rel("元英", "庙村", 0.5)
        out = v.validate([orig], "原文")
        assert len(out) == 1
        corrected = out[0]
        assert (corrected.subject, corrected.predicate, corrected.object) == ("丁元英", "帮助", "王庙村")
        assert corrected.validated is True
        assert corrected.confidence > orig.confidence

    def test_string_index_in_corrected(self):
        reply = '{"corrected": [{"index": "0", "subject": "修正后主语"}]}'
        v = _validator(_FakeLLM(reply))
        out = v.validate([_rel("甲某", "乙某", 0.5)], "原文")
        assert len(out) == 1
        assert out[0].subject == "修正后主语"

    def test_dedup_valid_and_corrected(self):
        # 同一条关系既在 valid 又被"修正"为相同三元组时不应重复
        reply = ('{"valid": [0], "corrected": [{"index": 0}]}')
        v = _validator(_FakeLLM(reply))
        out = v.validate([_rel("甲某", "乙某", 0.5)], "原文")
        assert len(out) == 1
