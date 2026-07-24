"""rag/metering.py 单测 —— 用量累计、费用估算、并发传播。

核心约定：token 一律取服务端 usage，未回传 usage 的调用记为「未计量」而非估算；
单价未知的模型不参与费用合计（宁可不报，也不报错数）。
"""

import pytest

from rag import config
from rag.metering import Meter, capture_usage, record_openai_usage, record_usage, step_timer
from rag.utils.concurrency import run_parallel_captured


class TestMeterAccumulation:
    def test_按模型与类型分桶(self):
        m = Meter()
        m.record("A", "chat", prompt_tokens=10, completion_tokens=5)
        m.record("A", "chat", prompt_tokens=20, completion_tokens=7)
        m.record("A", "embed", prompt_tokens=3)
        assert len(m.models) == 2
        chat = m.models["chat:A"]
        assert (chat.calls, chat.prompt_tokens, chat.completion_tokens) == (2, 30, 12)
        assert m.models["embed:A"].calls == 1

    def test_未回传usage记为未计量且不污染token(self):
        m = Meter()
        m.record("A", "chat", prompt_tokens=None, elapsed=1.0)
        u = m.models["chat:A"]
        assert u.calls == 1 and u.unmetered_calls == 1
        assert u.prompt_tokens == 0 and u.completion_tokens == 0
        assert u.elapsed == 1.0
        assert m.has_unmetered is True

    def test_思考token单列且计入输出(self):
        m = Meter()
        m.record("A", "chat", prompt_tokens=10, completion_tokens=5000,
                 reasoning_tokens=4987)
        u = m.models["chat:A"]
        # reasoning 计入 completion_tokens 一起按输出价计费，单列只为可见性
        assert u.completion_tokens == 5000 and u.reasoning_tokens == 4987


class TestCost:
    def test_按单价表估算(self, monkeypatch):
        monkeypatch.setattr(config, "MODEL_PRICES", {
            "M": {"input": 1.0, "output": 10.0, "cached_input": 0.1}})
        m = Meter()
        m.record("M", "chat", prompt_tokens=1_000_000, completion_tokens=1_000_000)
        assert m.models["chat:M"].cost == pytest.approx(11.0)

    def test_缓存命中按缓存价且不重复计费(self, monkeypatch):
        monkeypatch.setattr(config, "MODEL_PRICES", {
            "M": {"input": 1.0, "output": 0.0, "cached_input": 0.1}})
        m = Meter()
        # cached_tokens 是 prompt_tokens 的子集，非缓存部分按原价
        m.record("M", "chat", prompt_tokens=1_000_000, cached_tokens=400_000)
        assert m.models["chat:M"].cost == pytest.approx(0.6 + 0.04)

    def test_未知模型不静默算零(self, monkeypatch):
        monkeypatch.setattr(config, "MODEL_PRICES", {})
        m = Meter()
        m.record("未知模型", "chat", prompt_tokens=1000)
        assert m.models["chat:未知模型"].cost is None
        assert m.total_cost is None          # 合计也必须是 None，不能少报

    def test_缺单价时合计为None(self, monkeypatch):
        monkeypatch.setattr(config, "MODEL_PRICES", {"M": {"input": 1.0}})
        m = Meter()
        m.record("M", "chat", prompt_tokens=1000)
        m.record("X", "chat", prompt_tokens=1000)
        assert m.total_cost is None


class TestRecordOpenAIUsage:
    def test_解析dict形式的usage(self):
        with capture_usage() as m:
            record_openai_usage("M", "chat", {
                "prompt_tokens": 100, "completion_tokens": 50,
                "completion_tokens_details": {"reasoning_tokens": 40},
                "prompt_tokens_details": {"cached_tokens": 30},
            })
        u = m.models["chat:M"]
        assert (u.prompt_tokens, u.completion_tokens) == (100, 50)
        assert (u.reasoning_tokens, u.cached_tokens) == (40, 30)

    def test_解析对象形式的usage(self):
        class _Details:
            reasoning_tokens = 7

        class _Usage:
            prompt_tokens = 1
            completion_tokens = 2
            completion_tokens_details = _Details()
            prompt_tokens_details = None

        with capture_usage() as m:
            record_openai_usage("M", "chat", _Usage())
        u = m.models["chat:M"]
        assert (u.prompt_tokens, u.completion_tokens, u.reasoning_tokens) == (1, 2, 7)

    def test_usage缺失记为未计量(self):
        with capture_usage() as m:
            record_openai_usage("M", "chat", None)
        assert m.models["chat:M"].unmetered_calls == 1

    def test_字段缺失按零处理(self):
        with capture_usage() as m:
            record_openai_usage("M", "chat", {"prompt_tokens": 5})
        u = m.models["chat:M"]
        assert u.prompt_tokens == 5
        assert u.completion_tokens == 0 and u.reasoning_tokens == 0


class TestContext:
    def test_无计量上下文时静默忽略(self):
        record_usage("M", "chat", prompt_tokens=1)   # 不抛异常即可

    def test_开关关闭时不记录(self, monkeypatch):
        monkeypatch.setattr(config, "METERING_ENABLED", False)
        with capture_usage() as m:
            record_usage("M", "chat", prompt_tokens=1)
        assert m.models == {}

    def test_上下文退出后不再累计(self):
        with capture_usage() as m:
            record_usage("M", "chat", prompt_tokens=1)
        record_usage("M", "chat", prompt_tokens=999)
        assert m.models["chat:M"].prompt_tokens == 1

    def test_步骤计时(self):
        with capture_usage() as m:
            with step_timer("步骤 X"):
                pass
        assert len(m.steps) == 1 and m.steps[0][0] == "步骤 X"


class TestConcurrencyPropagation:
    def test_worker内的用量累加到同一个meter(self):
        """三路改写与子查询都在 ThreadPoolExecutor worker 里执行 ——
        contextvars 不传播的话，查询期绝大部分 token 都会被漏掉。"""
        with capture_usage() as m:
            run_parallel_captured(
                [lambda: record_usage("M", "chat", prompt_tokens=10),
                 lambda: record_usage("M", "chat", prompt_tokens=20),
                 lambda: record_usage("M", "chat", prompt_tokens=30)],
                max_workers=3,
            )
        u = m.models["chat:M"]
        assert u.calls == 3 and u.prompt_tokens == 60

    def test_worker内的日志顺序不受上下文传播影响(self):
        """回归：为传播 meter 引入 copy_context 后，日志仍须按任务提交顺序回放。"""
        import logging

        from rag.logging_utils import capture_pipeline_logs

        log = logging.getLogger("rag.test_metering")

        def _make(i):
            return lambda: log.info(f"任务{i}")

        with capture_pipeline_logs() as cap:
            run_parallel_captured([_make(1), _make(2), _make(3)], max_workers=3)
            lines = cap.drain()
        assert lines == ["任务1", "任务2", "任务3"]


class TestSummary:
    def test_无调用时汇总为空(self):
        assert Meter().summary_lines() == []

    def test_汇总包含思考占比与合计(self, monkeypatch):
        monkeypatch.setattr(config, "MODEL_PRICES", {"M": {"input": 1.0, "output": 1.0}})
        m = Meter()
        m.record("M", "chat", prompt_tokens=100, completion_tokens=1000,
                 reasoning_tokens=990)
        text = "\n".join(m.summary_lines())
        assert "思考 990" in text and "99%" in text
        assert "合计" in text

    def test_含未计量调用时给出提示(self):
        m = Meter()
        m.record("M", "chat", prompt_tokens=None)
        assert any("未计量" in line for line in m.summary_lines())
