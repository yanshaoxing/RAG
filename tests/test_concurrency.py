"""rag/utils/concurrency.py 单测 —— 并行执行的结果顺序、日志回放顺序与异常传播。"""

import logging
import time

import pytest

from rag.logging_utils import capture_pipeline_logs, replay_into_capture
from rag.utils.concurrency import run_parallel_captured

_logger = logging.getLogger("rag.tests.concurrency")


class TestRunParallelCaptured:
    def test_empty_tasks(self):
        assert run_parallel_captured([]) == []

    def test_results_in_submission_order(self):
        def make(v, delay):
            def task():
                time.sleep(delay)
                return v
            return task

        # 第一个任务最慢，仍应排在结果首位
        out = run_parallel_captured([make("a", 0.05), make("b", 0.0), make("c", 0.0)])
        assert out == ["a", "b", "c"]

    def test_single_task_runs_inline(self):
        # 单任务不进线程池，日志直接进入当前捕获上下文
        def task():
            _logger.info("单任务日志")
            return 1

        with capture_pipeline_logs() as cap:
            assert run_parallel_captured([task]) == [1]
            lines = cap.drain()
        assert lines == ["单任务日志"]

    def test_worker_logs_replayed_grouped_in_task_order(self):
        def slow_task():
            _logger.info("慢任务-1")
            time.sleep(0.05)
            _logger.info("慢任务-2")
            return "slow"

        def fast_task():
            _logger.info("快任务-1")
            return "fast"

        with capture_pipeline_logs() as cap:
            out = run_parallel_captured([slow_task, fast_task], max_workers=2)
            lines = cap.drain()

        assert out == ["slow", "fast"]
        # 快任务先完成，但日志按任务提交顺序分组回放
        assert lines == ["慢任务-1", "慢任务-2", "快任务-1"]

    def test_exception_propagated_after_replay(self):
        def ok_task():
            _logger.info("正常任务日志")
            return "ok"

        def bad_task():
            _logger.info("失败任务日志")
            raise ValueError("boom")

        with capture_pipeline_logs() as cap:
            with pytest.raises(ValueError, match="boom"):
                run_parallel_captured([ok_task, bad_task], max_workers=2)
            lines = cap.drain()
        # 异常前先回放全部日志
        assert lines == ["正常任务日志", "失败任务日志"]

    def test_max_workers_respected(self):
        import threading

        active = 0
        peak = 0
        lock = threading.Lock()

        def task():
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return True

        run_parallel_captured([task] * 6, max_workers=2)
        assert peak <= 2


class TestReplayIntoCapture:
    def test_replay_appends_to_current_buffer(self):
        with capture_pipeline_logs() as cap:
            replay_into_capture(["行一", "行二"])
            assert cap.drain() == ["行一", "行二"]

    def test_no_capture_context_is_noop(self):
        replay_into_capture(["无上下文时静默忽略"])  # 不应抛异常
