"""
查询期并行工具 —— 带日志捕获回放的并行任务执行。

查询管线的并行点（三路改写 / 子查询 / 图检索与主检索并发）都有同一个问题：
ThreadPoolExecutor 工作线程不继承 contextvars，worker 内的 rag.* 日志
进不了 capture_pipeline_logs() 的捕获缓冲，UI"运行流程"面板会缺失日志。

run_parallel_captured 的约定：
  - 每个 worker 内部用 capture_pipeline_logs() 独立捕获自己的日志
    （worker 线程 contextvars 为空，嵌套设置互不影响）；
  - 全部任务完成后，主线程按【任务提交顺序】把各 worker 的日志回放到
    当前捕获上下文（replay_into_capture）——UI 日志有序且分组连续；
  - 控制台输出不受影响：worker 发射日志时照常传播到 root handler（实时、
    可能交错），回放只写捕获缓冲、不重复打印；
  - 任一任务抛异常时：先回放全部日志，再抛出第一个异常（按任务顺序）。
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, TypeVar

from rag.logging_utils import capture_pipeline_logs, replay_into_capture

logger = logging.getLogger(__name__)

T = TypeVar("T")


def run_parallel_captured(
    tasks: list[Callable[[], T]],
    max_workers: Optional[int] = None,
) -> list[T]:
    """并行执行无参任务列表，按提交顺序返回结果列表。

    Args:
        tasks: 无参可调用列表（用闭包/lambda 绑定参数）
        max_workers: 最大并发数，默认等于任务数

    Returns:
        与 tasks 等长的结果列表，顺序与 tasks 一致
    """
    if not tasks:
        return []
    if len(tasks) == 1:
        # 单任务直接在当前线程执行，保持原有日志上下文
        return [tasks[0]()]

    def _run(fn: Callable[[], T]):
        with capture_pipeline_logs() as cap:
            try:
                return fn(), cap.drain(), None
            except Exception as e:  # noqa: BLE001 —— 延迟到主线程统一抛出
                return None, cap.drain(), e

    workers = min(max_workers or len(tasks), len(tasks))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run, fn) for fn in tasks]
        outcomes = [f.result() for f in futures]

    results: list[T] = []
    first_exc: Optional[Exception] = None
    for result, lines, exc in outcomes:
        replay_into_capture(lines)
        if exc is not None and first_exc is None:
            first_exc = exc
        results.append(result)

    if first_exc is not None:
        raise first_exc
    return results
