"""
管线日志工具 —— 用标准 logging + contextvars 取代层层传递的 log_list。

原有模式是把一个可变 list 穿透多层构造函数当作简易 logger，
存在线程安全问题，且与 Streamlit @st.cache_resource 交互时会导致
缓存引擎持有首次传入的旧 list（第二次查询起日志丢失 + 内存泄漏）。

新模式：
  - 组件内部直接 logger.info("步骤 3.1 …")（logger 名以 "rag" 开头即可被捕获）
  - 入口层用 capture_pipeline_logs() 圈住一段管线执行，
    期间 rag.* 下所有 INFO+ 日志会被收集到当前上下文的缓冲区：

      with capture_pipeline_logs() as cap:
          response = query_engine.query(question)
      for line in cap.drain():
          print(f"  {line}")

基于 contextvars，天然按调用上下文隔离，多会话（Streamlit）互不串扰。
注意：ThreadPoolExecutor 工作线程不继承上下文，工作线程内的日志
只会进入标准 logging，不进入捕获缓冲（管线各阶段的汇总日志均在主线程输出）。
查询期并行任务请使用 rag.utils.concurrency.run_parallel_captured ——
它在 worker 内独立捕获日志、由主线程按任务顺序回放（replay_into_capture），
保证 UI"运行流程"面板的日志有序且不丢失。
"""

import contextvars
import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager

_current_buffer: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "rag_pipeline_log_buffer", default=None
)

_install_lock = threading.Lock()
_handler_installed = False


class _ContextBufferHandler(logging.Handler):
    """把日志记录追加到当前上下文的缓冲区（若存在）。"""

    def emit(self, record: logging.LogRecord) -> None:
        buffer = _current_buffer.get()
        if buffer is not None:
            try:
                buffer.append(record.getMessage())
            except Exception:  # noqa: BLE001 —— 日志失败不能影响主流程
                pass


def _install_handler() -> None:
    """在 "rag" 顶层 logger 上安装捕获 handler（幂等）。"""
    global _handler_installed
    if _handler_installed:
        return
    with _install_lock:
        if _handler_installed:
            return
        rag_logger = logging.getLogger("rag")
        handler = _ContextBufferHandler(level=logging.INFO)
        rag_logger.addHandler(handler)
        # 确保 INFO 记录能到达 handler（不影响向 root 的传播）
        if rag_logger.level == logging.NOTSET or rag_logger.level > logging.INFO:
            rag_logger.setLevel(logging.INFO)
        _handler_installed = True


class LogCapture:
    """一次管线执行的日志缓冲。"""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def drain(self) -> list[str]:
        """取出并清空已捕获的日志行。"""
        lines = self.lines[:]
        self.lines.clear()
        return lines


def replay_into_capture(lines: list) -> None:
    """把 worker 线程内预先捕获的日志行追加到当前上下文的捕获缓冲。

    仅写入捕获缓冲、不经过 logging（worker 发射时已进入标准 logging，
    再走 logger 会在控制台重复打印一遍）。无捕获上下文时静默忽略。
    """
    buffer = _current_buffer.get()
    if buffer is not None and lines:
        buffer.extend(lines)


@contextmanager
def capture_pipeline_logs() -> Iterator[LogCapture]:
    """捕获 with 块内 rag.* 模块产生的 INFO+ 日志。"""
    _install_handler()
    capture = LogCapture()
    token = _current_buffer.set(capture.lines)
    try:
        yield capture
    finally:
        _current_buffer.reset(token)
