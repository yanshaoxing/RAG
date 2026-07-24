"""用量计量 —— token 与耗时的按次累计，用于成本核算与性能定位。

**不做本地分词估算**：token 数一律取服务端在响应体里回传的 `usage` 字段
（OpenAI 兼容协议），那就是计费用的同一份数字。各链路实测均可拿到：

  - chat 非流式：`usage`（含 completion_tokens_details.reasoning_tokens、
    prompt_tokens_details.cached_tokens）
  - chat 流式：DashScope 默认就在末尾 chunk 回传 usage（choices 为空数组）
  - embedding：HTTP 响应有 usage，但 llama_index 的 OpenAILikeEmbedding 只返回
    向量、把 usage 丢了 —— 故用 rag/llm/embedding.py 的子类拦截
  - rerank：`usage.total_tokens`

**reasoning token 必须单列**：它计入 completion_tokens 按输出价计费，却不进
message.content（<think> 剥离器看不到），不单列则账单与可见输出永远对不上。

无 usage 的链路（Ollama 本地模型）只记调用次数与耗时，token 记为 None 并在
汇总里标注「未计量」—— 绝不用估算值冒充实测值。

用法与 capture_pipeline_logs 一致（同为 contextvars 上下文）：

    with capture_usage() as meter:
        response = query_engine.query(question)
    for line in meter.summary_lines():
        print(line)

并发：Meter 内部加锁，`run_parallel_captured` 通过 contextvars.copy_context()
让 worker 共享同一个 Meter 实例（日志是各自捕获后回放，用量则直接累加）。
"""

import contextvars
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional

from rag import config

_current_meter: contextvars.ContextVar[Optional["Meter"]] = contextvars.ContextVar(
    "rag_usage_meter", default=None
)


@dataclass
class ModelUsage:
    """按模型累计的用量。token 为 None 表示该链路不回传 usage（未计量）。"""

    model: str
    kind: str                      # "chat" | "embed" | "rerank"
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0      # 计入 completion_tokens，单列用于暴露思考开销
    cached_tokens: int = 0         # 计入 prompt_tokens，按缓存命中价计费
    elapsed: float = 0.0
    unmetered_calls: int = 0       # 未回传 usage 的调用次数

    @property
    def cost(self) -> Optional[float]:
        """按 config.MODEL_PRICES 估算费用（元）。未知模型返回 None，不静默算 0。"""
        price = config.MODEL_PRICES.get(self.model)
        if price is None:
            return None
        billed_input = max(self.prompt_tokens - self.cached_tokens, 0)
        cost = billed_input / 1e6 * price.get("input", 0.0)
        cost += self.cached_tokens / 1e6 * price.get("cached_input", price.get("input", 0.0))
        cost += self.completion_tokens / 1e6 * price.get("output", 0.0)
        return cost


@dataclass
class Meter:
    """一次执行（查询 / 构建阶段）的用量与耗时累计。线程安全。"""

    models: dict[str, ModelUsage] = field(default_factory=dict)
    steps: list[tuple[str, float]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ---------- 记录 ----------

    def _bucket(self, model: str, kind: str) -> ModelUsage:
        key = f"{kind}:{model}"
        usage = self.models.get(key)
        if usage is None:
            usage = ModelUsage(model=model, kind=kind)
            self.models[key] = usage
        return usage

    def record(
        self,
        model: str,
        kind: str,
        prompt_tokens: Optional[int] = None,
        completion_tokens: int = 0,
        reasoning_tokens: int = 0,
        cached_tokens: int = 0,
        elapsed: float = 0.0,
    ) -> None:
        """记录一次调用。prompt_tokens 为 None 表示该次调用未回传 usage。"""
        with self._lock:
            u = self._bucket(model, kind)
            u.calls += 1
            u.elapsed += elapsed
            if prompt_tokens is None:
                u.unmetered_calls += 1
                return
            u.prompt_tokens += prompt_tokens
            u.completion_tokens += completion_tokens
            u.reasoning_tokens += reasoning_tokens
            u.cached_tokens += cached_tokens

    def record_step(self, name: str, elapsed: float) -> None:
        with self._lock:
            self.steps.append((name, elapsed))

    # ---------- 汇总 ----------

    @property
    def total_cost(self) -> Optional[float]:
        """总费用（元）。任一模型缺单价则返回 None（宁可不报，也不报错数）。"""
        costs = [u.cost for u in self.models.values()]
        if not costs or any(c is None for c in costs):
            return None
        return sum(costs)

    @property
    def has_unmetered(self) -> bool:
        return any(u.unmetered_calls for u in self.models.values())

    def summary_lines(self) -> list[str]:
        """人类可读的用量汇总（无任何调用时返回空列表）。"""
        if not self.models:
            return []

        lines = ["用量汇总（token 数取自服务端 usage，非本地估算）："]
        for u in sorted(self.models.values(), key=lambda x: -(x.cost or 0)):
            cost = u.cost
            cost_str = f"约 ¥{cost:.4f}" if cost is not None else "单价未知"
            if u.kind == "chat":
                detail = f"输入 {u.prompt_tokens}，输出 {u.completion_tokens}"
                if u.reasoning_tokens:
                    ratio = u.reasoning_tokens / max(u.completion_tokens, 1) * 100
                    detail += f"（其中思考 {u.reasoning_tokens}，占 {ratio:.0f}%）"
                if u.cached_tokens:
                    detail += f"，缓存命中 {u.cached_tokens}"
            else:
                detail = f"token {u.prompt_tokens}"
            note = f"，{u.unmetered_calls} 次未计量" if u.unmetered_calls else ""
            lines.append(
                f"  {u.model}（{u.kind}）：{u.calls} 次调用，{detail}，"
                f"耗时 {u.elapsed:.1f}s，{cost_str}{note}"
            )

        total = self.total_cost
        if total is not None:
            lines.append(f"  合计：约 ¥{total:.4f}")
        if self.has_unmetered:
            lines.append("  ⚠️ 含未回传 usage 的调用（如 Ollama），其 token 未计入")
        return lines

    def step_lines(self) -> list[str]:
        """各步骤耗时（按记录顺序）。"""
        if not self.steps:
            return []
        lines = ["耗时分解："]
        for name, elapsed in self.steps:
            lines.append(f"  {name}: {elapsed:.2f}s")
        return lines


# ---------- 上下文 API ----------

def current_meter() -> Optional[Meter]:
    """当前上下文的 Meter（未开启计量时为 None）。"""
    return _current_meter.get()


@contextmanager
def capture_usage() -> Iterator[Meter]:
    """开启一段执行的用量计量。"""
    meter = Meter()
    token = _current_meter.set(meter)
    try:
        yield meter
    finally:
        _current_meter.reset(token)


def record_usage(
    model: str,
    kind: str,
    prompt_tokens: Optional[int] = None,
    completion_tokens: int = 0,
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
    elapsed: float = 0.0,
) -> None:
    """记录一次调用用量（无计量上下文时静默忽略）。"""
    if not config.METERING_ENABLED:
        return
    meter = _current_meter.get()
    if meter is None:
        return
    meter.record(model, kind, prompt_tokens, completion_tokens,
                 reasoning_tokens, cached_tokens, elapsed)


def record_openai_usage(model: str, kind: str, usage, elapsed: float = 0.0) -> None:
    """从 OpenAI 兼容响应的 usage 字段（dict 或对象）提取并记录。

    字段缺失一律按 0 处理；usage 整体缺失则记为未计量。
    """
    if usage is None:
        record_usage(model, kind, prompt_tokens=None, elapsed=elapsed)
        return

    def _get(obj, key, default=0):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default) or default
        return getattr(obj, key, default) or default

    completion_details = _get(usage, "completion_tokens_details", None)
    prompt_details = _get(usage, "prompt_tokens_details", None)
    record_usage(
        model=model,
        kind=kind,
        prompt_tokens=_get(usage, "prompt_tokens"),
        completion_tokens=_get(usage, "completion_tokens"),
        reasoning_tokens=_get(completion_details, "reasoning_tokens"),
        cached_tokens=_get(prompt_details, "cached_tokens"),
        elapsed=elapsed,
    )


@contextmanager
def step_timer(name: str) -> Iterator[None]:
    """记录一个步骤的耗时（无计量上下文时静默忽略）。"""
    start = time.perf_counter()
    try:
        yield
    finally:
        meter = _current_meter.get()
        if meter is not None and config.METERING_ENABLED:
            meter.record_step(name, time.perf_counter() - start)
