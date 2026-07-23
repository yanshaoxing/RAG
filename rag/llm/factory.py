"""
LLM 工厂模块 —— 根据 provider 选择 Ollama 本地模型或 Davy 云端模型。

统一实现 llama_index CustomLLM 接口。
"""

import json
import logging
import re
import time
from typing import Any, Optional, Sequence

import requests
from llama_index.core.llms import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.llms.ollama import Ollama

from rag import config

logger = logging.getLogger(__name__)


# ======================== 流式思考块过滤 ========================

class ThinkStreamFilter:
    """流式增量剥离 <think>/<thinking> 思考块（标签可跨 chunk 边界）。

    与批量版 DavyLLM._strip_thinking 语义一致：丢弃闭合思考块及其后随空白；
    可能构成标签前缀的尾部字符会被暂存，直到能判定是否为真实标签。
    流结束时调用 finalize() 取回暂存的非标签残余。
    """

    _OPEN_TAGS = ("<thinking>", "<think>")
    _CLOSE_TAGS = ("</thinking>", "</think>")

    def __init__(self):
        self._buf = ""
        self._in_think = False
        self._skip_ws = True  # 起始/思考块结束后跳过空白（对齐批量版 strip 行为）

    @staticmethod
    def _partial_suffix_len(buf: str, tags: tuple) -> int:
        """buf 尾部与任一标签前缀重叠的最大长度（这些字符需暂存，不能输出/丢弃）。"""
        max_len = 0
        for tag in tags:
            for k in range(min(len(buf), len(tag) - 1), 0, -1):
                if buf.endswith(tag[:k]):
                    max_len = max(max_len, k)
                    break
        return max_len

    @staticmethod
    def _find_first(buf: str, tags: tuple):
        """返回 buf 中最先出现的标签 (位置, 标签)，未找到返回 (-1, None)。"""
        best_idx, best_tag = -1, None
        for tag in tags:
            i = buf.find(tag)
            if i != -1 and (best_idx == -1 or i < best_idx):
                best_idx, best_tag = i, tag
        return best_idx, best_tag

    def feed(self, delta: str) -> str:
        """喂入一个增量，返回可安全输出的正文文本（可能为空串）。"""
        self._buf += delta
        out: list[str] = []

        while True:
            if self._in_think:
                idx, tag = self._find_first(self._buf, self._CLOSE_TAGS)
                if idx == -1:
                    # 未闭合：丢弃已确定的思考内容，仅暂存可能的闭合标签前缀
                    keep = self._partial_suffix_len(self._buf, self._CLOSE_TAGS)
                    self._buf = self._buf[len(self._buf) - keep:] if keep else ""
                    break
                self._buf = self._buf[idx + len(tag):]
                self._in_think = False
                self._skip_ws = True  # 思考块后随空白一并剥离
            else:
                if self._skip_ws:
                    stripped = self._buf.lstrip()
                    if not stripped and self._buf:
                        self._buf = ""
                        break
                    if stripped:
                        self._buf = stripped
                        self._skip_ws = False
                idx, tag = self._find_first(self._buf, self._OPEN_TAGS)
                if idx == -1:
                    # 无开始标签：输出除"可能是标签前缀的尾部"之外的内容
                    keep = self._partial_suffix_len(self._buf, self._OPEN_TAGS)
                    emit_end = len(self._buf) - keep
                    if emit_end > 0:
                        out.append(self._buf[:emit_end])
                        self._buf = self._buf[emit_end:]
                    break
                out.append(self._buf[:idx])
                self._buf = self._buf[idx + len(tag):]
                self._in_think = True

        return "".join(out)

    def finalize(self) -> str:
        """流结束：暂存的标签前缀若未成为完整标签，按正文返回；未闭合思考内容丢弃。"""
        if self._in_think:
            self._buf = ""
            return ""
        tail = self._buf
        self._buf = ""
        return tail


# ======================== DavyLLM ========================

class DavyLLM(CustomLLM):
    """OpenAI 兼容云端大模型适配器（Davy / 阿里云通用），实现 llama_index CustomLLM 接口。

    cert_path 为自定义 CA 证书路径（Davy 内网证书）；传空串 "" 表示公网端点，
    使用系统 CA 验证（requests verify=True）。
    """

    model_name: str = config.DAVY_MODEL_NAME
    base_url: str = config.DAVY_BASE_URL
    cert_path: str = config.DAVY_CERT_PATH
    api_key: str = config.DAVY_API_KEY
    temperature: float = config.DAVY_TEMPERATURE
    request_timeout: float = config.DAVY_TIMEOUT

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        cert_path: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        request_timeout: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if model_name is not None:
            self.model_name = model_name
        if base_url is not None:
            self.base_url = base_url
        if cert_path is not None:
            self.cert_path = cert_path
        if api_key is not None:
            self.api_key = api_key
        if temperature is not None:
            self.temperature = temperature
        if request_timeout is not None:
            self.request_timeout = request_timeout

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model_name, is_chat_model=True)

    @staticmethod
    def _strip_thinking(text: Optional[str]) -> str:
        """移除 <thinking>/<think> 思考块（Qwen/DeepSeek 惯用 <think>）。content 为 None 时返回空串。"""
        if not text:
            return ""
        return re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>\s*", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _extract_content(resp: Any) -> str:
        """从响应中防御性提取 content，payload 异常时抛出带上下文的 ValueError 而非裸 KeyError。"""
        try:
            return resp["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise ValueError(f"Davy 响应格式异常，无法提取 content: {str(resp)[:300]}")

    def _build_request_body(self, messages: Sequence[ChatMessage], **kwargs: Any) -> dict:
        msgs = []
        for msg in messages:
            role_map = {
                MessageRole.SYSTEM: "system",
                MessageRole.USER: "user",
                MessageRole.ASSISTANT: "assistant",
                MessageRole.TOOL: "tool",
                MessageRole.FUNCTION: "function",
            }
            role = role_map.get(msg.role, "user")
            msgs.append({"role": role, "content": msg.content})

        temperature = kwargs.get("temperature", self.temperature)
        return {"model": self.model_name, "messages": msgs, "stream": False, "temperature": temperature}

    # 可重试的 HTTP 状态码：限流 + 服务端瞬时错误
    _RETRYABLE_STATUS = (429, 500, 502, 503, 504)

    @staticmethod
    def _retry_delay(response: Optional[requests.Response], attempt: int) -> float:
        """计算重试等待时间：优先尊重 Retry-After 头，否则指数退避。"""
        if response is not None:
            retry_after = response.headers.get("Retry-After", "")
            try:
                return max(float(retry_after), 0.5)
            except ValueError:
                pass
        return config.DAVY_RETRY_BASE_DELAY * (2 ** attempt)

    def _post_with_retry(self, request_body: dict, stream: bool = False) -> requests.Response:
        """带重试的 POST：429/5xx/超时/连接错误时指数退避重试（config.DAVY_MAX_RETRIES 次）。"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        max_retries = config.DAVY_MAX_RETRIES
        response: Optional[requests.Response] = None

        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    url=f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=request_body,
                    verify=self.cert_path or True,  # 空串 = 公网端点，走系统 CA
                    timeout=self.request_timeout,
                    stream=stream,
                )
            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt >= max_retries:
                    raise
                delay = self._retry_delay(None, attempt)
                logger.warning(f"Davy 请求异常（{e.__class__.__name__}），{delay:.1f}s 后重试 "
                               f"({attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue

            if response.status_code in self._RETRYABLE_STATUS and attempt < max_retries:
                delay = self._retry_delay(response, attempt)
                logger.warning(f"Davy HTTP {response.status_code}，{delay:.1f}s 后重试 "
                               f"({attempt + 1}/{max_retries})")
                response.close()
                time.sleep(delay)
                continue

            response.raise_for_status()
            return response

        # 理论上不可达（循环内必定 return 或 raise），防御性兜底
        raise RuntimeError("Davy 请求重试逻辑异常退出")

    def _call(self, request_body: dict) -> dict:
        return self._post_with_retry(request_body).json()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        body = self._build_request_body(messages, **kwargs)
        resp = self._call(body)
        content = self._strip_thinking(self._extract_content(resp))
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
            raw=resp,
        )

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        """真流式：SSE 增量逐块 yield（思考块经 ThinkStreamFilter 增量剥离）。

        断流重试：_post_with_retry 只覆盖建连/首包阶段，SSE 流中途断开时若
        **尚未输出任何正文**（常见于连接建立后立即被掐断），整流重试一次；
        已有部分输出则无法安全重试（重发的回答不保证与已输出前缀一致），照原样抛出。
        """
        body = self._build_request_body(messages, **kwargs)
        body["stream"] = True

        def gen() -> ChatResponseGen:
            max_stream_attempts = 2
            for stream_attempt in range(max_stream_attempts):
                think_filter = ThinkStreamFilter()
                full_content = ""
                emitted = False
                response = self._post_with_retry(body, stream=True)
                try:
                    for line in response.iter_lines():
                        if not line:
                            continue
                        line = line.decode("utf-8").strip()
                        if line == "data: [DONE]":
                            break
                        if line.startswith("data: "):
                            line = line[6:]
                        try:
                            chunk = json.loads(line)
                            delta = chunk["choices"][0]["delta"].get("content", "")
                        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                            continue
                        if not delta:
                            continue
                        clean = think_filter.feed(delta)
                        if clean:
                            full_content += clean
                            emitted = True
                            yield ChatResponse(
                                message=ChatMessage(role=MessageRole.ASSISTANT, content=full_content),
                                delta=clean,
                            )
                    tail = think_filter.finalize()
                    if tail:
                        full_content += tail
                    if tail or not full_content:
                        # 输出残余；或全程无正文（如仅思考内容）时兜底 yield 一次空响应
                        yield ChatResponse(
                            message=ChatMessage(role=MessageRole.ASSISTANT, content=full_content),
                            delta=tail,
                        )
                    return
                except Exception as e:
                    if emitted or stream_attempt >= max_stream_attempts - 1:
                        raise
                    logger.warning(f"Davy 流式响应中断（{e.__class__.__name__}），"
                                   f"尚未输出正文，整流重试一次")
                finally:
                    response.close()

        return gen()

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        chat_resp = self.chat(messages, **kwargs)
        return CompletionResponse(text=chat_resp.message.content, raw=chat_resp.raw)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]

        def gen() -> CompletionResponseGen:
            for chat_resp in self.stream_chat(messages, **kwargs):
                yield CompletionResponse(
                    text=chat_resp.message.content or "",  # 累积文本（llama_index 约定）
                    delta=chat_resp.delta,
                    raw=chat_resp.raw,
                )

        return gen()


# ======================== 工厂函数 ========================

def _create_aliyun_llm(model_name: Optional[str] = None,
                       temperature: Optional[float] = None) -> DavyLLM:
    """创建阿里云（公网 OpenAI 兼容端点）LLM，复用 DavyLLM 客户端。"""
    return DavyLLM(
        model_name=model_name or config.ALIYUN_MAIN_MODEL,
        base_url=config.ALIYUN_CHAT_BASE_URL,
        api_key=config.ALIYUN_CHAT_API_KEY,
        cert_path="",  # 公网端点，系统 CA
        temperature=temperature,
        request_timeout=config.ALIYUN_CHAT_TIMEOUT,
    )


def create_answer_llm() -> CustomLLM:
    """创建最终回答用的 LLM。"""
    if config.ANSWER_PROVIDER == "aliyun":
        return _create_aliyun_llm()
    if config.ANSWER_PROVIDER == "davy":
        return DavyLLM()
    return Ollama(
        model=config.ANSWER_OLLAMA_MODEL,
        request_timeout=config.ANSWER_OLLAMA_TIMEOUT,
        temperature=config.ANSWER_OLLAMA_TEMPERATURE,
    )


def create_rewrite_llm() -> CustomLLM:
    """创建查询重写用的 LLM。"""
    if config.REWRITE_PROVIDER == "aliyun":
        return _create_aliyun_llm(temperature=config.REWRITE_TEMPERATURE)
    if config.REWRITE_PROVIDER == "davy":
        return DavyLLM()
    return Ollama(
        model=config.REWRITE_OLLAMA_MODEL,
        request_timeout=config.REWRITE_OLLAMA_TIMEOUT,
        temperature=config.REWRITE_TEMPERATURE,
    )


def create_summary_llm() -> CustomLLM:
    """创建摘要生成用的 LLM。"""
    if config.SUMMARY_LLM_PROVIDER == "aliyun":
        return _create_aliyun_llm(temperature=config.SUMMARY_LLM_TEMPERATURE)
    if config.SUMMARY_LLM_PROVIDER == "davy":
        return DavyLLM(temperature=config.SUMMARY_LLM_TEMPERATURE)
    return Ollama(
        model=config.SUMMARY_OLLAMA_MODEL,
        request_timeout=config.SUMMARY_OLLAMA_TIMEOUT,
        temperature=config.SUMMARY_LLM_TEMPERATURE,
    )


def create_validate_llm() -> Optional[CustomLLM]:
    """创建三元组校验用的 LLM（不同模型交叉校验更可靠）。"""
    if not config.GRAPH_VALIDATE_ENABLED:
        return None

    if config.GRAPH_VALIDATE_LLM_PROVIDER == "aliyun":
        return _create_aliyun_llm(
            model_name=config.ALIYUN_VALIDATE_MODEL,
            temperature=config.GRAPH_VALIDATE_LLM_TEMPERATURE,
        )

    if config.GRAPH_VALIDATE_LLM_PROVIDER == "davy":
        return DavyLLM(
            model_name=config.GRAPH_VALIDATE_DAVY_MODEL,
            temperature=config.GRAPH_VALIDATE_LLM_TEMPERATURE,
        )

    return Ollama(
        model=config.GRAPH_VALIDATE_LLM_MODEL,
        base_url=config.GRAPH_VALIDATE_LLM_BASE_URL,
        request_timeout=config.GRAPH_VALIDATE_LLM_TIMEOUT,
        temperature=config.GRAPH_VALIDATE_LLM_TEMPERATURE,
    )