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


# ======================== DavyLLM ========================

class DavyLLM(CustomLLM):
    """Davy 云端大模型适配器，实现 llama_index CustomLLM 接口。"""

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
                    verify=self.cert_path,
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
        body = self._build_request_body(messages, **kwargs)
        body["stream"] = True

        def gen() -> ChatResponseGen:
            full_content = ""
            response = self._post_with_retry(body, stream=True)
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
                    if delta:
                        full_content += delta
                except json.JSONDecodeError:
                    continue
            clean = DavyLLM._strip_thinking(full_content)
            yield ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content=clean),
                delta=clean,
            )

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
                    text=chat_resp.delta or "",
                    delta=chat_resp.delta,
                    raw=chat_resp.raw,
                )

        return gen()


# ======================== 工厂函数 ========================

def create_answer_llm() -> CustomLLM:
    """创建最终回答用的 LLM。"""
    if config.ANSWER_PROVIDER == "davy":
        return DavyLLM()
    return Ollama(
        model=config.ANSWER_OLLAMA_MODEL,
        request_timeout=config.ANSWER_OLLAMA_TIMEOUT,
        temperature=config.ANSWER_OLLAMA_TEMPERATURE,
    )


def create_rewrite_llm() -> CustomLLM:
    """创建查询重写用的 LLM。"""
    if config.REWRITE_PROVIDER == "davy":
        return DavyLLM()
    return Ollama(
        model=config.REWRITE_OLLAMA_MODEL,
        request_timeout=config.REWRITE_OLLAMA_TIMEOUT,
        temperature=config.REWRITE_OLLAMA_TEMPERATURE,
    )


def create_summary_llm() -> CustomLLM:
    """创建摘要生成用的 LLM。"""
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