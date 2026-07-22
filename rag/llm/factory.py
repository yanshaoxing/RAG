"""
LLM 工厂模块 —— 根据 provider 选择 Ollama 本地模型或 Davy 云端模型。

统一实现 llama_index CustomLLM 接口。
"""

import json
import logging
import re
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
    def _strip_thinking(text: str) -> str:
        """移除 <thinking>...</thinking> 思考块。"""
        return re.sub(r"<thinking>.*?</thinking>\s*", "", text, flags=re.DOTALL).strip()

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

    def _call(self, request_body: dict) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        response = requests.post(
            url=f"{self.base_url}/chat/completions",
            headers=headers,
            json=request_body,
            verify=self.cert_path,
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        return response.json()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        body = self._build_request_body(messages, **kwargs)
        resp = self._call(body)
        content = self._strip_thinking(resp["choices"][0]["message"]["content"])
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
            raw=resp,
        )

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        body = self._build_request_body(messages, **kwargs)
        body["stream"] = True

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        def gen() -> ChatResponseGen:
            full_content = ""
            response = requests.post(
                url=f"{self.base_url}/chat/completions",
                headers=headers,
                json=body,
                verify=self.cert_path,
                timeout=self.request_timeout,
                stream=True,
            )
            response.raise_for_status()
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
        model=config.REWRITE_OLLAMA_MODEL,
        request_timeout=config.ANSWER_OLLAMA_TIMEOUT,
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