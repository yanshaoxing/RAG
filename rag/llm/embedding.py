"""带用量计量的 embedding 客户端。

llama_index 的 `OpenAILikeEmbedding` 内部走 `get_embeddings()`，该函数只把
`response.data` 里的向量取出来返回，**服务端回传的 `usage` 被直接丢弃**
（实测 HTTP 响应里确有 `usage.prompt_tokens`）。embedding 在查询期也有开销
（每条查询 2 次向量检索各一次嵌入），不拦截就会在成本核算里留一个盲区。

本模块只覆盖取向量的三个入口，逐一改为「自己发请求 → 记录 usage → 返回向量」，
并保留父类的重试装饰器（`_create_retry_decorator`），行为与原实现一致。
"""

import time

from llama_index.embeddings.openai_like import OpenAILikeEmbedding

from rag.metering import record_openai_usage


class MeteredOpenAILikeEmbedding(OpenAILikeEmbedding):
    """在 OpenAILikeEmbedding 之上记录服务端返回的 embedding usage。"""

    def _embed_and_record(self, texts: list[str]) -> list[list[float]]:
        """发一次 embeddings 请求，记录 usage 后返回向量列表。"""
        client = self._get_client()
        retry_decorator = self._create_retry_decorator()

        # 与父类 get_embeddings 保持一致：换行会影响部分模型的嵌入质量
        cleaned = [text.replace("\n", " ") for text in texts]

        @retry_decorator
        def _call():
            return client.embeddings.create(
                input=cleaned, model=self._text_engine, **self.additional_kwargs
            )

        start = time.perf_counter()
        response = _call()
        record_openai_usage(self.model_name, "embed", getattr(response, "usage", None),
                            elapsed=time.perf_counter() - start)
        return [d.embedding for d in response.data]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._embed_and_record(texts)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed_and_record([text])[0]

    def _get_query_embedding(self, query: str) -> list[float]:
        # 查询向量走同一模型同一端点（父类亦如此），一并计量
        return self._embed_and_record([query])[0]
