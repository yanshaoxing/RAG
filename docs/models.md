# 模型配置

> 返回 [README](../README.md) · 相关文档：[架构](architecture.md) · [部署与使用](operations.md) · [模型配置](models.md)

当前默认使用**公网阿里云**端点（`provider="aliyun"`），按角色分配如下：

| 角色 | 模型 | 端点 | API Key 环境变量 |
|---|---|---|---|
| 主模型（回答 / 改写 / 摘要 / 图谱抽取） | `qwen3.5-flash` | `ws-…cn-beijing.maas.aliyuncs.com/compatible-mode/v1` | `RAG_PUBLIC_CHAT_API_KEY` |
| 图谱三元组校验（异模型交叉校验） | `qwen-flash` | 同上 | `RAG_PUBLIC_CHAT_API_KEY` |
| 嵌入（**1024 维**，FAISS 维度随之切换） | `qwen3.7-text-embedding` | 同上 | `RAG_PUBLIC_EMBED_API_KEY` |
| 重排序 | `qwen3-rerank` | `ws-…cn-beijing.maas.aliyuncs.com/compatible-api/v1/reranks` | `RAG_PUBLIC_RERANK_API_KEY` |

- 公网 key **不入 git**：写在项目根目录 `.env`（已 gitignore）或直接设环境变量；`rag/config.py` 启动时自动加载 `.env`。
- **chat 不能走 `dashscope-us` 公共端点**：该端点带内容审核（`data_inspection_failed`），小说文本会被 400 拦截；当前 cn-beijing workspace 专属部署无此审核（已实测小说正文），chat / embedding / rerank 共用同一把 workspace key。
- **`ALIYUN_ENABLE_THINKING` 默认 `False`**：Qwen3 系默认开启思考，实测同一条摘要请求「思考开 = 5019 输出 token（99% 是 reasoning）/ 45.7 秒」vs「思考关 = 24 token / 0.7 秒」，概括质量无可见差异。reasoning 不进 `message.content`，`<think>` 剥离看不到它但照样计费——构建期数百上千次调用必须关闭。
- 每个角色的 provider 可独立切回内网：`ANSWER_PROVIDER` / `REWRITE_PROVIDER` / `SUMMARY_LLM_PROVIDER` / `GRAPH_VALIDATE_LLM_PROVIDER` / `EMBED_PROVIDER` / `RERANK_PROVIDER`（`"ollama"` / `"davy"` / `"vllm"` 为原内网配置，参数仍保留）。
- **换嵌入模型必须重建向量索引**：公网 `qwen3.7-text-embedding` 为 1024 维，内网 `qwen3-embedding:8b` 为 4096 维。⚠️ 维度相同也不代表可复用——不同模型的向量空间不同，旧索引配新查询向量的相似度毫无意义。向量阶段的完成标记会记录 `embed_model`，加载时比对不一致直接报错，删除语料的 `data/vector/` 重建即可。
- 连通性自检：`.venv/bin/python scripts/test_public_llm.py`（四个端点各发一条最小请求）。
