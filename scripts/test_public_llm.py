# -*- coding: utf-8 -*-
"""公网 LLM（阿里云）连通性测试脚本。

测试 LLM.txt 中记录的 4 个端点（均用最短的测试数据）：
  1. 主模型         qwen3.6-flash        （OpenAI 兼容 chat，dashscope-us）
  2. 图谱校验模型   qwen3.5-flash        （OpenAI 兼容 chat，dashscope-us）
  3. 嵌入模型       qwen3.7-text-embedding（OpenAI 兼容 embeddings，ap-southeast-1 workspace）
  4. 重排模型       qwen3-rerank         （MaaS rerank 接口，cn-beijing workspace，直接 POST）

三个端点使用不同的 API Key，从环境变量（或项目根目录 .env）读取：
  RAG_PUBLIC_CHAT_API_KEY / RAG_PUBLIC_EMBED_API_KEY / RAG_PUBLIC_RERANK_API_KEY
不在代码中硬编码。用法：
    .venv/bin/python scripts/test_public_llm.py
"""

import os
import sys
import time

import requests
from openai import OpenAI

# ---------------------------------------------------------------- 配置
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# dashscope-us 有内容审核（小说文本被 400 拦截），chat 改用 ap-southeast-1
# workspace（与 embedding 同端点同 key，无审核）
CHAT_BASE_URL = "https://ws-hnkcnqxceqyt3qrt.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
CHAT_MAIN_MODEL = "qwen3.6-flash"
CHAT_VALIDATE_MODEL = "qwen3.5-flash"

EMBED_BASE_URL = "https://ws-hnkcnqxceqyt3qrt.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
# 注意：LLM.txt 写的 qwen3.7-text-embedding 在该 workspace 不存在（404），
# 实际可用的嵌入模型为 text-embedding-v4 / text-embedding-v3（v4 即 Qwen3 系）。
EMBED_MODEL = "text-embedding-v4"

RERANK_URL = "https://ws-prbh7fipy7z0uzpu.cn-beijing.maas.aliyuncs.com/compatible-api/v1/reranks"
RERANK_MODEL = "qwen3-rerank"

TIMEOUT = 60


def _load_env_file() -> dict[str, str]:
    """解析项目根目录 .env（KEY=VALUE 格式，# 开头为注释）。"""
    result: dict[str, str] = {}
    env_path = os.path.join(_ROOT, ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                result[k.strip()] = v.strip()
    return result


def _load_api_key(name: str, env_file: dict[str, str]) -> str:
    """优先读环境变量，其次读 .env；缺失则报错退出。"""
    key = os.environ.get(name, "").strip() or env_file.get(name, "")
    if not key:
        print(f"[错误] 未找到 API Key：请设置环境变量 {name} 或在 .env 中配置")
        sys.exit(1)
    return key


# ---------------------------------------------------------------- 各项测试
def test_chat(client: OpenAI, model: str) -> bool:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "回复“连通”两个字即可。"}],
        max_tokens=16,
    )
    text = (resp.choices[0].message.content or "").strip()
    print(f"    模型返回: {text[:50]!r}")
    return bool(text)


def test_embedding(api_key: str) -> bool:
    client = OpenAI(api_key=api_key, base_url=EMBED_BASE_URL, timeout=TIMEOUT)
    resp = client.embeddings.create(model=EMBED_MODEL, input="连通性测试")
    vec = resp.data[0].embedding
    print(f"    向量维度: {len(vec)}，前 3 维: {[round(v, 4) for v in vec[:3]]}")
    return len(vec) > 0


def test_rerank(api_key: str) -> bool:
    resp = requests.post(
        RERANK_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": RERANK_MODEL,
            "query": "丁元英是谁",
            "documents": ["丁元英是《遥远的救世主》的男主角。", "今天天气不错。"],
            "top_n": 2,
        },
        timeout=TIMEOUT,
    )
    if resp.status_code != 200:
        print(f"    HTTP {resp.status_code}: {resp.text[:200]}")
        return False
    data = resp.json()
    results = data.get("results") or data.get("output", {}).get("results") or []
    if not results:
        print(f"    响应中未找到 results 字段: {str(data)[:200]}")
        return False
    for r in results:
        idx = r.get("index")
        score = r.get("relevance_score", r.get("score"))
        print(f"    文档[{idx}] 得分: {score}")
    return True


# ---------------------------------------------------------------- 主流程
def main() -> None:
    env_file = _load_env_file()
    chat_key = _load_api_key("RAG_PUBLIC_CHAT_API_KEY", env_file)
    embed_key = _load_api_key("RAG_PUBLIC_EMBED_API_KEY", env_file)
    rerank_key = _load_api_key("RAG_PUBLIC_RERANK_API_KEY", env_file)
    chat_client = OpenAI(api_key=chat_key, base_url=CHAT_BASE_URL, timeout=TIMEOUT)

    tests = [
        (f"主模型 chat（{CHAT_MAIN_MODEL}）", lambda: test_chat(chat_client, CHAT_MAIN_MODEL)),
        (f"图谱校验 chat（{CHAT_VALIDATE_MODEL}）", lambda: test_chat(chat_client, CHAT_VALIDATE_MODEL)),
        (f"嵌入（{EMBED_MODEL}）", lambda: test_embedding(embed_key)),
        (f"重排（{RERANK_MODEL}）", lambda: test_rerank(rerank_key)),
    ]

    results = []
    for name, fn in tests:
        print(f"\n=== 测试 {name} ===")
        start = time.time()
        try:
            ok = fn()
        except Exception as e:
            print(f"    异常: {e.__class__.__name__}: {e}")
            ok = False
        elapsed = time.time() - start
        print(f"    结果: {'✅ 连通' if ok else '❌ 失败'}（耗时 {elapsed:.1f}s）")
        results.append((name, ok))

    print("\n" + "=" * 40)
    print("汇总:")
    for name, ok in results:
        print(f"  {'✅' if ok else '❌'} {name}")
    if not all(ok for _, ok in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
