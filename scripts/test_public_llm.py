"""公网 LLM（阿里云）连通性测试脚本 —— 与项目根目录 LLM.txt 保持一致。

测试 LLM.txt 中记录的 4 个模型（均用最短的测试数据），全部在同一个
cn-beijing workspace，共用一把 key：
  1. 主模型         qwen3.5-flash          （OpenAI 兼容 chat，回答/改写/摘要/图谱抽取）
  2. 图谱校验模型   qwen-flash             （OpenAI 兼容 chat，异模型交叉校验）
  3. 嵌入模型       qwen3.7-text-embedding （OpenAI 兼容 embeddings，1024 维）
  4. 重排模型       qwen3-rerank           （MaaS rerank 接口，直接 POST）

另含两项专项检查：
  - 嵌入维度是否仍与 config.EMBED_VECTOR_DIM 一致（不一致必须重建向量索引）
  - 小说正文是否被内容审核拦截（dashscope-us 公共端点会 400，本 workspace 不会）

API Key 从环境变量（或项目根目录 .env）读取，不在代码中硬编码：
  RAG_PUBLIC_CHAT_API_KEY / RAG_PUBLIC_EMBED_API_KEY / RAG_PUBLIC_RERANK_API_KEY
用法：
    .venv/bin/python scripts/test_public_llm.py
"""

import os
import sys
import time

import requests
from openai import OpenAI

# ---------------------------------------------------------------- 配置
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# 端点与模型一律取自 rag.config，避免脚本与项目实际配置漂移 ——
# 换模型时只需改 config.py 与 LLM.txt 两处，本脚本自动跟随
from rag import config  # noqa: E402

CHAT_BASE_URL = config.ALIYUN_CHAT_BASE_URL
CHAT_MAIN_MODEL = config.ALIYUN_MAIN_MODEL
CHAT_VALIDATE_MODEL = config.ALIYUN_VALIDATE_MODEL

EMBED_BASE_URL = config.ALIYUN_EMBED_BASE_URL
EMBED_MODEL = config.ALIYUN_EMBED_MODEL
EXPECTED_DIM = config.EMBED_VECTOR_DIM

RERANK_URL = config.ALIYUN_RERANK_URL
RERANK_MODEL = config.ALIYUN_RERANK_MODEL

TIMEOUT = 60

# 小说正文片段：验证该端点是否带内容审核（dashscope-us 会 400 data_inspection_failed）
NOVEL_SNIPPET = (
    "丁元英从柏林回到北京，私募基金已经解散。他坐在古城的出租屋里，点上一支烟，"
    "想着王庙村的扶贫计划究竟是杀富济贫还是文化属性的必然。"
)


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
def _extra_body() -> dict:
    """与 DavyLLM 一致的思考开关（reasoning token 可占输出 99%，见 config）。"""
    return {"enable_thinking": config.ALIYUN_ENABLE_THINKING}


def test_chat(client: OpenAI, model: str) -> bool:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "回复“连通”两个字即可。"}],
        max_tokens=16,
        extra_body=_extra_body(),
    )
    text = (resp.choices[0].message.content or "").strip()
    details = resp.usage.completion_tokens_details
    reasoning = getattr(details, "reasoning_tokens", None) if details else None
    print(f"    模型返回: {text[:50]!r}")
    print(f"    输出 token={resp.usage.completion_tokens}，其中 reasoning={reasoning}"
          f"（enable_thinking={config.ALIYUN_ENABLE_THINKING}）")
    return bool(text)


def test_novel_inspection(client: OpenAI) -> bool:
    """小说正文是否被内容审核拦截（本项目语料是小说，这条必须通过）。"""
    resp = client.chat.completions.create(
        model=CHAT_MAIN_MODEL,
        messages=[{"role": "user", "content": f"用一句话概括：\n{NOVEL_SNIPPET}"}],
        extra_body=_extra_body(),
    )
    text = (resp.choices[0].message.content or "").strip()
    print(f"    概括结果: {text[:60]!r}")
    return bool(text)


def test_embedding(api_key: str) -> bool:
    client = OpenAI(api_key=api_key, base_url=EMBED_BASE_URL, timeout=TIMEOUT)
    resp = client.embeddings.create(model=EMBED_MODEL, input="连通性测试")
    vec = resp.data[0].embedding
    print(f"    向量维度: {len(vec)}，前 3 维: {[round(v, 4) for v in vec[:3]]}")
    if len(vec) != EXPECTED_DIM:
        print(f"    ❌ 维度与 config.EMBED_VECTOR_DIM({EXPECTED_DIM}) 不符 —— "
              f"必须同步 config 并重建各语料的 data/vector/")
        return False
    return True


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
        (f"嵌入（{EMBED_MODEL}，应为 {EXPECTED_DIM} 维）", lambda: test_embedding(embed_key)),
        (f"重排（{RERANK_MODEL}）", lambda: test_rerank(rerank_key)),
        ("小说正文内容审核", lambda: test_novel_inspection(chat_client)),
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
