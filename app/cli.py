"""
CLI 入口 —— 完整的 RAG 检索问答流程。

用法：
  python -m app.cli "你的问题"
  python -m app.cli            # 使用默认示例问题

流程：
  步骤 0：开始运行
  步骤 1：初始化全局设置（embedding + LLM）
  步骤 2：构建查询引擎（索引 → 检索器 → 查询引擎，装配逻辑在 rag/engine/bootstrap.py）
  步骤 3：执行查询（检索日志由 capture_pipeline_logs 捕获后统一输出）
  步骤 4：LLM 生成回答
  步骤 5：输出参考文献
"""

import sys

from rag.engine.bootstrap import init_settings, build_query_engine, format_source_nodes
from rag.logging_utils import capture_pipeline_logs

DEFAULT_QUESTION = "欧阳雪为什么为难丁元英"


def print_step(msg: str):
    """统一打印步骤日志。"""
    print(msg, flush=True)


def run_query(question: str) -> None:
    """执行一次完整查询并打印结果。"""
    # ---- 步骤 0 ----
    print_step("步骤 0：开始运行")

    # ---- 步骤 1：初始化全局设置 ----
    print_step("步骤 1：初始化全局设置 (embedding + LLM)")
    init_settings()

    # ---- 步骤 2：构建查询引擎 ----
    print_step("步骤 2：构建查询引擎")
    print("  正在进入索引构建/加载流程，请稍候...", flush=True)

    with capture_pipeline_logs() as cap:
        try:
            query_engine = build_query_engine()
        except Exception as e:
            for line in cap.drain():
                print(f"  {line}")
            print(f"构建查询引擎失败: {e}", file=sys.stderr)
            raise SystemExit(1)

    for line in cap.drain():
        print(f"  {line}")
    print_step("  组装完成")

    # ---- 步骤 3：执行查询 ----
    print_step("步骤 3：执行查询")
    with capture_pipeline_logs() as cap:
        try:
            response = query_engine.query(question)
        except Exception as e:
            for line in cap.drain():
                print(f"  {line}")
            print(f"查询失败（网络/LLM 服务异常）: {e}", file=sys.stderr)
            raise SystemExit(1)

    # ---- 打印检索日志 ----
    print("=" * 60, flush=True)
    for line in cap.drain():
        print(f"  {line}", flush=True)
    print("=" * 60, flush=True)

    # ---- 步骤 4：LLM 生成回答 ----
    print_step("步骤 4：LLM 生成回答")
    print(f"  {response}")

    # ---- 步骤 5：输出参考文献 ----
    print_step("步骤 5：输出参考文献")
    if response.source_nodes:
        for title, preview in format_source_nodes(response.source_nodes):
            print(f"  {title} | {preview}...")
    else:
        print("  （无参考文献）")


# ======================== 主入口 ========================
if __name__ == "__main__":
    q = " ".join(sys.argv[1:]).strip() or DEFAULT_QUESTION
    run_query(q)
