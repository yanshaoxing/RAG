"""
CLI 入口 —— 完整的 RAG 检索问答流程。

用法：
  python -m app.cli "你的问题"              # 单次查询（默认语料）
  python -m app.cli --corpus <slug> "问题"  # 指定语料（书）查询
  python -m app.cli --list                  # 列出全部可用语料
  python -m app.cli --rebuild vector        # 删除某阶段及其下游产物（预览；加 --yes 真删）
  python -m app.cli                         # 交互模式：索引加载一次，循环提问（exit/quit/空行 退出）

流程：
  步骤 0：开始运行
  步骤 1：初始化全局设置（embedding + LLM）
  步骤 2：构建查询引擎（索引 → 检索器 → 查询引擎，装配逻辑在 rag/engine/bootstrap.py）
  步骤 3：执行查询（检索日志由 capture_pipeline_logs 捕获后统一输出）
  步骤 4：LLM 生成回答（流式逐块打印）
  步骤 5：输出参考文献
"""

import argparse
import os
import sys

from rag import config, corpus
from rag.engine.bootstrap import build_query_engine, format_source_nodes, init_settings
from rag.logging_utils import capture_pipeline_logs
from rag.metering import capture_usage, step_timer


def print_step(msg: str):
    """统一打印步骤日志。"""
    print(msg, flush=True)


def build_engine(corpus_slug=None):
    """初始化 Settings 并构建查询引擎（交互模式下只执行一次）。"""
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
            query_engine = build_query_engine(corpus_slug)
        except Exception as e:
            for line in cap.drain():
                print(f"  {line}")
            print(f"构建查询引擎失败: {e}", file=sys.stderr)
            raise SystemExit(1)

    for line in cap.drain():
        print(f"  {line}")
    print_step("  组装完成")
    return query_engine


def run_query(query_engine, question: str) -> bool:
    """对已构建的引擎执行一次查询并打印结果。返回是否成功（供单次模式设置 exit code）。"""
    # 计量上下文必须覆盖到步骤 4：流式回答是惰性生成的，
    # 那次 LLM 调用的 usage 在 response_gen 被消费时才产生
    with capture_usage() as meter:
        # ---- 步骤 3：执行查询 ----
        print_step("步骤 3：执行查询")
        with capture_pipeline_logs() as cap:
            try:
                with step_timer("步骤 3 检索（改写+三路+过滤+RRF+重排+图谱）"):
                    response = query_engine.query(question)
            except Exception as e:
                for line in cap.drain():
                    print(f"  {line}")
                print(f"查询失败（网络/LLM 服务异常）: {e}", file=sys.stderr)
                return False

        # ---- 打印检索日志 ----
        print("=" * 60, flush=True)
        for line in cap.drain():
            print(f"  {line}", flush=True)
        print("=" * 60, flush=True)

        # ---- 步骤 4：LLM 生成回答（流式逐块打印） ----
        print_step("步骤 4：LLM 生成回答")
        try:
            with step_timer("步骤 4 回答生成"):
                if hasattr(response, "response_gen"):
                    print("  ", end="", flush=True)
                    for chunk in response.response_gen:
                        print(chunk, end="", flush=True)
                    print(flush=True)
                else:
                    print(f"  {response}")
        except Exception as e:
            print(f"\n回答生成失败（网络/LLM 服务异常）: {e}", file=sys.stderr)
            return False

    # ---- 步骤 5：输出参考文献 ----
    print_step("步骤 5：输出参考文献")
    if response.source_nodes:
        for title, preview in format_source_nodes(response.source_nodes):
            print(f"  {title} | {preview}...")
    else:
        print("  （无参考文献）")

    # ---- 步骤 6：用量与耗时 ----
    for line in meter.step_lines() + meter.summary_lines():
        print(f"  {line}", flush=True)
    return True


def interactive_loop(query_engine) -> None:
    """交互模式：循环读取问题，索引/引擎复用不重建。"""
    print_step("进入交互模式（输入 exit / quit / 空行 退出）")
    while True:
        try:
            question = input("\n请输入问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in ("exit", "quit", "q"):
            break
        run_query(query_engine, question)
    print_step("已退出")


def run_rebuild(stage_key: str, corpus_slug: str | None, apply: bool) -> int:
    """删除指定阶段及其下游依赖闭包的产物目录（默认仅预览，--yes 才真删）。

    阶段路径随激活语料变化，故先绑定目标语料再解析路径。删除后由下次查询触发重建。
    返回进程退出码。
    """
    from rag.indexing import staged_indexer

    slug = corpus_slug or corpus.get_active_slug()
    try:
        # --rebuild 是构建期操作，需真正切换激活语料（config 动态路径随之指向该书）
        profile = corpus.set_active_corpus(slug)
    except (FileNotFoundError, ValueError) as e:
        available = "、".join(p.slug for p in corpus.list_corpora()) or "（无）"
        print(f"{e}\n可用语料：{available}", file=sys.stderr)
        return 1

    try:
        stages = staged_indexer.plan_rebuild(stage_key)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    print(f"语料《{profile.title}》（{slug}）：重建阶段「{stage_key}」将删除以下阶段产物"
          f"（含下游依赖闭包）：")
    for st in stages:
        tag = "" if os.path.exists(st.path) else "  [目录不存在，跳过]"
        print(f"  - {st.name}（{st.key}）: {st.path}{tag}")

    if not apply:
        print("\n以上仅为预览。确认无误后加 --yes 真正删除，"
              f"随后运行 `python -m app.cli -c {slug} \"<问题>\"` 触发重建。")
        return 0

    staged_indexer.rebuild_stages(stage_key, apply=True)
    print(f"\n已删除。下次运行 `python -m app.cli -c {slug} \"<问题>\"` 将自动重建这些阶段。")
    return 0


# ======================== 主入口 ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m app.cli", description="RAG 检索问答 CLI")
    parser.add_argument("question", nargs="*", help="问题（省略则进入交互模式）")
    parser.add_argument("-c", "--corpus", default=None, metavar="SLUG",
                        help="语料（书）slug，默认取 RAG_CORPUS / 配置默认语料")
    parser.add_argument("--list", action="store_true", help="列出全部可用语料后退出")
    parser.add_argument("--rebuild", metavar="STAGE",
                        choices=["chunks", "summary", "bm25", "vector", "graph"],
                        help="删除该阶段及其下游阶段的产物目录（预览；配合 --yes 真删），下次运行时重建")
    parser.add_argument("--yes", action="store_true", help="配合 --rebuild：确认真正删除")
    args = parser.parse_args()

    if args.rebuild:
        raise SystemExit(run_rebuild(args.rebuild, args.corpus, args.yes))

    if args.list:
        profiles = corpus.list_corpora()
        if not profiles:
            print("（corpora/ 下没有可用语料）")
        for p in profiles:
            marker = " *" if p.slug == corpus.get_active_slug() else ""
            print(f"  {p.slug}  《{p.title}》{marker}  {p.description}")
            # embedding 全局约束：向量索引由别的嵌入模型构建时，提前提示需重建
            status = corpus.probe_vector_index(p)
            if status.built and not status.matches_current:
                print(f"      ⚠️ 向量索引由 {status.embed_model!r} 构建，与当前嵌入模型 "
                      f"{config.ACTIVE_EMBED_MODEL_NAME!r} 不一致；查询该书前需重建向量阶段："
                      f"python -m app.cli --rebuild vector -c {p.slug} --yes")
        raise SystemExit(0)

    if args.corpus:
        # 这里只校验档案可加载，不改全局激活态 ——
        # 语料切换统一由 build_query_engine(corpus_slug) 在构建锁内完成
        try:
            profile = corpus.load_profile(args.corpus)
        except (FileNotFoundError, ValueError) as e:
            available = "、".join(p.slug for p in corpus.list_corpora()) or "（无）"
            print(f"{e}\n可用语料：{available}", file=sys.stderr)
            raise SystemExit(1)
    else:
        profile = corpus.get_active_profile()
    print(f"当前语料：《{profile.title}》（{profile.slug}）", flush=True)

    q = " ".join(args.question).strip()
    engine = build_engine(args.corpus)   # None = 沿用当前激活语料
    if q:
        # 单次模式：查询失败以非 0 退出码结束，便于脚本化调用判断
        ok = run_query(engine, q)
        raise SystemExit(0 if ok else 1)
    else:
        interactive_loop(engine)
