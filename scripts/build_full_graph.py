"""构建全书的完整知识图谱"""
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from llama_index.core import Settings

from rag import config
from rag.engine.bootstrap import init_settings
from rag.graph.graph_constructor import build_graph
from rag.ingestion.preprocessor import load_documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    print("=" * 60)
    print("构建全书知识图谱")
    print("=" * 60)

    # 命令行参数：
    #   python build_full_graph.py          → 自动续传（默认）
    #   python build_full_graph.py --force  → 删除缓存，从头开始
    #   python build_full_graph.py 31       → 从第 31 个 chunk 续传
    force_rebuild = False
    resume_from = None
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--force":
            force_rebuild = True
            print("  🔥 --force: 将删除旧缓存，从头开始")
        else:
            resume_from = int(arg)
            print(f"  从第 {resume_from} 个 chunk 续传")

    # 初始化 Settings（embedding + LLM 装配统一走 bootstrap，随 provider 配置切换）
    init_settings()
    answer_model = getattr(Settings.llm, "model_name", None) or getattr(Settings.llm, "model", "?")
    embed_model = getattr(Settings.embed_model, "model_name", "?")
    print(f"\nLLM: {config.ANSWER_PROVIDER} ({answer_model})")
    print(f"Embed: {config.EMBED_PROVIDER} ({embed_model})")

    # 加载全部文档
    print("\n[1/2] 加载全部文档...")
    raw_documents = load_documents()
    print(f"  加载 {len(raw_documents)} 个文档")

    # 构建知识图谱
    print("\n[2/2] 构建知识图谱...")
    graph_index = build_graph(
        raw_documents, Settings.llm,
        force_rebuild=force_rebuild, resume_from=resume_from,
    )

    if graph_index:
        print("\n" + "=" * 60)
        print("知识图谱构建成功！")
        print(f"  持久化到: {config.GRAPH_DB_DIR}")
        print("=" * 60)
    else:
        print("\n知识图谱构建失败或无有效三元组")