"""
Streamlit Web 入口 —— 完整的 RAG 检索问答流程，用于客户端使用。

调用链:
  config → bootstrap.build_query_engine(corpus_slug) → query_engine → 查询输出

多书：侧边栏选书，引擎按语料 slug 缓存（@st.cache_resource 带参数），
对话历史每本书独立。装配逻辑统一在 rag/engine/bootstrap.py；检索日志通过
capture_pipeline_logs 按查询捕获（与缓存的引擎解耦，
不再出现"第二次查询起日志为空"的问题）。
"""

import logging

import streamlit as st

from rag import corpus
from rag.engine.bootstrap import init_settings, build_query_engine, format_source_nodes
from rag.logging_utils import capture_pipeline_logs
from rag.metering import capture_usage, step_timer

logger = logging.getLogger(__name__)

st.set_page_config(page_title="知识库问答", page_icon="📚")

# ---------- 侧边栏：选书 ----------
profiles = corpus.list_corpora()
if not profiles:
    st.error("corpora/ 下没有可用语料，请先创建语料目录（corpus.json + raw/）")
    st.stop()

slugs = [p.slug for p in profiles]
default_idx = slugs.index(corpus.get_active_slug()) if corpus.get_active_slug() in slugs else 0
with st.sidebar:
    st.header("📖 选择书目")
    selected = st.selectbox(
        "语料",
        profiles,
        index=default_idx,
        format_func=lambda p: f"《{p.title}》",
    )
    if selected.description:
        st.caption(selected.description)

st.title(f"📚 《{selected.title}》知识库问答")


# ---------- 加载索引（按语料缓存，进程内每本书只构建一次） ----------
@st.cache_resource
def load_index_and_engine(slug: str):
    """初始化 Settings 并加载/构建指定语料的索引，组装检索 + 查询引擎。"""
    init_settings()
    return build_query_engine(slug)


# ---------- 对话管理（每本书独立历史） ----------
if "messages_by_corpus" not in st.session_state:
    st.session_state.messages_by_corpus = {}
messages = st.session_state.messages_by_corpus.setdefault(selected.slug, [])

# 渲染历史消息
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- 查询入口 ----------
if prompt := st.chat_input("请输入你的问题"):
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        query_engine = load_index_and_engine(selected.slug)
    except Exception as e:
        st.error(f"索引加载/构建失败: {e}")
        st.stop()

    # ---- 步骤 3：执行查询（检索阶段；流式开启时回答在步骤 4 逐块渲染） ----
    with st.chat_message("assistant"):
        # 计量覆盖到步骤 4：流式回答惰性生成，usage 在消费 response_gen 时才产生
        with capture_usage() as meter:
            with st.spinner("思考中..."):
                with capture_pipeline_logs() as cap:
                    try:
                        with step_timer("步骤 3 检索"):
                            response = query_engine.query(prompt)
                    except Exception as e:
                        logger.exception("查询失败")
                        st.error(f"查询失败（网络/LLM 服务异常）: {e}")
                        st.stop()
            run_logs = cap.drain()

            # ---- 步骤 4：LLM 生成回答（流式逐块渲染） ----
            st.markdown("**步骤 4：LLM 生成回答**")
            try:
                with step_timer("步骤 4 回答生成"):
                    if hasattr(response, "response_gen"):
                        answer = str(st.write_stream(response.response_gen))
                    else:
                        answer = str(response)
                        st.markdown(answer)
            except Exception as e:
                logger.exception("回答生成失败")
                st.error(f"回答生成失败（网络/LLM 服务异常）: {e}")
                st.stop()

        # ---- 步骤 5：输出参考文献 ----
        st.markdown("**步骤 5：输出参考文献**")
        if response.source_nodes:
            for title, preview in format_source_nodes(response.source_nodes):
                st.caption(title)
                st.text(preview + "...")
        else:
            st.caption("（无参考文献）")

        # 本次查询的用量与耗时（token 取自服务端 usage）
        meter_lines = meter.step_lines() + meter.summary_lines()
        if meter_lines:
            with st.expander("用量与耗时"):
                for line in meter_lines:
                    st.text(line)

        # 本次查询的运行日志（检索管线内部产生）
        if run_logs:
            with st.expander("运行流程"):
                for line in run_logs:
                    st.text(line)

    messages.append({"role": "assistant", "content": answer})
