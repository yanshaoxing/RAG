"""
Streamlit Web 入口 —— 完整的 RAG 检索问答流程，用于客户端使用。

调用链:
  config → bootstrap.build_query_engine → query_engine → 查询输出

装配逻辑统一在 rag/engine/bootstrap.py；检索日志通过
capture_pipeline_logs 按查询捕获（与 @st.cache_resource 缓存的引擎解耦，
不再出现"第二次查询起日志为空"的问题）。
"""

import logging

import streamlit as st

from rag.engine.bootstrap import init_settings, build_query_engine, format_source_nodes
from rag.logging_utils import capture_pipeline_logs

logger = logging.getLogger(__name__)

st.set_page_config(page_title="知识库问答", page_icon="📚")
st.title("📚 内部知识库问答")


# ---------- 加载索引（自动判断构建/加载，进程内只执行一次） ----------
@st.cache_resource
def load_index_and_engine():
    """初始化 Settings 并加载/构建索引，组装检索 + 查询引擎。"""
    init_settings()
    return build_query_engine()


# ---------- 对话管理 ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- 查询入口 ----------
if prompt := st.chat_input("请输入你的问题"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        query_engine = load_index_and_engine()
    except Exception as e:
        st.error(f"索引加载/构建失败: {e}")
        st.stop()

    # ---- 步骤 3：执行查询（检索阶段；流式开启时回答在步骤 4 逐块渲染） ----
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            with capture_pipeline_logs() as cap:
                try:
                    response = query_engine.query(prompt)
                except Exception as e:
                    logger.exception("查询失败")
                    st.error(f"查询失败（网络/LLM 服务异常）: {e}")
                    st.stop()
        run_logs = cap.drain()

        # ---- 步骤 4：LLM 生成回答（流式逐块渲染） ----
        st.markdown("**步骤 4：LLM 生成回答**")
        try:
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

        # 本次查询的运行日志（检索管线内部产生）
        if run_logs:
            with st.expander("运行流程"):
                for line in run_logs:
                    st.text(line)

    st.session_state.messages.append({"role": "assistant", "content": answer})
