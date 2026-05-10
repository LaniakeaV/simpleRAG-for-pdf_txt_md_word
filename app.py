import streamlit as st
from rag_backend import RAGSystem
import os
import tkinter as tk
from tkinter import filedialog

st.set_page_config(page_title="通用本地 RAG 系统", layout="wide", page_icon="📚")

# 窗口选择文件夹的函数
def select_folder():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askdirectory(master=root)
    root.destroy()
    return path

st.title("📚 通用本地 RAG 系统")
st.markdown("支持 PDF, Word, TXT, Markdown 格式的本地知识库检索。")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置中心")
    api_key = st.text_input("1. 填入 API Key", type="password", placeholder="sk-...")
    api_base = st.text_input("2. API 地址", placeholder="例如：https://api.openai.com/v1")
    model_name = st.text_input("3. 模型名称", placeholder="例如：gpt-4o-mini")
    
    st.divider()
    st.subheader("📂 资料来源")
    
    if 'folder_path' not in st.session_state:
        st.session_state.folder_path = os.path.join(os.getcwd(), "data")

    manual_path = st.text_input("文件夹路径", value=st.session_state.folder_path)
    if manual_path and manual_path != st.session_state.folder_path:
        st.session_state.folder_path = manual_path

    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"当前选定: `{st.session_state.folder_path}`")
    with col2:
        if st.button("📁 选择"):
            selected_path = select_folder()
            if selected_path:
                st.session_state.folder_path = selected_path
                st.rerun()
    
    use_cache = st.checkbox("优先使用本地索引缓存", value=True)
    force_rebuild = st.checkbox("强制重新建立索引", value=False)

    st.info("💡 提示：更改路径后需要重新点击下方按钮。索引会缓存在 faiss_index 目录。")
    
    if st.button("🚀 开始分析文件夹中文件"):
        if not st.session_state.folder_path or not os.path.exists(st.session_state.folder_path):
            st.error("请选择有效的文件夹！")
        else:
            with st.status("正在建立本地知识索引...", expanded=True) as status:
                st.write("正在准备嵌入模型（首次加载可能稍慢）...")
                if 'rag' not in st.session_state:
                    st.session_state.rag = RAGSystem()
                
                result = st.session_state.rag.ingest_documents(
                    st.session_state.folder_path,
                    use_cache=use_cache,
                    force_rebuild=force_rebuild
                )
                
                if "错误" in result:
                    status.update(label="索引失败", state="error")
                    st.error(result)
                else:
                    status.update(label="✅ 索引完成！", state="complete", expanded=False)
                    st.success(result)

# 聊天界面 (后续逻辑保持不变)
# ... 保持之前的聊天逻辑不变 ...

# 聊天界面
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 输入框
if prompt := st.chat_input("基于本地文档提问..."):
    if not api_key:
        st.warning("请在侧边栏输入 API Key。")
    elif not api_base:
        st.warning("请在侧边栏输入 API 地址。")
    elif not model_name:
        st.warning("请在侧边栏输入模型名称。")
    elif 'rag' not in st.session_state or st.session_state.rag.retriever is None:
        st.warning("请先点击侧边栏按钮建立索引。")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("查阅文档中..."):
                try:
                    response = st.session_state.rag.get_response(
                        prompt, 
                        api_key, 
                        model_name=model_name, 
                        api_base=api_base,
                        timeout=60,
                        max_retries=2
                    )
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"对话出错: {str(e)}")
