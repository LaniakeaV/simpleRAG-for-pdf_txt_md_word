# 🚀 Local-Multi-RAG: 通用本地知识库助手

这是一个基于 **LangChain** 和 **Streamlit** 构建的轻量级 RAG（检索增强生成）系统。它允许你直接对本地任意文件夹内的文档（PDF, Word, TXT, MD）进行提问。

## ✨ 功能特点
- **多格式支持**：兼容 `.pdf`, `.docx`, `.txt`, `.md`。
- **自选目录**：支持在界面上动态指定任意本本地文件夹。
- **本地向量化**：使用 `sentence-transformers` 进行本地嵌入，保护隐私。
- **多样性检索**：采用 MMR (Maximal Marginal Relevance) 算法，确保回答引用自多份不同文档。
- **出处溯源**：AI 回答会明确标注信息来源于哪个文件。

## 🛠️ 安装与运行

### 1. 克隆项目
```bash
git clone <your-repo-url>
cd RAG
```

### 2. 创建虚拟环境并安装依赖
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 运行应用
- **Windows (推荐)**: 资源管理器中直接双击 `双击启动应用.bat` 文件。
- **命令行模式**:
  ```bash
  streamlit run app.py
  ```

## 📖 使用指南
1. **API Key**: 启动后在侧边栏填入兼容 OpenAI 接口的 API Key。
2. **文件夹路径**: 在侧边栏输入你想分析的本地文件夹绝对路径（默认会自动扫描项目下的 `data` 目录）。
3. **建立索引**: 点击“开始分析”，系统会自动处理文档并建立本地向量索引。
4. **对话提问**: 在右侧聊天框开始提问！

## 📄 注意事项
- 首次运行需要自动下载 Embedding 模型（约 80MB）。
- 服务器地址已预设为：`https://api.xiaomimimo.com/v1`。

## 🧩 核心算法：分块 (Chunking) 逻辑
本项目采用了 **RecursiveCharacterTextSplitter (递归字符切分器)**，这是目前最推荐的通用分块方案。其逻辑如下：
1. **语义优先**：它会按级尝试分隔符：`段落 (\n\n) -> 换行 (\n) -> 空格 -> 字符`。这保证了 AI 看到的内容尽量是完整的段落或句子。
2. **窗口配置**：
   - `chunk_size=600`: 保证每个片段包含足够的信息量（约 600 字符）。
   - `chunk_overlap=150`: 相邻块之间保留 150 字的重叠，像“胶水”一样保证语义不被从中间切断，减少 AI 的理解误差。
3. **多样性检索**：检索阶段使用 **MMR (Maximal Marginal Relevance)** 算法，能有效避免 AI 只读到重复段落，确保答案覆盖多份文档的观点。
