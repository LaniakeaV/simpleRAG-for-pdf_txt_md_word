import os
import hashlib
from langchain_community.document_loaders import (
    PyPDFLoader, 
    DirectoryLoader, 
    TextLoader, 
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class RAGSystem:
    def __init__(self, api_key=None):
        self.api_key = api_key
        # 模型配置：优先使用本地路径，不存在则自动下载
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.local_model_path = os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")
        
        # 兼容性处理：如果本地文件夹存在且不为空，则加载本地
        embed_model = self.local_model_path if os.path.exists(self.local_model_path) and os.listdir(self.local_model_path) else self.model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=embed_model)
        
        self.vector_store = None
        self.retriever = None

    @staticmethod
    def supported_extensions():
        return (".pdf", ".txt", ".docx", ".md")

    def _get_supported_files(self, folder_path):
        """返回目录下支持的文件路径列表。"""
        supported_files = []
        for root, _, files in os.walk(folder_path):
            for file_name in files:
                if file_name.lower().endswith(self.supported_extensions()):
                    supported_files.append(os.path.join(root, file_name))
        return sorted(supported_files)

    def _build_index_fingerprint(self, folder_path, files):
        """根据目录路径、文件路径、大小和修改时间生成索引缓存指纹。"""
        hasher = hashlib.sha256()
        hasher.update(os.path.abspath(folder_path).encode("utf-8", errors="ignore"))
        for file_path in files:
            try:
                stat = os.stat(file_path)
            except OSError:
                continue
            relative_path = os.path.relpath(file_path, folder_path)
            hasher.update(relative_path.encode("utf-8", errors="ignore"))
            hasher.update(str(stat.st_size).encode("utf-8"))
            hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
        return hasher.hexdigest()[:16]

    def _get_index_dir(self, folder_path, files):
        fingerprint = self._build_index_fingerprint(folder_path, files)
        return os.path.join(os.path.dirname(__file__), "faiss_index", fingerprint)

    def _set_retriever(self):
        """基于当前向量库创建 MMR 检索器。"""
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.5}
        )
        
    def ingest_documents(self, folder_path, use_cache=True, force_rebuild=False):
        """加载指定目录下的多种格式文件并创建向量库"""
        if not os.path.exists(folder_path):
            return f"错误：目录 {folder_path} 不存在"

        print(f"正在从 {folder_path} 加载文档...")

        supported_files = self._get_supported_files(folder_path)
        if not supported_files:
            return "未在指定目录下找到支持的文档 (.pdf, .docx, .txt, .md)"

        index_dir = self._get_index_dir(folder_path, supported_files)
        if use_cache and not force_rebuild and os.path.exists(os.path.join(index_dir, "index.faiss")):
            self.vector_store = FAISS.load_local(
                index_dir,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self._set_retriever()
            return f"已从本地缓存加载索引，共匹配 {len(supported_files)} 个源文件。"
        
        # 支持多种格式。文本类文件优先按 UTF-8 读取，减少 Windows 中文环境乱码概率。
        loaders = {
            ".pdf": (PyPDFLoader, {}),
            ".txt": (TextLoader, {"encoding": "utf-8", "autodetect_encoding": True}),
            ".docx": (Docx2txtLoader, {}),
            ".md": (TextLoader, {"encoding": "utf-8", "autodetect_encoding": True})
        }

        docs = []
        source_files = set()
        for ext, (loader_cls, loader_kwargs) in loaders.items():
            loader = DirectoryLoader(
                folder_path, 
                glob=f"**/*{ext}", 
                loader_cls=loader_cls,
                loader_kwargs=loader_kwargs,
                silent_errors=True
            )
            try:
                loaded_docs = loader.load()
                docs.extend(loaded_docs)
                source_files.update(
                    doc.metadata.get('source')
                    for doc in loaded_docs
                    if doc.metadata.get('source')
                )
            except Exception as e:
                print(f"加载 {ext} 文件时出错: {e}")
        
        if not docs:
            return "未在指定目录下找到支持的文档 (.pdf, .docx, .txt, .md)"

        # 过滤过短内容
        docs = [d for d in docs if len(d.page_content.strip()) > 50]
        if not docs:
            return "找到的文档内容过短或为空，无法建立索引。"

        # 注入元数据
        for doc in docs:
            file_name = os.path.basename(doc.metadata.get('source', '未知文件'))
            doc.page_content = f"--- 来源于《{file_name}》 ---\n{doc.page_content}"
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)
        
        print(f"正在创建向量库，包含 {len(splits)} 个分块...")
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        os.makedirs(index_dir, exist_ok=True)
        self.vector_store.save_local(index_dir)
        
        # MMR 检索
        self._set_retriever()
        return f"成功索引了 {len(source_files)} 个文件，加载了 {len(docs)} 个文档片段，生成了 {len(splits)} 个知识点。"

    def get_response(self, query, api_key, model_name, api_base, timeout=60, max_retries=2):
        """获取 RAG 回答"""
        if not self.retriever:
            return "请先加载文档。"
        if not api_key:
            return "请先配置 API Key。"
        if not api_base:
            return "请先配置 API 地址。"
        if not model_name:
            return "请先配置模型名称。"
            
        llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=api_base,
            temperature=0.3,
            timeout=timeout,
            max_retries=max_retries
        )
        
        system_prompt = (
            "你是一个专业的资料助理。请结合上下文给出建议。\n"
            "1. 必须说明信息来源（文件名）。\n"
            "2. 如果多个文件观点不同，请进行对比。\n"
            "3. 无法回答时，请说明原因并基于通用常识给出简单提示。\n"
            "\n上下文：\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)
        
        response = rag_chain.invoke({"input": query})
        return response["answer"]
