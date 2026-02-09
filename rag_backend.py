import os
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
        
    def ingest_documents(self, folder_path):
        """加载指定目录下的多种格式文件并创建向量库"""
        if not os.path.exists(folder_path):
            return f"错误：目录 {folder_path} 不存在"

        print(f"正在从 {folder_path} 加载文档...")
        
        # 支持多种格式
        loaders = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".docx": Docx2txtLoader,
            ".md": TextLoader
        }

        docs = []
        for ext, loader_cls in loaders.items():
            loader = DirectoryLoader(
                folder_path, 
                glob=f"**/*{ext}", 
                loader_cls=loader_cls,
                silent_errors=True
            )
            try:
                loaded_docs = loader.load()
                docs.extend(loaded_docs)
            except Exception as e:
                print(f"加载 {ext} 文件时出错: {e}")
        
        if not docs:
            return "未在指定目录下找到支持的文档 (.pdf, .docx, .txt, .md)"

        # 过滤过短内容
        docs = [d for d in docs if len(d.page_content.strip()) > 50]

        # 注入元数据
        for doc in docs:
            file_name = os.path.basename(doc.metadata.get('source', '未知文件'))
            doc.page_content = f"--- 来源于《{file_name}》 ---\n{doc.page_content}"
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)
        
        print(f"正在创建向量库，包含 {len(splits)} 个分块...")
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        
        # MMR 检索
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.5}
        )
        return f"成功索引了 {len(docs)} 个文件，生成了 {len(splits)} 个知识点。"

    def get_response(self, query, api_key, model_name="mimo-v2-flash", api_base="https://api.xiaomimimo.com/v1"):
        """获取 RAG 回答"""
        if not self.retriever:
            return "请先加载文档。"
            
        llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=api_base,
            temperature=0.3
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
