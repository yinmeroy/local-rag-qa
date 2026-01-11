import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

class DocProcessor:
    """封装文档加载和分块"""
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？"]
        )

    def load_document(self, file_path: str):
        """加载PDF/TXT文档"""
        if not os.path.exists(file_path):
            raise ValueError(f"文件不存在：{file_path}")
        
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError("仅支持PDF/TXT格式")
        
        return loader.load()

    def split_document(self, documents):
        """文档分块"""
        return self.text_splitter.split_documents(documents)

    def process_file(self, file_path: str):
        """一站式处理：加载+分块"""
        docs = self.load_document(file_path)
        split_docs = self.split_document(docs)
        # 给每个分块添加元数据
        for i, doc in enumerate(split_docs):
            doc.metadata["chunk_id"] = i
            doc.metadata["source"] = os.path.basename(file_path)
        return split_docs

# 单例实例
doc_processor = DocProcessor()