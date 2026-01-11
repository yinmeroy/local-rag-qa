import os
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from config.settings import FAISS_INDEX_PATH, TOP_K

class CustomEmbeddings(Embeddings):
    """适配动态嵌入模型"""
    def __init__(self, embed_func):
        self.embed_func = embed_func

    def embed_documents(self, texts):
        return [self.embed_func(text) for text in texts]
    
    def embed_query(self, text):
        return self.embed_func(text)

class FaissManager:
    """封装FAISS向量库管理（适配动态嵌入模型）"""
    def __init__(self):
        self.index_path = FAISS_INDEX_PATH
        self.top_k = TOP_K
        self.embeddings = None  # 动态设置
        self.db = None

    def load_or_create_db(self, documents=None):
        """加载已有向量库，无则创建"""
        if not self.embeddings:
            raise Exception("请先初始化嵌入模型！")
        
        if os.path.exists(self.index_path):
            self.db = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        elif documents:
            self.db = FAISS.from_documents(documents, self.embeddings)
            self.db.save_local(self.index_path)
        return self.db

    def add_documents(self, documents):
        """添加文档到向量库"""
        if not self.embeddings:
            raise Exception("请先初始化嵌入模型！")
        
        if not self.db:
            self.load_or_create_db(documents)
        else:
            self.db.add_documents(documents)
            self.db.save_local(self.index_path)

    def similarity_search(self, query):
        """相似性检索"""
        if not self.embeddings:
            raise Exception("请先初始化嵌入模型！")
        
        if not self.db:
            self.load_or_create_db()
        if not self.db:
            return []
        return self.db.similarity_search(query, k=self.top_k)

# 单例实例
faiss_manager = FaissManager()