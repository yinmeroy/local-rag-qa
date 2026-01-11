import os
import requests
import ollama
from config.settings import OLLAMA_HOST

# 全局设置Ollama地址
os.environ["OLLAMA_HOST"] = OLLAMA_HOST

class OllamaClient:
    """封装Ollama调用，支持懒加载+未下载模型提示"""
    def __init__(self, llm_model: str, embedding_model: str):
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        # 初始化时只检测，不下载
        self.llm_exists = self._check_model_exists(llm_model)
        self.embedding_exists = self._check_model_exists(embedding_model)

    def _check_model_exists(self, model_name: str) -> bool:
        """检测模型是否已下载到本地"""
        try:
            response = requests.get(f"{OLLAMA_HOST}/api/tags")
            if response.status_code != 200:
                return False
            local_models = [m["name"].split(":")[0] for m in response.json()["models"]]
            return model_name in local_models
        except Exception:
            return False

    def _get_download_command(self, model_name: str) -> str:
        """返回模型的下载命令"""
        return f"请在终端执行命令下载模型：\nollama pull {model_name}"

    def get_embedding(self, text: str) -> list[float]:
        """获取文本嵌入向量（懒加载，未下载则提示）"""
        if not self.embedding_exists:
            print(self._get_download_command(self.embedding_model))
            return []
        try:
            resp = ollama.embeddings(model=self.embedding_model, prompt=text.strip())
            return resp["embedding"]
        except Exception as e:
            print(f"获取嵌入失败：{e}")
            return []

    def batch_get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """批量获取嵌入向量"""
        return [self.get_embedding(text) for text in texts]

    def chat(self, prompt: str, chat_history: list[tuple[str, str]] = None) -> str:
        """对话调用（懒加载，未下载则提示）"""
        if not self.llm_exists:
            return self._get_download_command(self.llm_model)
        
        if chat_history is None:
            chat_history = []
        
        messages = []
        for human, ai in chat_history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": ai})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = ollama.chat(model=self.llm_model, messages=messages)
            return resp["message"]["content"]
        except Exception as e:
            return f"模型调用失败：{str(e)}"