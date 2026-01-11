import gradio as gr
import os
import shutil
from config.settings import (
    GRADIO_HOST, GRADIO_PORT, GRADIO_SHARE,
    SUPPORTED_LLM_MODELS, SUPPORTED_EMBEDDING_MODELS,
    DEFAULT_LLM_MODEL, DEFAULT_EMBEDDING_MODEL
)
from document.doc_processor import doc_processor
from vector_db.faiss_manager import faiss_manager
from models.ollama_client import OllamaClient
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

os.environ["GRADIO_API_ENABLED"] = "False"

class GradioUI:
    """封装Gradio界面（新增模型状态校验）"""
    def __init__(self):
        self.faiss_manager = faiss_manager
        self.doc_processor = doc_processor
        self.qa_chain = None
        self.ollama_client = None  # 动态初始化Ollama客户端
        # 新增：记录模型是否可用
        self.model_available = False

    def init_ollama_client(self, llm_model, embedding_model):
        """初始化Ollama客户端（仅检测，不下载）"""
        try:
            self.ollama_client = OllamaClient(llm_model, embedding_model)
            # 新增：校验模型是否都已下载
            self.model_available = self.ollama_client.llm_exists and self.ollama_client.embedding_exists
            
            # 拼接检测结果提示
            llm_status = "✅ 已下载" if self.ollama_client.llm_exists else "❌ 未下载"
            embed_status = "✅ 已下载" if self.ollama_client.embedding_exists else "❌ 未下载"
            
            if self.model_available:
                return f"模型检测完成：\nLLM({llm_model}) {llm_status}\n嵌入({embedding_model}) {embed_status}\n✅ 所有模型已就绪，可上传文件"
            else:
                download_cmds = []
                if not self.ollama_client.llm_exists:
                    download_cmds.append(f"ollama pull {llm_model}")
                if not self.ollama_client.embedding_exists:
                    download_cmds.append(f"ollama pull {embedding_model}")
                return f"模型检测完成：\nLLM({llm_model}) {llm_status}\n嵌入({embedding_model}) {embed_status}\n❌ 请先下载模型：\n{chr(10).join(download_cmds)}"
        except Exception as e:
            self.model_available = False
            return f"模型检测失败：{str(e)}"

    def upload_file(self, file_obj, chatbot):
        """上传文件：新增模型状态校验"""
        # 新增：优先校验模型是否可用
        if not self.ollama_client:
            return "请先选择并初始化模型！", chatbot
        if not self.model_available:
            return "❌ 所选模型未全部下载，无法上传文件！请先下载模型后重新初始化", chatbot
        
        # 1. 自动清空对话+清除向量缓存
        chatbot = []
        self.clear_vector_db()

        if not file_obj:
            return "请选择文件！", chatbot
        
        try:
            file_path = file_obj.name
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return "文件内容为空，请上传非空文件！", chatbot
            
            # 2. 正确初始化向量库的Embeddings实例
            from vector_db.faiss_manager import CustomEmbeddings
            self.faiss_manager.embeddings = CustomEmbeddings(self.ollama_client.get_embedding)
            
            # 3. 隐含构建向量库
            split_docs = self.doc_processor.process_file(file_path)
            self.faiss_manager.add_documents(split_docs)
            self._create_qa_chain()
            
            return "上传成功，可开始提问", chatbot
        except Exception as e:
            return f"上传失败：{str(e)}", chatbot

    def _create_qa_chain(self):
        """创建检索增强问答链"""
        ctx_prompt = ChatPromptTemplate.from_messages([
            ("system", "根据对话历史和当前问题，生成独立的检索问题，仅返回问题本身"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        def llm_wrapper(prompt_value):
            messages = prompt_value.to_messages()
            current_input = messages[-1].content
            chat_history = []
            for i in range(0, len(messages)-1, 2):
                if i+1 < len(messages):
                    chat_history.append((messages[i].content, messages[i+1].content))
            return self.ollama_client.chat(current_input, chat_history)

        retriever = self.faiss_manager.db.as_retriever(search_kwargs={"k": 3})
        history_retriever = create_history_aware_retriever(llm_wrapper, retriever, ctx_prompt)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "仅根据上下文回答问题，要求：1. 回复统一使用中文；2. 修正错别字和表述错误；3. 内容完整通顺无截断。无信息则说'未找到相关答案'。上下文：{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        qa_chain = create_stuff_documents_chain(llm_wrapper, qa_prompt)
        self.qa_chain = create_retrieval_chain(history_retriever, qa_chain)

    def chat(self, msg, chat_history):
        """处理对话：新增模型状态校验"""
        # 新增：校验模型是否可用
        if not self.ollama_client:
            chat_history.append((msg, "❌ 请先选择并初始化模型！"))
            return "", chat_history
        if not self.model_available:
            chat_history.append((msg, "❌ 所选模型未全部下载，无法问答！请先下载模型后重新初始化"))
            return "", chat_history
        
        if not self.qa_chain:
            chat_history.append((msg, "❌ 请先上传文件！"))
            return "", chat_history

        try:
            lc_history = []
            for h, a in chat_history:
                lc_history.append(("human", h))
                lc_history.append(("ai", a))

            resp = self.qa_chain.invoke({"input": msg, "chat_history": lc_history})
            answer = resp["answer"]

            chat_history.append((msg, answer))
            return "", chat_history
        except Exception as e:
            chat_history.append((msg, f"问答失败：{str(e)}"))
            return "", chat_history

    def clear_vector_db(self):
        """清空向量库"""
        if os.path.exists(self.faiss_manager.index_path):
            shutil.rmtree(self.faiss_manager.index_path)
        self.faiss_manager.db = None
        self.qa_chain = None
        return "文件缓存已删除"

    def create_ui(self):
        """整合所有功能到同一界面"""
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 本地知识库问答工具（多模型版）")

            # 第一行：模型选择器
            with gr.Row():
                llm_model = gr.Dropdown(
                    label="大语言模型",
                    choices=list(SUPPORTED_LLM_MODELS.keys()),
                    value=DEFAULT_LLM_MODEL,
                    info=SUPPORTED_LLM_MODELS[DEFAULT_LLM_MODEL]
                )
                embedding_model = gr.Dropdown(
                    label="嵌入模型",
                    choices=list(SUPPORTED_EMBEDDING_MODELS.keys()),
                    value=DEFAULT_EMBEDDING_MODEL,
                    info=SUPPORTED_EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL]
                )
                init_model_btn = gr.Button("初始化模型", variant="secondary")
                model_status = gr.Textbox(label="模型状态", interactive=False)

            # 第二行：文件上传
            with gr.Row():
                file = gr.File(label="上传PDF/TXT", file_types=[".pdf", ".txt"])
                upload_btn = gr.Button("上传文件", variant="primary")
                upload_status = gr.Textbox(label="上传状态", interactive=False)

            # 第三行：问答区域
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(placeholder="输入问题...")

            # 第四行：缓存管理
            clear_vdb_btn = gr.Button("删除文件缓存")

            # 绑定事件
            # 1. 模型初始化
            init_model_btn.click(
                self.init_ollama_client,
                inputs=[llm_model, embedding_model],
                outputs=[model_status]
            )
            # 2. 文件上传
            upload_btn.click(
                self.upload_file,
                inputs=[file, chatbot],
                outputs=[upload_status, chatbot]
            )
            # 3. 问答
            msg.submit(self.chat, inputs=[msg, chatbot], outputs=[msg, chatbot])
            # 4. 清除缓存
            clear_vdb_btn.click(self.clear_vector_db, outputs=[upload_status])

        return demo

    def run(self):
        """启动界面"""
        demo = self.create_ui()
        demo.launch(
            server_name=GRADIO_HOST,
            server_port=GRADIO_PORT,
            share=GRADIO_SHARE,
            show_api=False
        )

# 单例实例
gradio_ui = GradioUI()