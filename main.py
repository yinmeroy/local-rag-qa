import os
# 先验证Ollama是否启动
try:
    import ollama
    from config.settings import OLLAMA_HOST, LLM_MODEL
    os.environ["OLLAMA_HOST"] = OLLAMA_HOST
    ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": "test"}])
    print("✅ Ollama连接正常")
except Exception as e:
    print(f"❌ Ollama未启动或连接失败：{e}")
    exit(1)

# 启动Gradio界面
from ui.gradio_ui import gradio_ui

if __name__ == "__main__":
    gradio_ui.run()