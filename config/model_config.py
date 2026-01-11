# 支持的模型列表（可自行扩展）
SUPPORTED_LLM_MODELS = {
    "llama3": "Meta Llama3（推荐）",
    "qwen": "阿里云通义千问",
    "phi3": "Microsoft Phi3",
    "gemma": "Google Gemma"
}

SUPPORTED_EMBEDDING_MODELS = {
    "nomic-embed-text": "Nomic Embed Text（推荐）",
    "bge-large": "BGE Large",
    "all-minilm": "All MiniLM"
}

# 默认模型
DEFAULT_LLM_MODEL = "llama3"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"