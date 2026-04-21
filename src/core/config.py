import os
from dotenv import load_dotenv

load_dotenv()

# Настройки ollama
ollama_model = "mistral:latest"
ollama_temperature = 0.1
ollama_top_p = 0.9

# Эмбеддинги через Ollama
ollama_embedding_model = "nomic-embed-text:latest"
#ollama_embedding_model = "all-minilm:latest"

# Пути
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs")
chroma_persist_dir = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db")
log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")

# Параметры разбиения документов
chunk_size = 500
chunk_overlap = 100

# Количество извлекаемых фрагментов для rag
retrieval_k = 4