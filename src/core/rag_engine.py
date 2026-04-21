import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from .config import (
    data_path, chroma_persist_dir, ollama_embedding_model,
    chunk_size, chunk_overlap, retrieval_k
)

class RagEngine:
    def __init__(self):
        # Используем эмбеддинги Ollama
        self.embeddings = OllamaEmbeddings(model=ollama_embedding_model)
        self.vectorstore = None

    def index_documents(self):
        """Загружает все pdf и txt из папки data_path и создаёт индекс."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Папка с документами не найдена: {data_path}")

        documents = []
        for file in os.listdir(data_path):
            file_path = os.path.join(data_path, file)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                continue
            print(f"Загружается: {file}")
            documents.extend(loader.load())

        if not documents:
            raise ValueError("Не найдено ни одного pdf или txt файла.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)

        self.vectorstore = Chroma.from_documents(
            chunks,
            self.embeddings,
            persist_directory=chroma_persist_dir
        )
        print(f"Индексация завершена. Всего фрагментов: {len(chunks)}")

    def retrieve(self, query: str, k: int = retrieval_k):
        """Возвращает k релевантных фрагментов по запросу."""
        if self.vectorstore is None:
            if os.path.exists(chroma_persist_dir):
                self.vectorstore = Chroma(
                    persist_directory=chroma_persist_dir,
                    embedding_function=self.embeddings
                )
            else:
                raise RuntimeError("Векторное хранилище не найдено. Сначала выполните index_documents().")
        return self.vectorstore.similarity_search(query, k=k)