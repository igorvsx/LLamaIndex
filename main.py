from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
import chromadb
import os
import requests
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import List
from pathlib import Path
import time


class OptimizedIndexer:
    def __init__(
            self,
            ollama_model: str = "qwen2.5:7b",
            docs_path: str = "docs",
            index_path: str = "storage",
            batch_size: int = 32
    ):
        self.ollama_model = ollama_model
        self.docs_path = Path(docs_path)
        self.index_path = Path(index_path)
        self.batch_size = batch_size

        # Отключаем лишние логи для ускорения
        logging.getLogger("chromadb").setLevel(logging.ERROR)

        # Проверяем доступность Ollama перед началом работы
        self._check_ollama_availability()
        self._initialize_models()
        self._setup_chroma()

    def _check_ollama_availability(self, max_retries=3):
        """Проверка доступности Ollama сервера"""
        for i in range(max_retries):
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=10)
                if response.status_code == 200:
                    print("✅ Ollama сервер доступен")
                    return
            except requests.exceptions.RequestException:
                if i < max_retries - 1:
                    print(f"⚠️ Попытка подключения к Ollama {i + 1}/{max_retries}")
                    time.sleep(2)
                continue
        raise ConnectionError("❌ Не удалось подключиться к Ollama серверу")

    def _initialize_models(self):
        """Инициализация моделей с оптимальными настройками"""
        self.llm = Ollama(
            model=self.ollama_model,
            temperature=0.2,
            request_timeout=120.0,
            additional_kwargs={
                "num_thread": 8,
                "num_ctx": 4096,
                "timeout": 120
            }
        )

        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            embed_batch_size=32
        )

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

    def _setup_chroma(self):
        """Настройка ChromaDB с оптимизированными параметрами"""
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.index_path),
            settings=chromadb.Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.collection = self.chroma_client.get_or_create_collection(
            "default",
            metadata={"hnsw:space": "cosine"}
        )

        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)

    def _batch_process_documents(self, documents: List) -> List:
        """Обработка документов батчами"""
        processed_docs = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            processed_docs.extend(batch)
        return processed_docs

    def create_or_load_index(self):
        """Создание или загрузка индекса с оптимизациями"""
        if self.collection.count() > 0:
            print("🔄 Loading existing index...")
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            return load_index_from_storage(storage_context)

        print("📄 Indexing documents...")
        documents = SimpleDirectoryReader(str(self.docs_path)).load_data()

        processed_docs = self._batch_process_documents(documents)

        index = VectorStoreIndex.from_documents(
            processed_docs,
            vector_store=self.vector_store,
            show_progress=True
        )

        index.storage_context.persist(persist_dir=str(self.index_path))
        return index

    def query(self, question: str, similarity_top_k: int = 3, max_retries: int = 3):
        """Оптимизированный запрос к индексу с повторными попытками"""
        index = self.create_or_load_index()
        query_engine = index.as_query_engine(
            similarity_top_k=similarity_top_k
        )

        for attempt in range(max_retries):
            try:
                return query_engine.query(question)
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Попытка {attempt + 1}/{max_retries} не удалась: {str(e)}")
                    time.sleep(2)
                    continue
                raise


if __name__ == "__main__":
    indexer = OptimizedIndexer()
    question = "Кто авторы документов?"
    response = indexer.query(question)

    print("❓ Вопрос:", question)
    print("✅ Ответ:", response)