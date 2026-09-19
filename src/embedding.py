from typing import Any

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: list[Any]) -> list[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_documents(self, documents: list[Any]) -> np.ndarray:
        chunks = self.chunk_documents(documents)
        embeddings = self.model.encode({chunk.page_content for chunk in chunks}, show_progress_bar=True)
        print(f"[INFO] Created embeddings for {len(embeddings)} chunks")
        return embeddings

    def embed_chunks(self, chunks: list[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(chunks)} chunks.")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings