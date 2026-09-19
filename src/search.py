import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from src.vectorstore import FaissVectorStore

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "anthropic:claude-sonnet-4.5"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        #Load or Build Vector Store
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("../rag/data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        self.llm = init_chat_model(llm_model)
        print(f"[INFO] Claude LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])

        if isinstance(response.content, str):
            return response.content
        
        return str(response.content)


# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What are Ray's experiences?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)