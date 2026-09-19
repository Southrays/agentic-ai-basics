
from src.search import RAGSearch
from src.vectorstore import FaissVectorStore

#Example Usage for storing
# if __name__ == "__main__":
#     docs = load_all_documents("rag/data")
#     store = FaissVectorStore("faiss_store")
#     store.build_from_documents(docs)


#Example Usage for loading after your have stored
# if __name__ == "__main__":
#     store = FaissVectorStore("faiss_store")
#     store.load()
#     print(store.query("What is Ray's email", top_k=3))


#Example Usage of Rag search after you have stored
if __name__ == "__main__":
    store = FaissVectorStore("faiss_store")
    store.load()

    rag_search = RAGSearch()
    query = "What are Ray's experiences?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)

    