# Rag Pipeline - Data Ingestion To Vector DB Pipeline
import os
import uuid
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

load_dotenv()

model = init_chat_model("anthropic:claude-sonnet-4.5")


#Read all the Pdfs inside the directory
def process_all_pdfs(pdf_directory):
    """Process all Pdfs in a directory"""
    all_documents = []
    pdf_dir = Path(pdf_directory)

    #Find all Pdfs recursively
    pdf_files = list(pdf_dir.glob("**/*.pdf"))

    print(f"Found {len(pdf_files)} Pdf files to process")

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")

        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()

            #Add Source Information to metadata
            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"

            all_documents.extend(documents)
            print(f"Loaded {len(documents)} Pages")

        except Exception as e:
            print(f" Error: {e}")
            raise

    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents


all_pdf_documents = process_all_pdfs("../data/pdf-files")
print(all_pdf_documents)


#Text Splitting into chunks
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks for better RAG performance"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    #Show example of a chunk
    if split_docs:
        print(f"\nExample chunk: {split_docs[0]}")
        print(f"\nContent: {split_docs[0].page_content[:200]}...")
        print(f"\nMetadata: {split_docs[0].metadata}")

    return split_docs


chunks = split_documents(all_pdf_documents)
print(chunks)


#Embedding the chunks
class EmbeddingManager:
    """Handles document embedding using sentence transformer"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager

        Args:
            model_name: HuggingFace model name for sentence embedding
        """

        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            print(f"Loading Embedding Model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model Loaded Successfully. Embedding Dimension: {self.model.get_embedding_dimension()}")
        except Exception as e:
            print(f"Error Loading Model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of strings to embed

        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """

        if not self.model:
            raise ValueError("Model not loaded")

        print(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape {embeddings.shape}")
        return embeddings


#Initialize the Embedding Manager
embedding_manager = EmbeddingManager()
print(embedding_manager)


#Vector Store
class VectorStore:
    """Manages document embeddings in chromadb vector store"""

    def __init__(self, collection_name: str = "pdf-documents", persist_directory: str = "../data/vector_store"):
        """
        Initialize the vector store

        Args:
            collection_name: Name of the chromadb collection
            persist_directory: Directory to persist the vector store
        """

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "PDF document embeddings for RAG"}
        )
        self._initialize_store()

    def _initialize_store(self):
        """Initialize chromadb client and collection"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            #Get or create collection
            self.collection = self.client.get_or_create_collection(
                name = self.collection_name,
                metadata = {"description": "PDF document embeddings for RAG"}
            )
            print(f"Vector store initiallized, Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error Initializing vector store: {e}")
            raise

    def add_documents(self, documents: list[Any], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store

        Args:
            documents: List of langchain documents
            embeddings: Corresponding embeddings for the documents
        """

        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        print(f"Adding {len(documents)} documents to vector store...")

        #Prepare data for chromadb
        ids = []
        metadatas = []
        documents_texts = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            #Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            #Prepare metadata
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            #Document content
            documents_texts.append(doc.page_content)

            #Embeddings
            embeddings_list.append(embedding.tolist())

        #Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_texts
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise


vector_store = VectorStore()
print(vector_store)



#Convert Texts to Embeddings
texts = [doc.page_content for doc in chunks]


#Generate the Embeddings
embeddings = embedding_manager.generate_embeddings(texts)


#Store inVector database
vector_store.add_documents(chunks, embeddings)



#RAG Retriever
class RAGRetriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initialize the retriever

        Args:
            vector_store: Vector store containing document embeddings
            embedding_manager: Manager for generating query embeddings
        """

        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> list[dict[str, Any]]:
        """
        Retrieve relevant documents for a query

        Args:
            query: Search the query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, score threshold: {score_threshold}")

        #Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        #Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            #Process results
            retrieved_docs = []

            if (
                results["documents"]
                and results["documents"][0]
                and results["metadatas"]
                and results["metadatas"][0]
                and results["distances"]
                and results["distances"][0]
                and results["ids"]
                and results["ids"][0]
            ):
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    #Convert Distance to similarity score
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            "id": doc_id,
                            "content": document,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            "distance": distance,
                            "rank": i + 1
                        })

                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")

            else:
                print("No documents found")

            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
            raise


rag_retriever = RAGRetriever(vector_store, embedding_manager)
print(rag_retriever)


rag_retriever.retrieve("What is Ray's surnname")


#Enhanced RAG function: Retrieve context + Generate response
def rag_advanced(query, retriever, model, top_k=5, min_score=0.2, return_context=False):
    """
    RAG Pipeline with extra features:
    - Returns answer, sources, confidence score and optionally full context.
    """

    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)

    if not results:
        return {"answer": "No relevant context found.", "sources": [], "confidence": 0.0, "context": ""}

    #Prepare context and sources
    context = "\n\n".join([doc["content"] for doc in results])
    sources = [{
        "source": doc["metadata"].get("source_file", doc["metadata"].get("source", "unknown")),
        "page": doc["metadata"].get("page", "unknown"),
        "score": doc["similarity_score"],
        "preview": doc["content"][:300] + "..."
    } for doc in results]
    confidence = max(doc["similarity_score"] for doc in results)

    #Generate answer
    prompt = f"Use the following context to answer the question concisely.\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = model.invoke([prompt.format(context=context, query=query)])

    output = {
        "answer": response.content,
        "sources": sources,
        "confidence": confidence
    }

    if return_context:
        output["context"] = context

    return output


#Example Usage
result = rag_advanced("What is Ray's Phone number?", rag_retriever, model, top_k=3, min_score=0.1, return_context=True)
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
print(f"confidence: {result['confidence']}")
print(f"Context Preview: {result['context'][:300]}")
