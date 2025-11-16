# src/vectorstore_manager.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pathlib import Path
import json
from typing import List, Dict

class VectorStoreManager:
    """Manages embeddings and FAISS vector store"""
    
    def __init__(self, embedding_model_name: str, vectorstore_dir: str):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore_dir = Path(vectorstore_dir)
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore = None
    
    def create_vectorstore(self, chunks: List[Dict]):
        """
        Create FAISS vectorstore from chunks
        """
        # Convert chunks to LangChain Documents
        documents = [
            Document(
                page_content=chunk["text"],
                metadata=chunk["metadata"]
            )
            for chunk in chunks
        ]
        
        print(f"Creating embeddings for {len(documents)} documents...")
        self.vectorstore = FAISS.from_documents(
            documents,
            self.embedding_model
        )
        
        # Save vectorstore
        self.save_vectorstore()
        print("Vector store created and saved!")
    
    def save_vectorstore(self):
        """Save FAISS index to disk"""
        if self.vectorstore:
            self.vectorstore.save_local(str(self.vectorstore_dir))
    
    def load_vectorstore(self):
        """Load existing FAISS index"""
        try:
            self.vectorstore = FAISS.load_local(
                str(self.vectorstore_dir),
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            print("Vector store loaded successfully!")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return None
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for similar documents
        """
        if not self.vectorstore:
            self.load_vectorstore()
        
        results = self.vectorstore.similarity_search(query, k=k)
        return results


# Usage example
if __name__ == "__main__":
    # Load processed chunks
    with open("data/processed/processed_chunks_with_images.json", 'r') as f:
        chunks = json.load(f)
    
    # Create vector store
    vs_manager = VectorStoreManager(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        vectorstore_dir="vectorstore"
    )
    vs_manager.create_vectorstore(chunks)