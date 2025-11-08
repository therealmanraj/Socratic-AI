# src/rag_pipeline.py
from langchain_community.llms import Ollama
from src.vectorstore_manager import VectorStoreManager
from typing import Dict

class RAGPipeline:
    """Main RAG pipeline for query answering - simplified version"""
    
    def __init__(self, vectorstore_manager: VectorStoreManager, llm_model: str):
        self.vs_manager = vectorstore_manager
        self.llm = Ollama(model=llm_model, temperature=0)
        
        self.system_prompt = """You are an aircraft maintenance assistant. Use the following context from maintenance manuals to answer the question.

CRITICAL RULES:
1. ALWAYS cite the specific source and page number from the context
2. If you're not certain, say so - never make up information
3. Format your answer clearly with procedure steps if applicable
4. End with: "⚠️ Verify with certified manual before performing maintenance"
"""
    
    def create_prompt(self, context: str, question: str) -> str:
        """Create the full prompt with context"""
        return f"""{self.system_prompt}

Context from manuals:
{context}

Question: {question}

Answer with citations:"""
    
    def query(self, question: str, return_sources: bool = True) -> Dict:
        """
        Query the RAG system
        """
        # Ensure vectorstore is loaded
        if not self.vs_manager.vectorstore:
            self.vs_manager.load_vectorstore()
        
        # Retrieve relevant documents
        docs = self.vs_manager.vectorstore.similarity_search(question, k=5)
        
        # Format context with sources
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            context_parts.append(f"[Source {i}: {source}, Page {page}]\n{doc.page_content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Create full prompt
        full_prompt = self.create_prompt(context, question)
        
        # Get answer from LLM
        answer = self.llm.invoke(full_prompt)
        
        # Format response
        result = {
            "answer": answer,
            "sources": []
        }
        
        if return_sources:
            for doc in docs:
                result["sources"].append({
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A")
                })
        
        return result


# Usage example
if __name__ == "__main__":
    from src.config import Config
    
    vs_manager = VectorStoreManager(
        embedding_model_name=Config.EMBEDDING_MODEL,
        vectorstore_dir=Config.VECTORSTORE_DIR
    )
    
    rag = RAGPipeline(vs_manager, llm_model=Config.LLM_MODEL)
    
    result = rag.query("How do I start the APU?")
    print(f"Answer: {result['answer']}\n")
    print("\nSources:")
    for i, source in enumerate(result['sources'], 1):
        print(f"{i}. {source['source']}, Page {source['page']}")