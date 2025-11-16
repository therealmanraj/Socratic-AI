# src/rag_pipeline.py
from langchain_community.llms import Ollama
from src.vectorstore_manager import VectorStoreManager
from typing import Dict

class RAGPipeline:
    def __init__(self, vectorstore_manager: VectorStoreManager, llm_model: str):
        self.vs_manager = vectorstore_manager
        self.llm = Ollama(model=llm_model, temperature=0)
        
        # IMPROVED PROMPT with better instructions
        self.system_prompt = """You are an aircraft maintenance assistant with strict guidelines:

CRITICAL RULES:
1. ONLY answer based on the provided context - never use outside knowledge
2. If the context doesn't contain the answer, say: "I cannot find this information in the available manuals. Please consult a certified technician or the complete manual."
3. ALWAYS cite: [Source: {manual name}, Page: {page}]
4. Use aviation terminology correctly (APU, MEL, ETOPS, etc.)
5. For procedures, use numbered steps
6. For safety-critical items, emphasize warnings
7. End EVERY answer with: "⚠️ Verify with certified manual before performing maintenance"

Context quality indicators:
- If context seems incomplete or unclear, say so
- If multiple sources contradict, mention both
- If context is about a different aircraft type, note this
"""

    def _check_answer_quality(self, answer: str, sources: list) -> dict:
        """Check if answer is high quality"""
        quality_issues = []
        
        # Check 1: Does it have citations?
        if "Page:" not in answer and "Source:" not in answer:
            quality_issues.append("Missing citations")
        
        # Check 2: Is it too short? (might be hallucinating)
        if len(answer) < 50:
            quality_issues.append("Answer too brief")
        
        # Check 3: Did it find relevant sources?
        if len(sources) == 0:
            quality_issues.append("No relevant sources found")
        
        # Check 4: Does it say "I don't know"?
        uncertainty_phrases = [
            "cannot find",
            "not available",
            "unclear",
            "insufficient information"
        ]
        is_uncertain = any(phrase in answer.lower() for phrase in uncertainty_phrases)
        
        return {
            "has_issues": len(quality_issues) > 0,
            "issues": quality_issues,
            "is_uncertain": is_uncertain,
            "confidence": "low" if quality_issues or is_uncertain else "high"
        }
    
    def query(self, question: str, return_sources: bool = True) -> dict:
        """Enhanced query with quality checking"""
        
        # Load vectorstore
        if not self.vs_manager.vectorstore:
            self.vs_manager.load_vectorstore()
        
        # Retrieve documents
        docs = self.vs_manager.vectorstore.similarity_search(question, k=5)
        
        # Check if we have relevant documents
        if not docs or len(docs) == 0:
            return {
                "answer": "⚠️ I cannot find relevant information in the loaded manuals. Please verify the question or check if the appropriate manual is loaded.",
                "sources": [],
                "confidence": "none",
                "quality_check": {"has_issues": True, "issues": ["No documents found"]}
            }
        
        # Build context
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            context_parts.append(
                f"[Source {i}: {source}, Page {page}]\n{doc.page_content}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Create prompt
        full_prompt = f"""{self.system_prompt}

Context from manuals:
{context}

Question: {question}

Answer with citations:"""
        
        # Get answer
        answer = self.llm.invoke(full_prompt)
        
        # Format sources
        sources = []
        if return_sources:
            for doc in docs:
                sources.append({
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                    "relevance_score": None  # Could add similarity scores later
                })
        
        # Quality check
        quality_check = self._check_answer_quality(answer, sources)
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": quality_check["confidence"],
            "quality_check": quality_check
        }