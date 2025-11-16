# src/rag_pipeline_multimodal.py
from langchain_community.llms import Ollama
from src.vectorstore_manager import VectorStoreManager
from typing import Dict, List
import base64
import json
from pathlib import Path

class MultimodalRAGPipeline:
    """RAG Pipeline that handles BOTH text and image queries"""
    
    def __init__(self, vectorstore_manager: VectorStoreManager, 
                 text_llm_model: str = "llama3.2",
                 vision_llm_model: str = "llava"):
        self.vs_manager = vectorstore_manager
        self.text_llm = Ollama(model=text_llm_model, temperature=0)
        self.vision_llm = Ollama(model=vision_llm_model, temperature=0)
        
        # Load image catalog
        self.images_catalog = self._load_images_catalog()
        
        self.system_prompt = """You are an aircraft maintenance assistant with strict guidelines:

CRITICAL RULES:
1. ONLY answer based on the provided context - never use outside knowledge
2. If the context doesn't contain the answer, say: "I cannot find this information in the available manuals."
3. ALWAYS cite: [Source: {manual name}, Page: {page}]
4. When diagrams are relevant, reference them: [See Diagram: Figure X on Page Y]
5. Use aviation terminology correctly
6. For procedures, use numbered steps
7. End EVERY answer with: "⚠️ Verify with certified manual before performing maintenance"
"""
    
    def _load_images_catalog(self) -> Dict:
        """Load the images catalog"""
        try:
            catalog_path = Path("data/processed/images_catalog.json")
            if catalog_path.exists():
                with open(catalog_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Could not load images catalog: {e}")
            return {}
    
    def _detect_diagram_request(self, question: str) -> bool:
        """Detect if user is asking about diagrams/images"""
        diagram_keywords = [
            "diagram", "figure", "illustration", "picture", "image",
            "drawing", "schematic", "show me", "what does it look like",
            "visual", "chart", "graph", "layout", "location"
        ]
        return any(keyword in question.lower() for keyword in diagram_keywords)
    
    def _find_relevant_images(self, docs: List, question: str) -> List[Dict]:
        """Find images related to retrieved documents"""
        relevant_images = []
        
        for doc in docs:
            page = doc.metadata.get("page")
            source = doc.metadata.get("source")
            
            # Find images from the same page/source
            for img_data in self.images_catalog:
                if (img_data.get("page") == page and 
                    img_data.get("source") == source):
                    relevant_images.append(img_data)
        
        return relevant_images
    
    def _analyze_image_with_vision_model(self, image_path: str, question: str) -> str:
        """Use vision model to analyze an image"""
        try:
            # Read image and convert to base64
            with open(image_path, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Create vision prompt
            vision_prompt = f"""You are analyzing an aircraft maintenance manual diagram.

Question: {question}

Describe what you see in this diagram that relates to the question. Focus on:
- Components labeled in the diagram
- Any arrows, connections, or flow indicators
- Numbers or reference marks
- Safety warnings or important callouts

Keep your description technical and precise."""

            # Call vision model (llava format)
            response = self.vision_llm.invoke(
                vision_prompt,
                images=[image_data]
            )
            
            return response
            
        except Exception as e:
            return f"Could not analyze image: {str(e)}"
    
    def query(self, question: str, return_sources: bool = True, 
              include_images: bool = True) -> dict:
        """Enhanced query with image support"""
        
        # Load vectorstore if needed
        if not self.vs_manager.vectorstore:
            self.vs_manager.load_vectorstore()
        
        # Retrieve documents
        docs = self.vs_manager.vectorstore.similarity_search(question, k=5)
        
        if not docs:
            return {
                "answer": "⚠️ I cannot find relevant information in the loaded manuals.",
                "sources": [],
                "images": [],
                "confidence": "none"
            }
        
        # Check if user wants diagram information
        wants_diagrams = self._detect_diagram_request(question)
        
        # Build text context
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            context_parts.append(
                f"[Source {i}: {source}, Page {page}]\n{doc.page_content}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Find relevant images
        relevant_images = []
        image_analyses = []
        
        if include_images and (wants_diagrams or any(
            doc.metadata.get("has_diagram") for doc in docs
        )):
            relevant_images = self._find_relevant_images(docs, question)
            
            # Analyze images with vision model if available
            for img in relevant_images[:3]:  # Limit to 3 images to save time
                img_path = img.get("path")
                if img_path and Path(img_path).exists():
                    analysis = self._analyze_image_with_vision_model(img_path, question)
                    image_analyses.append({
                        "filename": img.get("filename"),
                        "page": img.get("page"),
                        "analysis": analysis
                    })
        
        # Add image analyses to context if available
        if image_analyses:
            context += "\n\n--- DIAGRAM ANALYSIS ---\n\n"
            for idx, img_analysis in enumerate(image_analyses, 1):
                context += f"[Diagram {idx}: {img_analysis['filename']}, Page {img_analysis['page']}]\n"
                context += f"{img_analysis['analysis']}\n\n"
        
        # Create full prompt
        full_prompt = f"""{self.system_prompt}

Context from manuals:
{context}

Question: {question}

Answer with citations (include diagram references if relevant):"""
        
        # Get answer
        answer = self.text_llm.invoke(full_prompt)
        
        # Format sources
        sources = []
        if return_sources:
            for doc in docs:
                sources.append({
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                })
        
        # Prepare image results
        image_results = []
        for img in relevant_images:
            image_results.append({
                "filename": img.get("filename"),
                "path": img.get("path"),
                "page": img.get("page"),
                "source": img.get("source"),
                "width": img.get("width"),
                "height": img.get("height")
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "images": image_results,
            "image_analyses": image_analyses,
            "confidence": "high" if (sources and not wants_diagrams) or (sources and image_results) else "medium",
            "has_diagrams": len(image_results) > 0
        }


# Simple text-only fallback version (if vision model not available)
class SimpleImageRAGPipeline:
    """Simpler version that just retrieves and displays images without analysis"""
    
    def __init__(self, vectorstore_manager: VectorStoreManager, llm_model: str):
        self.vs_manager = vectorstore_manager
        self.text_llm = Ollama(model=llm_model, temperature=0)
        self.images_catalog = self._load_images_catalog()
    
    def _load_images_catalog(self):
        try:
            with open("data/processed/images_catalog.json", 'r') as f:
                return json.load(f)
        except:
            return []
    
    def query(self, question: str) -> dict:
        """Query with image retrieval (but no vision analysis)"""
        if not self.vs_manager.vectorstore:
            self.vs_manager.load_vectorstore()
        
        docs = self.vs_manager.vectorstore.similarity_search(question, k=5)
        
        if not docs:
            return {
                "answer": "No relevant information found.",
                "sources": [],
                "images": []
            }
        
        # Find related images
        relevant_images = []
        for doc in docs:
            page = doc.metadata.get("page")
            source = doc.metadata.get("source")
            
            for img in self.images_catalog:
                if img.get("page") == page and img.get("source") == source:
                    relevant_images.append(img)
        
        # Build context
        context = "\n\n".join([
            f"[{doc.metadata.get('source')}, Page {doc.metadata.get('page')}]\n{doc.page_content}"
            for doc in docs
        ])
        
        # Add note about diagrams if found
        if relevant_images:
            context += f"\n\n[NOTE: {len(relevant_images)} diagram(s) found on these pages]"
        
        prompt = f"""Context: {context}

Question: {question}

Answer with citations:"""
        
        answer = self.text_llm.invoke(prompt)
        
        return {
            "answer": answer,
            "sources": [{"source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in docs],
            "images": relevant_images[:5]  # Limit to 5 images
        }