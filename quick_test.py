# quick_test.py
from src.vectorstore_manager import VectorStoreManager
from src.rag_pipeline import RAGPipeline
from src.config import Config

def quick_demo():
    """Quick smoke test"""
    vs_manager = VectorStoreManager(
        embedding_model_name=Config.EMBEDDING_MODEL,
        vectorstore_dir=Config.VECTORSTORE_DIR
    )
    
    rag = RAGPipeline(vs_manager, Config.LLM_MODEL)
    
    tests = [
        "What is the APU?",
        "How do I start the APU?",
        "What should I do if hydraulic pressure is low?",
        "Tell me about landing gear"
    ]
    
    print("\n" + "="*80)
    print("AEROMIND QUICK TEST")
    print("="*80 + "\n")
    
    for i, query in enumerate(tests, 1):
        print(f"\n[Test {i}] {query}")
        print("-"*80)
        
        result = rag.query(query)
        
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Confidence: {result.get('confidence', 'unknown')}")
        print(f"Sources: {len(result['sources'])}")
        
        input("\nPress Enter for next test...")

if __name__ == "__main__":
    quick_demo()