# test_multi_manual.py - Test multi-manual handling

from src.vectorstore_manager import VectorStoreManager
from src.rag_pipeline import RAGPipeline
from src.config import Config
import json
from pathlib import Path

def test_multi_manual():
    """Test if system handles multiple manuals correctly"""
    
    # Check how many manuals we have
    chunks_file = Path(Config.PROCESSED_DATA_DIR) / "processed_chunks.json"
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    
    # Get unique sources
    sources = set(chunk['metadata']['source'] for chunk in chunks)
    
    print("\n" + "="*70)
    print("📚 MULTI-MANUAL TEST")
    print("="*70)
    print(f"\nLoaded manuals: {len(sources)}")
    for source in sources:
        source_chunks = [c for c in chunks if c['metadata']['source'] == source]
        print(f"  - {source}: {len(source_chunks)} chunks")
    
    # Initialize RAG
    vs_manager = VectorStoreManager(
        embedding_model_name=Config.EMBEDDING_MODEL,
        vectorstore_dir=Config.VECTORSTORE_DIR
    )
    rag = RAGPipeline(vs_manager, Config.LLM_MODEL)
    
    # Test query that might be in multiple manuals
    print("\n" + "-"*70)
    print("🔍 Testing query: 'How do I inspect for corrosion?'")
    print("-"*70 + "\n")
    
    result = rag.query("How do I inspect for corrosion?")
    
    print("Answer:")
    print(result['answer'])
    
    print("\n📚 Sources referenced:")
    sources_used = set()
    for source in result['sources']:
        sources_used.add(source['source'])
        print(f"  - {source['source']}, Page {source['page']}")
    
    print(f"\n✅ Retrieved from {len(sources_used)} different manual(s)")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_multi_manual()