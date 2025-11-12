# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Paths
    RAW_DATA_DIR = "data/raw"
    PROCESSED_DATA_DIR = "data/processed"
    VECTORSTORE_DIR = "vectorstore"
    
    # Model settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Free, good
    LLM_MODEL = "llama3.2"  # Ollama model
    
    # RAG settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RESULTS = 5
    
    # OpenAI (if using)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")