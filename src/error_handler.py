# src/error_handler.py

class AeroMindError(Exception):
    """Base exception for AeroMind"""
    pass

class ManualNotFoundError(AeroMindError):
    """No manuals loaded"""
    pass

class NoRelevantDocumentsError(AeroMindError):
    """No relevant documents for query"""
    pass

def handle_query_errors(func):
    """Decorator for error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return {
                "answer": f"⚠️ Error processing query: {str(e)}. Please try rephrasing your question or contact support.",
                "sources": [],
                "confidence": "error",
                "error": str(e)
            }
    return wrapper