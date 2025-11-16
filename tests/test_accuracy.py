# tests/test_accuracy.py

import json
from src.vectorstore_manager import VectorStoreManager
from src.rag_pipeline import RAGPipeline
from src.config import Config

class AccuracyTester:
    """Test system accuracy with known Q&A pairs"""
    
    def __init__(self):
        self.vs_manager = VectorStoreManager(
            embedding_model_name=Config.EMBEDDING_MODEL,
            vectorstore_dir=Config.VECTORSTORE_DIR
        )
        self.rag = RAGPipeline(self.vs_manager, Config.LLM_MODEL)
        
        # Load test cases
        self.test_cases = self.load_test_cases()
    
    def load_test_cases(self):
        """Load predefined test Q&A pairs"""
        # You'll create this file with known answers
        try:
            with open("tests/test_cases.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.get_default_test_cases()
    
    def get_default_test_cases(self):
        """Default test cases"""
        return [
            {
                "question": "What is the APU?",
                "expected_topics": ["auxiliary power unit", "power", "ground operations"],
                "should_have_citation": True
            },
            {
                "question": "How do I start the APU?",
                "expected_topics": ["procedure", "steps", "start"],
                "should_have_citation": True,
                "is_procedure": True
            },
            # Add more...
        ]
    
    def test_retrieval_accuracy(self):
        """Test if correct documents are retrieved"""
        results = []
        
        for test_case in self.test_cases:
            result = self.rag.query(test_case["question"])
            
            # Check if expected topics appear in answer
            answer_lower = result["answer"].lower()
            topics_found = [
                topic for topic in test_case["expected_topics"]
                if topic.lower() in answer_lower
            ]
            
            # Check citations
            has_citations = len(result["sources"]) > 0
            
            test_result = {
                "question": test_case["question"],
                "passed": len(topics_found) > 0,
                "topics_found": topics_found,
                "topics_expected": test_case["expected_topics"],
                "has_citations": has_citations,
                "expected_citations": test_case["should_have_citation"],
                "confidence": result.get("confidence", "unknown")
            }
            
            results.append(test_result)
        
        # Calculate accuracy
        passed = sum(1 for r in results if r["passed"])
        accuracy = (passed / len(results)) * 100 if results else 0
        
        print(f"\n{'='*60}")
        print(f"ACCURACY TEST RESULTS")
        print(f"{'='*60}")
        print(f"Total tests: {len(results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {len(results) - passed}")
        print(f"Accuracy: {accuracy:.1f}%")
        print(f"{'='*60}\n")
        
        return results, accuracy

if __name__ == "__main__":
    tester = AccuracyTester()
    results, accuracy = tester.test_retrieval_accuracy()