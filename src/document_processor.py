# src/document_processor.py
import pymupdf  # PyMuPDF
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
import json

class DocumentProcessor:
    """Handles PDF ingestion and chunking with metadata preservation"""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract text from PDF with page-level metadata
        Returns list of dicts with text and metadata
        """
        doc = pymupdf.open(pdf_path)
        pages = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            pages.append({
                "text": text,
                "page": page_num + 1,
                "source": Path(pdf_path).name
            })
        
        doc.close()
        return pages
    
    def create_chunks(self, pages: List[Dict]) -> List[Dict]:
        """
        Split pages into chunks while preserving metadata
        """
        chunks = []
        
        for page in pages:
            text_chunks = self.text_splitter.split_text(page["text"])
            
            for i, chunk_text in enumerate(text_chunks):
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "source": page["source"],
                        "page": page["page"],
                        "chunk_index": i
                    }
                })
        
        return chunks
    
    def process_directory(self, input_dir: str, output_dir: str):
        """
        Process all PDFs in directory and save chunks
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_chunks = []
        
        for pdf_file in input_path.glob("*.pdf"):
            print(f"Processing {pdf_file.name}...")
            pages = self.extract_text_from_pdf(str(pdf_file))
            chunks = self.create_chunks(pages)
            all_chunks.extend(chunks)
        
        # Save processed chunks
        output_file = output_path / "processed_chunks.json"
        with open(output_file, 'w') as f:
            json.dump(all_chunks, f, indent=2)
        
        print(f"Processed {len(all_chunks)} chunks from {len(list(input_path.glob('*.pdf')))} PDFs")
        return all_chunks


# Usage example
if __name__ == "__main__":
    processor = DocumentProcessor()
    chunks = processor.process_directory("data/raw", "data/processed")