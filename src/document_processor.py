# src/document_processor.py - Enhanced version

import pymupdf
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
import json
import re

class DocumentProcessor:
    """Enhanced PDF processor with aviation-specific handling"""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Use custom separators for aviation manuals
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n## ",      # Major sections
                "\n### ",     # Subsections
                "\nSTEP ",    # Procedure steps
                "\nNOTE:",    # Important notes
                "\nWARNING:", # Safety warnings
                "\nCAUTION:", # Caution notices
                "\n\n",       # Paragraph breaks
                "\n",
                ". ",
                " ",
                ""
            ],
            keep_separator=True
        )
    
    def extract_metadata_from_text(self, text: str, filename: str) -> dict:
        """Extract aviation-specific metadata"""
        metadata = {
            "source": filename,
            "ata_chapter": None,
            "aircraft_type": None,
            "has_warning": "WARNING" in text.upper(),
            "has_caution": "CAUTION" in text.upper(),
            "is_procedure": bool(re.search(r'STEP \d+|PROCEDURE', text, re.IGNORECASE))
        }
        
        # Try to extract ATA chapter (e.g., "32-41-00" or "ATA 32")
        ata_match = re.search(r'(?:ATA[\s-]?)?(\d{2})-?(\d{2})?-?(\d{2})?', text)
        if ata_match:
            metadata["ata_chapter"] = ata_match.group(0)
        
        # Try to extract aircraft type
        aircraft_patterns = [
            r'(A\d{3}(?:-\d{3})?)',  # Airbus: A320, A320-200
            r'(Boeing\s+)?(\d{3}(?:-\d{1,3})?)',  # Boeing: 737, 737-800
            r'(B\d{3})'  # B737
        ]
        for pattern in aircraft_patterns:
            match = re.search(pattern, text)
            if match:
                metadata["aircraft_type"] = match.group(0)
                break
        
        return metadata
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text with enhanced metadata"""
        doc = pymupdf.open(pdf_path)
        pages = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            
            # Extract metadata
            base_metadata = {
                "page": page_num + 1,
                "source": Path(pdf_path).name
            }
            
            # Add aviation-specific metadata
            aviation_metadata = self.extract_metadata_from_text(text, Path(pdf_path).name)
            base_metadata.update(aviation_metadata)
            
            pages.append({
                "text": text,
                "metadata": base_metadata
            })
        
        doc.close()
        return pages
    
    def create_chunks(self, pages: List[Dict]) -> List[Dict]:
        """Create chunks preserving metadata"""
        chunks = []
        
        for page in pages:
            text_chunks = self.text_splitter.split_text(page["text"])
            
            for i, chunk_text in enumerate(text_chunks):
                # Create chunk with inherited metadata
                chunk_metadata = page["metadata"].copy()
                chunk_metadata["chunk_index"] = i
                
                # Re-analyze chunk for specific metadata
                chunk_specific = self.extract_metadata_from_text(
                    chunk_text,
                    page["metadata"]["source"]
                )
                
                # Merge metadata (chunk-specific overrides page-level)
                chunk_metadata.update({
                    k: v for k, v in chunk_specific.items() 
                    if v is not None and k not in ["source"]
                })
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
        
        return chunks
    
    def process_directory(self, input_dir: str, output_dir: str):
        """Process all PDFs with enhanced metadata"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_chunks = []
        stats = {
            "total_pdfs": 0,
            "total_pages": 0,
            "total_chunks": 0,
            "chunks_with_warnings": 0,
            "chunks_with_procedures": 0
        }
        
        for pdf_file in input_path.glob("*.pdf"):
            print(f"Processing {pdf_file.name}...")
            stats["total_pdfs"] += 1
            
            pages = self.extract_text_from_pdf(str(pdf_file))
            stats["total_pages"] += len(pages)
            
            chunks = self.create_chunks(pages)
            
            # Update stats
            stats["total_chunks"] += len(chunks)
            stats["chunks_with_warnings"] += sum(
                1 for c in chunks if c["metadata"].get("has_warning")
            )
            stats["chunks_with_procedures"] += sum(
                1 for c in chunks if c["metadata"].get("is_procedure")
            )
            
            all_chunks.extend(chunks)
        
        # Save chunks
        output_file = output_path / "processed_chunks.json"
        with open(output_file, 'w') as f:
            json.dump(all_chunks, f, indent=2)
        
        # Save stats
        stats_file = output_path / "processing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("PROCESSING SUMMARY")
        print("="*50)
        for key, value in stats.items():
            print(f"{key}: {value}")
        print("="*50)
        
        return all_chunks
    
if __name__ == "__main__":
    from src.config import Config
    
    print("="*60)
    print("AEROMIND DOCUMENT PROCESSOR")
    print("="*60)
    
    processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    
    print(f"\nProcessing PDFs from: {Config.RAW_DATA_DIR}")
    print(f"Output will be saved to: {Config.PROCESSED_DATA_DIR}\n")
    
    chunks = processor.process_directory(
        Config.RAW_DATA_DIR, 
        Config.PROCESSED_DATA_DIR
    )
    
    print(f"\n✅ Processing complete!")
    print(f"📄 Total chunks created: {len(chunks)}")