# src/document_processor_with_images.py - Enhanced with image extraction

import pymupdf
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
import json
import re
import base64
from PIL import Image
import io

class DocumentProcessorWithImages:
    """Enhanced PDF processor that handles both text AND images/diagrams"""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200, extract_images=True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extract_images = extract_images
        
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
            "is_procedure": bool(re.search(r'STEP \d+|PROCEDURE', text, re.IGNORECASE)),
            "has_diagram": bool(re.search(r'FIG(?:URE)?[\s\.]?\d+|DIAGRAM|ILLUSTRATION', text, re.IGNORECASE))
        }
        
        # Try to extract ATA chapter
        ata_match = re.search(r'(?:ATA[\s-]?)?(\d{2})-?(\d{2})?-?(\d{2})?', text)
        if ata_match:
            metadata["ata_chapter"] = ata_match.group(0)
        
        # Try to extract aircraft type
        aircraft_patterns = [
            r'(A\d{3}(?:-\d{3})?)',
            r'(Boeing\s+)?(\d{3}(?:-\d{1,3})?)',
            r'(B\d{3})'
        ]
        for pattern in aircraft_patterns:
            match = re.search(pattern, text)
            if match:
                metadata["aircraft_type"] = match.group(0)
                break
        
        return metadata
    
    def extract_images_from_page(self, page, page_num: int, pdf_name: str, output_dir: Path) -> List[Dict]:
        """Extract images from a PDF page and save them"""
        images_data = []
        
        if not self.extract_images:
            return images_data
        
        # Create images directory
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Get images from page
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Save image
                image_filename = f"{Path(pdf_name).stem}_p{page_num}_img{img_index}.{image_ext}"
                image_path = images_dir / image_filename
                
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                # Convert to base64 for storage (optional - for multimodal models)
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                
                # Get image dimensions
                try:
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    width, height = pil_image.size
                except:
                    width, height = None, None
                
                images_data.append({
                    "image_index": img_index,
                    "filename": image_filename,
                    "path": str(image_path),
                    "base64": image_base64[:100] + "...",  # Store truncated for JSON
                    "full_base64": image_base64,  # Full version for later use
                    "ext": image_ext,
                    "width": width,
                    "height": height,
                    "page": page_num,
                    "source": pdf_name
                })
                
            except Exception as e:
                print(f"Error extracting image {img_index} from page {page_num}: {e}")
                continue
        
        return images_data
    
    def extract_text_and_images_from_pdf(self, pdf_path: str, output_dir: Path) -> List[Dict]:
        """Extract both text AND images with metadata"""
        doc = pymupdf.open(pdf_path)
        pages = []
        
        for page_num, page in enumerate(doc):
            # Extract text
            text = page.get_text()
            
            # Extract images
            images = self.extract_images_from_page(
                page, 
                page_num + 1, 
                Path(pdf_path).name,
                output_dir
            )
            
            # Extract metadata
            base_metadata = {
                "page": page_num + 1,
                "source": Path(pdf_path).name,
                "num_images": len(images)
            }
            
            # Add aviation-specific metadata
            aviation_metadata = self.extract_metadata_from_text(text, Path(pdf_path).name)
            base_metadata.update(aviation_metadata)
            
            pages.append({
                "text": text,
                "metadata": base_metadata,
                "images": images
            })
        
        doc.close()
        return pages
    
    def create_chunks_with_images(self, pages: List[Dict]) -> List[Dict]:
        """Create chunks and associate them with relevant images"""
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
                
                # Merge metadata
                chunk_metadata.update({
                    k: v for k, v in chunk_specific.items() 
                    if v is not None and k not in ["source"]
                })
                
                # Associate images with this chunk if they're on the same page
                associated_images = []
                if page.get("images"):
                    # Check if chunk mentions figures/diagrams
                    fig_refs = re.findall(r'FIG(?:URE)?[\s\.]?(\d+)', chunk_text, re.IGNORECASE)
                    
                    # If chunk references figures, associate all page images
                    # (More sophisticated matching could be done here)
                    if fig_refs or chunk_metadata.get("has_diagram"):
                        associated_images = page["images"]
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata,
                    "images": associated_images
                })
        
        return chunks
    
    def process_directory(self, input_dir: str, output_dir: str):
        """Process all PDFs with text AND image extraction"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_chunks = []
        all_images = []
        stats = {
            "total_pdfs": 0,
            "total_pages": 0,
            "total_chunks": 0,
            "total_images": 0,
            "chunks_with_warnings": 0,
            "chunks_with_procedures": 0,
            "chunks_with_diagrams": 0
        }
        
        for pdf_file in input_path.glob("*.pdf"):
            print(f"Processing {pdf_file.name}...")
            stats["total_pdfs"] += 1
            
            pages = self.extract_text_and_images_from_pdf(str(pdf_file), output_path)
            stats["total_pages"] += len(pages)
            
            # Collect all images
            for page in pages:
                all_images.extend(page.get("images", []))
            
            chunks = self.create_chunks_with_images(pages)
            
            # Update stats
            stats["total_chunks"] += len(chunks)
            stats["total_images"] += sum(len(p.get("images", [])) for p in pages)
            stats["chunks_with_warnings"] += sum(
                1 for c in chunks if c["metadata"].get("has_warning")
            )
            stats["chunks_with_procedures"] += sum(
                1 for c in chunks if c["metadata"].get("is_procedure")
            )
            stats["chunks_with_diagrams"] += sum(
                1 for c in chunks if c.get("images") and len(c["images"]) > 0
            )
            
            all_chunks.extend(chunks)
        
        # Save chunks (without full base64 to keep file size manageable)
        chunks_for_save = []
        for chunk in all_chunks:
            chunk_copy = chunk.copy()
            # Remove full_base64 from images to reduce file size
            if chunk_copy.get("images"):
                chunk_copy["images"] = [
                    {k: v for k, v in img.items() if k != "full_base64"}
                    for img in chunk_copy["images"]
                ]
            chunks_for_save.append(chunk_copy)
        
        output_file = output_path / "processed_chunks_with_images.json"
        with open(output_file, 'w') as f:
            json.dump(chunks_for_save, f, indent=2)
        
        # Save image catalog separately
        images_catalog = output_path / "images_catalog.json"
        with open(images_catalog, 'w') as f:
            json.dump(all_images, f, indent=2)
        
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
    import sys
    from pathlib import Path
    
    # Add parent directory to path so we can import from src
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.config import Config
    
    print("="*60)
    print("AEROMIND DOCUMENT PROCESSOR (WITH IMAGES)")
    print("="*60)
    
    processor = DocumentProcessorWithImages(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        extract_images=True
    )
    
    print(f"\nProcessing PDFs from: {Config.RAW_DATA_DIR}")
    print(f"Output will be saved to: {Config.PROCESSED_DATA_DIR}\n")
    
    chunks = processor.process_directory(
        Config.RAW_DATA_DIR, 
        Config.PROCESSED_DATA_DIR
    )
    
    print(f"\n✅ Processing complete!")
    print(f"📄 Total chunks created: {len(chunks)}")
    print(f"🖼️  Images extracted to: {Config.PROCESSED_DATA_DIR}/images/")