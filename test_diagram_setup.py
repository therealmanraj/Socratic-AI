#!/usr/bin/env python3
"""
Quick test script to verify diagram extraction is working
"""

import sys
from pathlib import Path

print("="*60)
print("AEROMIND DIAGRAM FEATURE TEST")
print("="*60)

# Test 1: Check if image processing works
print("\n[1/5] Testing image processing library...")
try:
    from PIL import Image
    import io
    print("✅ PIL/Pillow installed correctly")
except ImportError as e:
    print(f"❌ Error: {e}")
    print("Run: pip install Pillow")
    sys.exit(1)

# Test 2: Check if vision model is available
print("\n[2/5] Checking for vision model (llava)...")
try:
    import subprocess
    result = subprocess.run(
        ["ollama", "list"], 
        capture_output=True, 
        text=True
    )
    if "llava" in result.stdout.lower():
        print("✅ LLaVA vision model found")
        vision_available = True
    else:
        print("⚠️  LLaVA not found (optional)")
        print("   To enable vision: ollama pull llava")
        vision_available = False
except Exception as e:
    print(f"⚠️  Could not check Ollama models: {e}")
    vision_available = False

# Test 3: Check if PDFs exist
print("\n[3/5] Checking for PDF manuals...")
pdf_dir = Path("data/raw")
if pdf_dir.exists():
    pdfs = list(pdf_dir.glob("*.pdf"))
    if pdfs:
        print(f"✅ Found {len(pdfs)} PDF(s):")
        for pdf in pdfs:
            print(f"   - {pdf.name}")
    else:
        print("⚠️  No PDFs found in data/raw/")
        print("   Add some PDF manuals to test image extraction")
else:
    print("❌ data/raw/ directory not found")
    print("   Create it and add PDF manuals")

# Test 4: Try extracting images from first page of first PDF
print("\n[4/5] Testing image extraction...")
if pdfs:
    try:
        import pymupdf
        
        test_pdf = pdfs[0]
        print(f"   Testing with: {test_pdf.name}")
        
        doc = pymupdf.open(str(test_pdf))
        first_page = doc[0]
        images = first_page.get_images()
        
        print(f"✅ Found {len(images)} image(s) on first page")
        
        if images:
            # Try to extract first image
            xref = images[0][0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Verify it's a valid image
            pil_image = Image.open(io.BytesIO(image_bytes))
            width, height = pil_image.size
            print(f"   First image: {width}x{height}px, format: {base_image['ext']}")
            print("✅ Image extraction working!")
        else:
            print("ℹ️  No images on first page (may be on other pages)")
        
        doc.close()
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
else:
    print("⏭️  Skipping (no PDFs to test)")

# Test 5: Check if enhanced processor exists
print("\n[5/5] Checking for enhanced processor...")
if Path("document_processor_with_images.py").exists():
    print("✅ Enhanced processor file found")
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Run the enhanced processor:")
    print("   python document_processor_with_images.py")
    print("\n2. Launch the app with diagrams:")
    print("   streamlit run streamlit_app_with_diagrams.py")
    print("\n3. Try a query like:")
    print("   'Show me the hydraulic system diagram'")
    
    if not vision_available:
        print("\n💡 OPTIONAL: For AI diagram analysis:")
        print("   ollama pull llava")
    
    print("="*60)
else:
    print("❌ Enhanced processor not found")
    print("   Make sure document_processor_with_images.py is in your directory")

print("\n✨ Test complete!")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Image Processing: {'✅' if 'Image' in dir() else '❌'}")
print(f"Vision Model: {'✅' if vision_available else '⚠️  (optional)'}")
print(f"PDF Manuals: {'✅' if pdfs else '⚠️'}")
print(f"Enhanced Processor: {'✅' if Path('document_processor_with_images.py').exists() else '❌'}")
print("="*60)