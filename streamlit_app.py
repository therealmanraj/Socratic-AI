import streamlit as st
from src.vectorstore_manager import VectorStoreManager
from src.rag_pipeline_multimodal import MultimodalRAGPipeline, SimpleImageRAGPipeline
from src.config import Config
from datetime import datetime
from pathlib import Path
from PIL import Image

# Page config
st.set_page_config(
    page_title="AeroMind - Aircraft Maintenance AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E40AF;
    }
    .diagram-box {
        background-color: #F9FAFB;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #3B82F6;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E40AF;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    with st.spinner("Loading AeroMind with Vision Support..."):
        vs_manager = VectorStoreManager(
            embedding_model_name=Config.EMBEDDING_MODEL,
            vectorstore_dir=Config.VECTORSTORE_DIR
        )
        
        # Try to use multimodal, fall back to simple if vision model not available
        try:
            st.session_state.rag_pipeline = MultimodalRAGPipeline(
                vs_manager, 
                text_llm_model=Config.LLM_MODEL,
                vision_llm_model="llava"  # Requires: ollama pull llava
            )
            st.session_state.vision_enabled = True
        except:
            st.session_state.rag_pipeline = SimpleImageRAGPipeline(
                vs_manager,
                llm_model=Config.LLM_MODEL
            )
            st.session_state.vision_enabled = False
        
        st.session_state.query_history = []

# Header
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.markdown('<p class="main-header">✈️ AeroMind</p>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Aircraft Maintenance Assistant with Diagram Support**")
with col2:
    st.metric("Manuals Loaded", "3 PDFs")
with col3:
    if st.session_state.vision_enabled:
        st.success("🖼️ Vision: ON")
    else:
        st.warning("🖼️ Vision: OFF")

# Sidebar
with st.sidebar:
    st.header("📚 About AeroMind")
    st.write("""
    AeroMind helps maintenance technicians find answers faster by intelligently 
    searching through aircraft maintenance manuals **and their diagrams**.
    """)
    
    if not st.session_state.vision_enabled:
        st.info("💡 **Enable Vision:** Install llava model for diagram analysis:\n`ollama pull llava`")
    
    st.divider()
    
    st.header("💡 Example Questions")
    example_queries = [
        "Show me the hydraulic system diagram",
        "Where is the APU located? Show me a diagram",
        "Landing gear components diagram",
        "Fuel system schematic",
        "How do I identify the bleed air valves? (with diagram)",
        "Engine fire detection system layout"
    ]
    
    for example in example_queries:
        if st.button(example, key=f"example_{example[:20]}", use_container_width=True):
            st.session_state.current_query = example
    
    st.divider()
    
    st.header("⚙️ Options")
    include_diagrams = st.checkbox("Include Diagrams", value=True)
    analyze_diagrams = st.checkbox(
        "Analyze Diagrams (slower)", 
        value=st.session_state.vision_enabled,
        disabled=not st.session_state.vision_enabled
    )
    
    st.divider()
    
    st.header("📊 Session Stats")
    st.metric("Queries This Session", len(st.session_state.query_history))
    
    if st.button("Clear History", use_container_width=True):
        st.session_state.query_history = []
        st.rerun()

# Main interface
st.subheader("Ask a Maintenance Question")

query = st.text_input(
    "Type your question (mention 'diagram' for visual results):",
    value=st.session_state.get('current_query', ''),
    placeholder="e.g., Show me the fuel system diagram",
    key="query_input"
)

col1, col2, col3 = st.columns([2, 2, 6])
with col1:
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("Clear", use_container_width=True)

if clear_button:
    st.session_state.current_query = ""
    st.rerun()

# Process query
if search_button and query:
    with st.spinner("🔎 Searching manuals and diagrams..."):
        result = st.session_state.rag_pipeline.query(
            query,
            return_sources=True,
            include_images=include_diagrams
        )
        
        st.session_state.query_history.append({
            "timestamp": datetime.now(),
            "query": query,
            "result": result
        })
        
        st.divider()
        
        # Confidence indicator
        confidence = result.get("confidence", "unknown")
        if confidence == "high":
            st.success("✅ High Confidence Answer")
        elif confidence == "medium":
            st.info("ℹ️ Medium Confidence - Review diagrams carefully")
        else:
            st.warning("⚠️ Low Confidence")
        
        # Answer
        st.markdown("### 💡 Answer")
        st.markdown(result["answer"])
        
        # DIAGRAMS SECTION (New!)
        if result.get("images") and len(result["images"]) > 0:
            st.markdown("### 🖼️ Related Diagrams")
            
            num_images = len(result["images"])
            cols_per_row = 2
            
            for i in range(0, num_images, cols_per_row):
                cols = st.columns(cols_per_row)
                
                for j, col in enumerate(cols):
                    img_idx = i + j
                    if img_idx < num_images:
                        img_data = result["images"][img_idx]
                        
                        with col:
                            st.markdown(f"**📄 {img_data['source']} - Page {img_data['page']}**")
                            
                            # Display image
                            img_path = Path(img_data['path'])
                            if img_path.exists():
                                try:
                                    image = Image.open(img_path)
                                    st.image(
                                        image, 
                                        caption=f"{img_data['filename']}", 
                                        use_container_width=True
                                    )
                                    
                                    # Show vision analysis if available
                                    if analyze_diagrams and result.get("image_analyses"):
                                        for analysis in result["image_analyses"]:
                                            if analysis["filename"] == img_data["filename"]:
                                                with st.expander("🤖 AI Analysis of Diagram"):
                                                    st.write(analysis["analysis"])
                                    
                                    # Download button
                                    with open(img_path, "rb") as file:
                                        st.download_button(
                                            label="⬇️ Download",
                                            data=file,
                                            file_name=img_data["filename"],
                                            mime=f"image/{img_data.get('ext', 'png')}",
                                            key=f"download_{img_idx}"
                                        )
                                except Exception as e:
                                    st.error(f"Could not load image: {e}")
                            else:
                                st.warning(f"Image file not found: {img_path}")
        
        elif include_diagrams:
            st.info("ℹ️ No diagrams found for this query. Try asking explicitly for diagrams.")
        
        # Text Sources
        if result["sources"]:
            st.markdown("### 📚 Text Sources Referenced")
            
            for idx, source in enumerate(result["sources"], 1):
                with st.expander(
                    f"📄 Source {idx}: {source['source']} - Page {source['page']}",
                    expanded=(idx == 1)
                ):
                    st.markdown(f"**Content Preview:**")
                    st.text(source.get("content", "No preview available"))
        
        # Quality warnings
        quality_check = result.get("quality_check", {})
        if quality_check.get("has_issues"):
            with st.expander("⚠️ Quality Issues Detected"):
                for issue in quality_check.get("issues", []):
                    st.warning(f"• {issue}")

elif search_button and not query:
    st.warning("⚠️ Please enter a question")

# Show history
if st.session_state.query_history:
    with st.expander("📜 Query History", expanded=False):
        for i, item in enumerate(reversed(st.session_state.query_history[-5:]), 1):
            st.markdown(f"**{i}.** {item['query']}")
            st.caption(f"Asked at: {item['timestamp'].strftime('%H:%M:%S')}")
            if item['result'].get('has_diagrams'):
                st.caption(f"   🖼️ {len(item['result']['images'])} diagram(s) found")
            st.divider()

# Footer
st.markdown("---")
st.caption("⚠️ **Important:** Always verify information with certified manuals before performing maintenance.")
st.caption("🔒 Prototype v0.2 - Now with Diagram Support | For demonstration purposes")