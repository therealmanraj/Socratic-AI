# streamlit_app.py
import streamlit as st
from src.vectorstore_manager import VectorStoreManager
from src.rag_pipeline import RAGPipeline
from src.config import Config

# Page config
st.set_page_config(
    page_title="AeroMind - Aircraft Maintenance AI",
    page_icon="✈️",
    layout="wide"
)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    vs_manager = VectorStoreManager(
        embedding_model_name=Config.EMBEDDING_MODEL,
        vectorstore_dir=Config.VECTORSTORE_DIR
    )
    st.session_state.rag_pipeline = RAGPipeline(vs_manager, Config.LLM_MODEL)

# Header
st.title("✈️ AeroMind")
st.subheader("AI-Powered Aircraft Maintenance Assistant")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("""
    AeroMind helps aircraft maintenance technicians find answers faster by 
    intelligently searching through maintenance manuals.
    """)
    
    st.header("Example Questions")
    st.write("""
    - How do I troubleshoot APU start failure?
    - What's the procedure for landing gear retraction test?
    - Engine fault code 77-3421
    """)

# Main interface
query = st.text_input(
    "Ask a maintenance question:",
    placeholder="e.g., How do I troubleshoot hydraulic pressure low?"
)

if st.button("Search", type="primary") or query:
    if query:
        with st.spinner("Searching manuals..."):
            result = st.session_state.rag_pipeline.query(query)
            
            # Display answer
            st.markdown("### Answer")
            st.write(result["answer"])
            
            # Display sources
            st.markdown("### Sources")
            for idx, source in enumerate(result["sources"], 1):
                with st.expander(f"Source {idx}: {source['source']} (Page {source['page']})"):
                    st.text(source["content"])
    else:
        st.warning("Please enter a question")

# Footer
st.markdown("---")
st.caption("⚠️ Always verify information with certified manuals before performing maintenance")