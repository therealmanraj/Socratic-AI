import streamlit as st
from src.vectorstore_manager import VectorStoreManager
from src.rag_pipeline import RAGPipeline
from src.config import Config
from datetime import datetime

# Page config
st.set_page_config(
    page_title="AeroMind - Aircraft Maintenance AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better look
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E40AF;
    }
    .confidence-high {
        color: #059669;
        font-weight: bold;
    }
    .confidence-low {
        color: #DC2626;
        font-weight: bold;
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
    with st.spinner("Loading AeroMind..."):
        vs_manager = VectorStoreManager(
            embedding_model_name=Config.EMBEDDING_MODEL,
            vectorstore_dir=Config.VECTORSTORE_DIR
        )
        st.session_state.rag_pipeline = RAGPipeline(vs_manager, Config.LLM_MODEL)
        st.session_state.query_history = []

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<p class="main-header">✈️ AeroMind</p>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Aircraft Maintenance Assistant**")
with col2:
    st.metric("Manuals Loaded", "A320 Manual")  # Make this dynamic later

# Sidebar
with st.sidebar:
    st.header("📚 About AeroMind")
    st.write("""
    AeroMind helps maintenance technicians find answers faster by intelligently 
    searching through aircraft maintenance manuals.
    """)
    
    st.divider()
    
    st.header("💡 Example Questions")
    example_queries = [
        "How do I troubleshoot APU start failure?",
        "What is the hydraulic fluid specification?",
        "Landing gear retraction test procedure",
        "Engine fire warning light troubleshooting",
        "Pre-flight inspection checklist"
    ]
    
    for example in example_queries:
        if st.button(example, key=f"example_{example[:20]}", use_container_width=True):
            st.session_state.current_query = example
    
    st.divider()
    
    st.header("📊 Session Stats")
    st.metric("Queries This Session", len(st.session_state.query_history))
    
    if st.button("Clear History", use_container_width=True):
        st.session_state.query_history = []
        st.rerun()

# Main interface
st.subheader("Ask a Maintenance Question")

# Query input
query = st.text_input(
    "Type your question:",
    value=st.session_state.get('current_query', ''),
    placeholder="e.g., How do I check hydraulic pressure?",
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
    with st.spinner("🔎 Searching manuals..."):
        result = st.session_state.rag_pipeline.query(query)
        
        # Add to history
        st.session_state.query_history.append({
            "timestamp": datetime.now(),
            "query": query,
            "result": result
        })
        
        # Display results
        st.divider()
        
        # Confidence indicator
        confidence = result.get("confidence", "unknown")
        if confidence == "high":
            st.success("✅ High Confidence Answer")
        elif confidence == "low":
            st.warning("⚠️ Low Confidence - Verify carefully")
        else:
            st.error("❌ No relevant information found")
        
        # Answer
        st.markdown("### 💡 Answer")
        st.markdown(result["answer"])
        
        # Sources
        if result["sources"]:
            st.markdown("### 📚 Sources Referenced")
            
            for idx, source in enumerate(result["sources"], 1):
                with st.expander(
                    f"📄 Source {idx}: {source['source']} - Page {source['page']}",
                    expanded=(idx == 1)  # Expand first source
                ):
                    st.markdown(f"**Content Preview:**")
                    st.text(source["content"])
        else:
            st.warning("⚠️ No sources found - answer may not be reliable")
        
        # Quality warnings
        quality_check = result.get("quality_check", {})
        if quality_check.get("has_issues"):
            with st.expander("⚠️ Quality Issues Detected"):
                for issue in quality_check.get("issues", []):
                    st.warning(f"• {issue}")

elif search_button and not query:
    st.warning("⚠️ Please enter a question")

# Show history (optional - collapsible)
if st.session_state.query_history:
    with st.expander("📜 Query History", expanded=False):
        for i, item in enumerate(reversed(st.session_state.query_history[-5:]), 1):
            st.markdown(f"**{i}.** {item['query']}")
            st.caption(f"Asked at: {item['timestamp'].strftime('%H:%M:%S')}")
            st.divider()

# Footer
st.markdown("---")
st.caption("⚠️ **Important:** Always verify information with certified manuals before performing maintenance. AeroMind is an assistant tool, not a replacement for certified procedures.")
st.caption("🔒 Prototype v0.1 | For demonstration purposes")