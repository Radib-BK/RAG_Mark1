"""
Streamlit Frontend for Multilingual RAG System

A simple web interface to test the HSC Bangla RAG system.
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st
import requests
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Streamlit page config
st.set_page_config(
    page_title="HSC Bangla RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

def check_api_health():
    """Check if the API server is running and healthy"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)  # Reduced timeout
        if response.status_code == 200:
            health_data = response.json()
            status = health_data.get("status", "unknown")
            return status, health_data
        return "offline", None
    except requests.exceptions.Timeout:
        return "timeout", None
    except requests.exceptions.ConnectionError:
        return "connection_error", None
    except:
        return "offline", None

def ask_question(question: str, max_chunks: int = 5, threshold: float = 0.3, include_sources: bool = True):
    """Send question to RAG API"""
    try:
        payload = {
            "question": question,
            "max_chunks": max_chunks,
            "threshold": threshold,
            "include_sources": include_sources
        }
        
        response = requests.post(f"{API_URL}/ask", json=payload, timeout=120)
        
        if response.status_code == 200:
            return response.json(), None
        elif response.status_code == 408:
            return None, "⏰ Request timed out. Please try a shorter or simpler question."
        elif response.status_code == 500:
            error_detail = response.json().get("detail", "Internal server error")
            return None, f"❌ Server error: {error_detail}"
        else:
            return None, f"❌ Error {response.status_code}: {response.text}"
    
    except requests.exceptions.ConnectionError:
        return None, "❌ Cannot connect to API server. Please check if it's running on http://localhost:8000"
    except requests.exceptions.Timeout:
        return None, "⏰ Request timeout (120s). The question might be too complex or the server is overloaded."
    except Exception as e:
        return None, f"❌ Unexpected error: {str(e)}"

def get_system_stats():
    """Get system statistics from API"""
    try:
        response = requests.get(f"{API_URL}/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def clear_memory():
    """Clear conversation memory"""
    try:
        response = requests.post(f"{API_URL}/clear-memory", timeout=10)
        return response.status_code == 200
    except:
        return False

# Main UI
def main():
    st.title("📚 HSC Bangla RAG System")
    st.markdown("*Multilingual Question Answering for HSC Educational Content*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Status
        api_status, health_data = check_api_health()
        
        if api_status == "healthy":
            st.success("✅ API Server Connected")
        elif api_status == "degraded":
            st.warning("⚠️ API Server Degraded")
            if health_data:
                st.caption("Some components may not be fully functional")
        elif api_status == "unhealthy":
            st.error("❌ API Server Unhealthy")
            if health_data:
                st.caption("System initialized but has issues")
            st.markdown("**Check server logs or restart:**")
            st.code("python -m api.app", language="bash")
        elif api_status == "timeout":
            st.warning("⏳ API Server Slow")
            st.caption("Server is responding slowly, questions may take longer")
        elif api_status == "connection_error":
            st.error("🔌 Connection Failed")
            st.caption("Cannot connect to API server")
            st.markdown("**Start the server:**")
            st.code("python -m api.app", language="bash")
        else:  # offline
            st.error("❌ API Server Offline")
            st.markdown("**Start the server:**")
            st.code("python -m api.app", language="bash")
        
        # Don't stop the app - let users try asking questions anyway
        
        # Configuration
        st.subheader("🔧 Query Settings")
        max_chunks = st.slider("Max Context Chunks", 1, 10, 5)
        threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.3, 0.1)
        include_sources = st.checkbox("Include Sources", True)
        
        st.subheader("📊 System Status")
        if st.button("🔄 Refresh Stats"):
            stats = get_system_stats()
            if stats:
                st.json(stats['system_status'])
        
        if st.button("🧹 Clear Memory"):
            if clear_memory():
                st.success("Memory cleared!")
            else:
                st.error("Failed to clear memory")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("❓ Ask Questions")
        
        # Sample questions from HSC Bangla textbook
        st.markdown("**📝 Demo Questions (Test the Fixed RAG System):**")
        sample_questions = [
            "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
            "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
            "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
            "অনুপমের বয়স কত বছর?"
        ]
        
        for i, q in enumerate(sample_questions):
            if st.button(f"💡 {q[:50]}...", key=f"sample_{i}"):
                st.session_state.question_input = q
        
        # Question input
        question = st.text_area(
            "Your Question:",
            value=st.session_state.get('question_input', ''),
            height=100,
            placeholder="আপনার প্রশ্ন লিখুন... / Type your question..."
        )
        
        col_ask, col_clear = st.columns([3, 1])
        with col_ask:
            ask_button = st.button("🚀 Ask Question", type="primary", use_container_width=True)
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.question_input = ""
                st.rerun()
    
    with col2:
        st.header("📈 Quick Stats")
        stats = get_system_stats()
        if stats:
            # Safely get stats with defaults
            total_vectors = stats.get('vector_store_stats', {}).get('total_vectors', 0)
            conversations = stats.get('conversation_count', 0)
            index_size = stats.get('vector_store_stats', {}).get('index_size_mb', 0.0)
            
            st.metric("Total Vectors", total_vectors)
            st.metric("Conversations", conversations)
            st.metric("Vector Store Size", f"{index_size:.1f} MB")
    
    # Process question
    if ask_button and question.strip():
        # Quick API check before processing
        quick_status, _ = check_api_health()
        if quick_status in ["connection_error", "offline"]:
            st.error("❌ Cannot connect to API server. Please start the server with: `python -m api.app`")
        else:
            with st.spinner("🤔 Thinking..."):
                start_time = time.time()
                result, error = ask_question(question, max_chunks, threshold, include_sources)
                elapsed_time = time.time() - start_time
            
            if error:
                st.error(error)
            elif result:
                # Display answer
                st.success("✅ Answer Generated")
                
                # Answer section
                with st.container():
                    st.markdown("### 💬 Answer")
                    st.markdown(f"**{result['answer']}**")
                    
                    # Metadata
                    col_meta1, col_meta2, col_meta3 = st.columns(3)
                    with col_meta1:
                        st.metric("Language", result['query_language'])
                    with col_meta2:
                        st.metric("Confidence", f"{result['confidence_score']:.2%}")
                    with col_meta3:
                        st.metric("Response Time", f"{result['response_time_ms']} ms")
                
                # Sources section
                if include_sources and result.get('sources'):
                    with st.expander("📚 Sources", expanded=False):
                        for i, source in enumerate(result['sources']):
                            st.markdown(f"**Source {i+1}** (Score: {source['score']:.3f})")
                            st.markdown(f"*Page {source['metadata']['source_page']} - {source['metadata'].get('content_type', 'content')}*")
                            st.text_area(
                                f"Content {i+1}:", 
                                source['content'][:500] + ("..." if len(source['content']) > 500 else ""),
                                height=100,
                                key=f"source_{i}",
                                disabled=True
                            )
                            st.markdown("---")
    
    elif ask_button:
        st.warning("⚠️ Please enter a question first!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p><strong>HSC Bangla RAG System</strong> | Built with ❤️ for multilingual education</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 