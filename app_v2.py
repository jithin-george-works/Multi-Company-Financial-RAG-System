# app.py
import streamlit as st
from agent import answer_question_streaming, answer_question_crewai
from config import DATA_DIR
import time
from datetime import datetime

st.set_page_config(page_title="Multi-Company RAG (CrewAI)", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
    }
    .step-header {
        font-size: 1.2em;
        font-weight: bold;
        color: #1976D2;
        margin: 20px 0 10px 0;
    }
    .answer-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Multi-Company RAG — AI Agent System")
st.markdown("**Real-time streaming analysis** powered by CrewAI agents and Ollama")

with st.sidebar:
    st.header("📋 Sample Queries")
    st.markdown("""
    **Simple Queries:**
    - What is the Q2 revenue of Company A?
    - Who is the CEO of Company B?
    - What was Company A's profit margin in Q4?
    
    **Complex Queries:**
    - Compare the yearly revenue of Company A and Company B
    - What are the revenues and profit margins of both companies?
    - Which company has better ESG initiatives?
    """)
    
    st.markdown("---")
    
    st.header("⚙️ Settings")
    streaming_mode = st.radio(
        "Response Mode",
        ["Real-time Streaming", "Full CrewAI (No Streaming)"],
        index=0
    )
    
    show_context = st.checkbox("Show retrieved context", value=False)
    show_metadata = st.checkbox("Show metadata", value=True)
    
    st.markdown("---")
    st.info("💡 **Streaming Mode:** Watch the AI think in real-time as it decomposes, researches, and synthesizes answers.")

# Main query input
query = st.text_area(
    "Enter your question about Company A and/or Company B:", 
    height=100,
    placeholder="E.g., What are the revenues of Company A and Company B?"
)

col1, col2 = st.columns([1, 5])
with col1:
    run_btn = st.button("🚀 Analyze", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ Clear History", use_container_width=True):
        if 'history' in st.session_state:
            st.session_state.history = []
            st.rerun()

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

if run_btn and query.strip():
    start_time = time.time()
    timestamp = datetime.now().strftime("%I:%M:%S %p")
    
    # Create containers
    status_container = st.container()
    workflow_container = st.container()
    answer_container = st.container()
    metadata_container = st.container()
    
    if streaming_mode == "Real-time Streaming":
        # ===== STREAMING MODE =====
        with status_container:
            st.markdown("### 🔄 Live Agent Workflow")
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
        
        with workflow_container:
            workflow_placeholder = st.empty()
        
        with answer_container:
            answer_placeholder = st.empty()
        
        # Initialize display
        workflow_text = ""
        answer_text = ""
        current_step = ""
        context_info = None
        progress = 0
        
        try:
            for event in answer_question_streaming(query):
                event_type = event['type']
                content = event['content']
                
                if event_type == 'status':
                    # Update status
                    status_placeholder.info(content)
                    
                elif event_type == 'step':
                    # Update progress based on step
                    current_step = content
                    if content == 'decomposition':
                        progress = 20
                        workflow_text += "\n\n### 🔍 Step 1: Query Decomposition\n\n"
                    elif content == 'research':
                        progress = 50
                        workflow_text += "\n\n### 📚 Step 2: Database Research\n\n"
                    elif content == 'synthesis':
                        progress = 80
                        workflow_text = ""  # Clear for final answer
                    
                    progress_bar.progress(progress)
                    workflow_placeholder.markdown(workflow_text)
                
                elif event_type == 'token':
                    # Stream tokens
                    if current_step == 'synthesis':
                        # Show answer tokens
                        answer_text += content
                        answer_placeholder.markdown(answer_text)
                    else:
                        # Show workflow tokens
                        workflow_text += content
                        workflow_placeholder.markdown(workflow_text)
                
                elif event_type == 'context':
                    # Store context info
                    context_info = content
            
            # Completion
            progress_bar.progress(100)
            status_placeholder.success("✅ Analysis complete!")
            
            elapsed_time = time.time() - start_time
            
            # Show metadata
            if show_metadata and context_info:
                with metadata_container:
                    st.markdown("---")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("⏱️ Time", f"{elapsed_time:.2f}s")
                    with col2:
                        st.metric("📄 Documents", context_info['total_documents'])
                    with col3:
                        st.metric("🔍 Sub-queries", len(context_info['research_results']))
                    with col4:
                        st.metric("🗄️ Databases", "2 (A & B)")
                    
                    if show_context:
                        with st.expander("🔍 Research Details"):
                            for i, res in enumerate(context_info['research_results'], 1):
                                st.markdown(f"""
                                **Query {i}:** {res['query']}  
                                - Company: {res['company']}  
                                - Documents found: {res['docs_found']}
                                """)
            
            # Add to history
            st.session_state.history.append({
                'timestamp': timestamp,
                'query': query,
                'answer': answer_text,
                'elapsed_time': elapsed_time,
                'mode': 'streaming'
            })
            
        except Exception as e:
            status_placeholder.error(f"❌ Error: {str(e)}")
            st.exception(e)
    
    else:
        # ===== NON-STREAMING CREWAI MODE =====
        with status_container:
            st.markdown("### 🤖 CrewAI Multi-Agent System")
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
        
        status_placeholder.info("🔄 Initializing 3 AI agents...")
        progress_bar.progress(10)
        time.sleep(0.3)
        
        status_placeholder.info("🔍 Agent 1: Decomposing query...")
        progress_bar.progress(30)
        time.sleep(0.3)
        
        status_placeholder.info("📚 Agent 2: Researching databases...")
        progress_bar.progress(60)
        
        try:
            result = answer_question_crewai(query, verbose=False)
            
            status_placeholder.info("🧠 Agent 3: Synthesizing answer...")
            progress_bar.progress(90)
            time.sleep(0.3)
            
            progress_bar.progress(100)
            status_placeholder.success("✅ All agents completed!")
            
            elapsed_time = time.time() - start_time
            
            # Display answer
            with answer_container:
                st.markdown("### 💡 Final Answer")
                st.markdown(f"""
                <div class="answer-box">
                {result}
                </div>
                """, unsafe_allow_html=True)
            
            # Metadata
            if show_metadata:
                with metadata_container:
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("⏱️ Time", f"{elapsed_time:.2f}s")
                    with col2:
                        st.metric("🤖 Agents", "3")
                    with col3:
                        st.metric("📋 Tasks", "3")
            
            # Add to history
            st.session_state.history.append({
                'timestamp': timestamp,
                'query': query,
                'answer': result,
                'elapsed_time': elapsed_time,
                'mode': 'crewai'
            })
            
        except Exception as e:
            status_placeholder.error(f"❌ Error: {str(e)}")
            st.exception(e)

# Show conversation history
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📜 Conversation History")
    
    for i, item in enumerate(reversed(st.session_state.history[-5:])):
        mode_badge = "🌊 Streaming" if item['mode'] == 'streaming' else "🤖 CrewAI"
        with st.expander(f"[{item['timestamp']}] {mode_badge} — {item['query'][:60]}...", expanded=(i==0)):
            st.markdown(f"**Q:** {item['query']}")
            st.markdown(f"**A:** {item['answer'][:500]}...")
            st.caption(f"⏱️ {item['elapsed_time']:.2f}s")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
🚀 Powered by <strong>CrewAI</strong> | <strong>LangChain</strong> | <strong>Streamlit</strong> | <strong>Ollama</strong>
</div>
""", unsafe_allow_html=True)
