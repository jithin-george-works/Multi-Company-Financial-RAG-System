# app.py
import streamlit as st
from agent import answer_question
from config import DATA_DIR
import time

st.set_page_config(page_title="Multi-Company RAG (PoC)", layout="wide")

st.title("Multi-Company RAG — PoC")
st.markdown("Ask about Company A, Company B, or compare both. The agent decides which data to use (LLM-based routing).")

with st.sidebar:
    st.header("Sample queries")
    st.write("""
    - What is the Q2 revenue of Company A?
    - Summarize Company B's ESG initiatives.
    - Compare the yearly revenue of Company A and Company B.
    - Who is the CEO of Company A?
    - What was Company B's profit margin in Q4?
    """)
    st.write("You can edit the sample files in `data/` and re-run `python build_db.py`")

query = st.text_area("Enter your question", height=120)
run_btn = st.button("Ask")

status = st.empty()
output_area = st.empty()

if run_btn and query.strip():
    status.info("Routing query to LLM to decide which company to query...")
    route, context, token_stream = answer_question(query)
    status.success(f"Agent decided route: {route}")

    # Show retrieved context (collapsible)
    with st.expander("Retrieved context (preview)"):
        st.write(context[:4000] if context else "No context retrieved.")

    # Stream the answer
    output_placeholder = output_area.empty()
    output_placeholder.markdown("**Answer (streaming):**\n")
    final_text = ""
    if token_stream is not None:
        try:
            for chunk in token_stream:
                # chunk may be multi-char chunk; append and update UI
                final_text += chunk
                output_placeholder.markdown(f"**Answer (streaming):**\n\n{final_text}")
                # Optional small sleep to allow UI to update smoothly
                time.sleep(0.02)
        except Exception as e:
            # If streaming or generator fails, display error and fallback
            st.error(f"Streaming error: {e}")
            # fallback: call synchronously and display
            _, _, sync_gen = answer_question(query)
            final_text = "".join(list(sync_gen))
            output_placeholder.markdown(f"**Answer:**\n\n{final_text}")
    else:
        st.write("No token stream available.")
