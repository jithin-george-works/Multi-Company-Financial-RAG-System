# Multi-Company Financial RAG (PoC)

## Overview
This repository demonstrates a **local multi-company Retrieval-Augmented Generation (RAG) system**. It is designed to:

- Parse HTML files from multiple companies.
- Chunk documents using **Unstructured**.
- Build **FAISS** indices (separate per company).
- Use a **local LLM** (`Ollama / llama3:latest`) for **LLM-based routing** to select the relevant company.
- Retrieve relevant contexts and generate answers.
- Provide a small **Streamlit UI** with streaming output.

This PoC allows you to experiment with multi-company RAG pipelines locally without external APIs.

---

