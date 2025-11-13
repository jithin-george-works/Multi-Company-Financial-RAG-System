# Multi-Company Financial RAG (PoC)

## Summary
This repository demonstrates a local **multi-company Retrieval-Augmented Generation (RAG)** system that:

- Parses HTML files for 2 companies.
- Chunks the content using **Unstructured**.
- Builds **FAISS indices** (separate per company).
- Uses a local LLM (`Ollama / llama3:latest`) to decide which company to query (**LLM-based routing**).
- Retrieves relevant contexts and generates answers.
- Provides a small **Streamlit**
