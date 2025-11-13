
````markdown
# Multi-Company Financial RAG - Usage Guide

This document provides a step-by-step guide to using the **Multi-Company Financial RAG PoC**.

---

## 1. Setup Environment

### 1.1 Install Ollama and Pull Model
Make sure Ollama is installed and the model is available locally:

```bash
# Install Ollama
https://ollama.ai/download

# Pull LLM model
ollama pull llama3:latest
````

### 1.2 Create Python Virtual Environment

```bash
# Create venv
python -m venv .venv

# Activate venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Prepare Company Data

* Place your company HTML files in the folder:

```
multi-company-rag/data/company_html/
```

* Example file names:

```
company_a.html
company_b.html
```

---

## 3. Build FAISS Indices

Run the ingestion script to parse, chunk, and index the documents:

```bash
python build_db.py
```

This will:

* Parse HTML files.
* Chunk content using Unstructured.
* Build **separate FAISS indices per company**.

> ⚠️ Re-run `build_db.py` whenever you add new company files.

---

## 4. Launch Streamlit UI

Start the interactive UI:

```bash
streamlit run app.py
```

* Open the browser URL printed by Streamlit (usually `http://127.0.0.1:8501`).
* Type a question in the input box.
* The system will:

  * Decide which company to query using LLM-based routing.
  * Retrieve relevant chunks from the FAISS index.
  * Generate a context-aware answer.
* Answers will stream live when supported by the model.

---

## 5. Example Queries

| Query                                             | Expected Behavior                                                            |
| ------------------------------------------------- | ---------------------------------------------------------------------------- |
| `What was the revenue of Company A last quarter?` | Routes to Company A index, retrieves relevant context, and generates answer. |
| `List top 5 expenses for Company B.`              | Routes to Company B index and provides extracted information.                |
| `Compare the net profit of both companies.`       | Retrieves info from both indices and provides a comparative answer.          |

---

## 6. Directory Reference

```
multi-company-rag/
├─ src/
│  └─ multi_company_rag/
│     ├─ __init__.py
│     ├─ agent.py
│     ├─ chunker.py
│     ├─ ingest.py
│     └─ config.py
├─ data/
│  └─ company_html/
├─ tests/
├─ docs/
├─ scripts/
├─ app.py
├─ build_db.py
├─ requirements.txt
└─ README.md
```

---

## 7. Tips & Notes

* Ensure **Ollama** is running locally before querying.
* For adding more companies, simply add HTML files and re-run `build_db.py`.
* FAISS indices are **separate per company**, allowing independent updates.
* Streamlit UI supports **streaming output** for long answers.
* Use consistent naming in `data/company_html/` for easier management.

---

## 8. References

* [Ollama Docs](https://ollama.ai/docs)
* [FAISS](https://github.com/facebookresearch/faiss)
* [Streamlit](https://streamlit.io)
* [Unstructured Parser](https://unstructured-io.github.io/unstructured/)
* [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)

---




