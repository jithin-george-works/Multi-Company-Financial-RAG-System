# ingest.py
from langchain_core.documents import Document
from typing import List, Dict, Any

def chunks_to_documents(chunks: List[Dict[str, Any]]):
    docs = []
    for c in chunks:
        docs.append(Document(page_content=c["content"], metadata=c["metadata"]))
    return docs
