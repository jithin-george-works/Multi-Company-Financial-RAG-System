# build_db.py
from config import DATA_DIR, INDEX_A, INDEX_B, SENTENCE_MODEL
from chunker import HTMLChunker
from ingest import chunks_to_documents
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_indices(company_a_html: Path, company_b_html: Path):
    chunker = HTMLChunker()
    chunks_a = chunker.chunk_html(company_a_html, "CompanyA")
    chunks_b = chunker.chunk_html(company_b_html, "CompanyB")

    docs_a = chunks_to_documents(chunks_a)
    docs_b = chunks_to_documents(chunks_b)

    embeddings = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{SENTENCE_MODEL}")

    logger.info("Building FAISS index for Company A...")
    db_a = FAISS.from_documents(docs_a, embeddings)
    logger.info("Building FAISS index for Company B...")
    db_b = FAISS.from_documents(docs_b, embeddings)

    INDEX_A.parent.mkdir(parents=True, exist_ok=True)
    INDEX_B.parent.mkdir(parents=True, exist_ok=True)

    db_a.save_local(str(INDEX_A))
    db_b.save_local(str(INDEX_B))
    logger.info("Saved FAISS indices to disk.")

if __name__ == "__main__":
    # default paths (assumes files exist)
    a = Path(DATA_DIR) / "companyA.html"
    b = Path(DATA_DIR) / "companyB.html"
    build_indices(a, b)
