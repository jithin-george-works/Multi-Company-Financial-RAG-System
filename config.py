from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INDICES_DIR = Path(os.getenv("FAISS_INDEX_DIR", BASE_DIR / "indices"))
SENTENCE_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")

# DB names / folder names
INDEX_A = INDICES_DIR / "faiss_companyA"
INDEX_B = INDICES_DIR / "faiss_companyB"

# Chunker params
CHUNK_MAX_CHARACTERS = 1200
CHUNK_NEW_AFTER_N_CHARS = 600
CHUNK_COMBINE_UNDER_N_CHARS = 200
