"""
Advanced HTML Chunking Module for Financial Documents
Uses Unstructured's `partition_html` + `chunk_by_title` for structure-aware chunking.
"""

import logging
from typing import List, Dict, Any
from pathlib import Path
from bs4 import BeautifulSoup
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element
import chardet

from config import (
    CHUNK_MAX_CHARACTERS,
    CHUNK_NEW_AFTER_N_CHARS,
    CHUNK_COMBINE_UNDER_N_CHARS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HTMLChunker:
    """Structure-aware HTML chunker for RAG preprocessing."""

    def __init__(
        self,
        max_characters: int = CHUNK_MAX_CHARACTERS,
        new_after_n_chars: int = CHUNK_NEW_AFTER_N_CHARS,
        combine_text_under_n_chars: int = CHUNK_COMBINE_UNDER_N_CHARS
    ):
        self.max_characters = max_characters
        self.new_after_n_chars = new_after_n_chars
        self.combine_text_under_n_chars = combine_text_under_n_chars

    def _detect_encoding(self, file_path: Path) -> str:
        """Safely detect file encoding for HTML with non-UTF8 chars."""
        with open(file_path, "rb") as f:
            raw_data = f.read(4096)
        result = chardet.detect(raw_data)
        encoding = result.get("encoding", "utf-8")
        logger.info(f"Detected encoding for {file_path.name}: {encoding}")
        return encoding

    def preprocess_html(self, html_file: Path) -> str:
        """Clean raw HTML: remove script, nav, footer, etc."""
        try:
            encoding = self._detect_encoding(html_file)
            with open(html_file, "r", encoding=encoding, errors="ignore") as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, "lxml")

            # Remove unwanted sections
            for tag in soup(["script", "style", "nav", "footer", "header", "meta", "link"]):
                tag.decompose()

            # Optional cleanup: remove empty tags
            for tag in soup.find_all():
                if not tag.text.strip():
                    tag.decompose()

            logger.info(f"Preprocessed HTML: {html_file.name}")
            return str(soup)

        except Exception as e:
            logger.error(f"Error preprocessing {html_file.name}: {e}")
            raise

    def chunk_html(self, html_file: Path, company_name: str) -> List[Dict[str, Any]]:
        """Chunk HTML file using Unstructured's by_title strategy."""
        try:
            cleaned_html = self.preprocess_html(html_file)

            # Partition and detect document elements
            elements = partition_html(text=cleaned_html, include_metadata=True)
            logger.info(f"Extracted {len(elements)} document elements from {html_file.name}")

            # Section-aware chunking
            chunks = chunk_by_title(
                elements,
                max_characters=self.max_characters,
                new_after_n_chars=self.new_after_n_chars,
                combine_text_under_n_chars=self.combine_text_under_n_chars
            )

            processed_chunks = []
            for idx, chunk in enumerate(chunks):
                processed_chunks.append({
                    "content": chunk.text.strip(),
                    "metadata": {
                        "company": company_name,
                        "chunk_id": idx,
                        "element_type": getattr(chunk, "category", "Unknown"),
                        "source_file": html_file.name,
                        "num_chars": len(chunk.text),
                        **(chunk.metadata.to_dict() if hasattr(chunk, "metadata") else {})
                    }
                })

            logger.info(f"Chunked {html_file.name}: {len(processed_chunks)} chunks created")
            self._log_chunk_stats(processed_chunks)
            return processed_chunks

        except Exception as e:
            logger.error(f"Error chunking {html_file.name}: {e}")
            raise

    def _log_chunk_stats(self, chunks: List[Dict[str, Any]]) -> None:
        """Log helpful chunk statistics."""
        total_chars = sum(len(c["content"]) for c in chunks)
        avg_chars = total_chars / len(chunks) if chunks else 0
        element_types = {}
        for chunk in chunks:
            etype = chunk["metadata"].get("element_type", "Unknown")
            element_types[etype] = element_types.get(etype, 0) + 1
        logger.info(
            f"Chunk Stats → Total: {len(chunks)}, Avg length: {avg_chars:.1f}, "
            f"Types: {element_types}"
        )


def create_chunks_for_company(html_file: Path, company_name: str) -> List[Dict[str, Any]]:
    """Convenience function for quick chunk creation."""
    return HTMLChunker().chunk_html(html_file, company_name)
