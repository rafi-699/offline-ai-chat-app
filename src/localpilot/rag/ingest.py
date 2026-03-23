from __future__ import annotations

import logging
from pathlib import Path
from typing import List

log = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[str]:
    """Split *text* into overlapping character-based chunks."""
    if not text.strip():
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += chunk_size - overlap
    return chunks


def ingest_text(
    text: str,
    doc_name: str,
    index,  # RAGIndex
    chunk_size: int = 512,
    overlap: int = 64,
) -> int:
    """Chunk and add *text* to the RAG index. Returns number of chunks added."""
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        log.warning("No chunks produced for document '%s'.", doc_name)
        return 0
    added = index.add_chunks(doc_name, chunks)
    log.info("Ingested %d chunks for '%s'.", added, doc_name)
    return added
