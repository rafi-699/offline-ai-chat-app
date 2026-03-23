import pytest
from pathlib import Path
import tempfile

from localpilot.rag.index import RAGIndex
from localpilot.rag.ingest import chunk_text, ingest_text


def test_chunk_text_basic():
    text = "Hello world " * 100
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 50 + 10  # slight tolerance for strip


def test_chunk_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_rag_ingest_and_query(tmp_path):
    idx = RAGIndex(tmp_path)
    text = "Python is a high-level programming language known for its readability. " * 20
    n = ingest_text(text, "test_doc.txt", idx, chunk_size=128, overlap=16)
    assert n > 0
    results = idx.query("Python programming language", top_k=2)
    assert len(results) > 0
    assert results[0][2] > 0  # score > 0


def test_rag_clear(tmp_path):
    idx = RAGIndex(tmp_path)
    ingest_text("Some text about databases.", "doc.txt", idx)
    idx.clear()
    stats = idx.stats()
    assert stats["total_chunks"] == 0


def test_rag_stats(tmp_path):
    idx = RAGIndex(tmp_path)
    ingest_text("Document one content here.", "doc1.txt", idx)
    ingest_text("Document two content here.", "doc2.txt", idx)
    stats = idx.stats()
    assert stats["total_docs"] == 2
