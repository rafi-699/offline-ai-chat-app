from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

log = logging.getLogger(__name__)

DB_FILE = "rag_meta.db"
TFIDF_FILE = "tfidf.joblib"
MATRIX_FILE = "tfidf_matrix.joblib"


class RAGIndex:
    """
    Offline TF-IDF based retrieval index persisted to disk.
    Metadata (doc_id, chunk_id, text, source) stored in SQLite.
    TF-IDF vectorizer + matrix persisted via joblib.
    """

    def __init__(self, data_dir: Path) -> None:
        self._dir = data_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db_path = data_dir / DB_FILE
        self._tfidf_path = data_dir / TFIDF_FILE
        self._matrix_path = data_dir / MATRIX_FILE
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._ids: List[int] = []
        self._init_db()
        self._load_index()

    # ── Database ──────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_name TEXT NOT NULL,
                    chunk_idx INTEGER NOT NULL,
                    text     TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )
                """
            )
            conn.commit()

    # ── Persist index ─────────────────────────────────────────────────────────

    def _load_index(self) -> None:
        if self._tfidf_path.exists() and self._matrix_path.exists():
            try:
                self._vectorizer = joblib.load(self._tfidf_path)
                self._matrix = joblib.load(self._matrix_path)
                with self._conn() as conn:
                    rows = conn.execute("SELECT id FROM chunks ORDER BY id").fetchall()
                    self._ids = [r["id"] for r in rows]
                log.info("RAG index loaded: %d chunks.", len(self._ids))
            except Exception as exc:
                log.warning("Failed to load RAG index (%s); starting fresh.", exc)
                self._vectorizer = None
                self._matrix = None
                self._ids = []

    def _save_index(self) -> None:
        if self._vectorizer is None or self._matrix is None:
            return
        joblib.dump(self._vectorizer, self._tfidf_path)
        joblib.dump(self._matrix, self._matrix_path)

    # ── Ingest ────────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        doc_name: str,
        chunks: List[str],
        metadata: dict | None = None,
    ) -> int:
        meta_str = json.dumps(metadata or {})
        with self._conn() as conn:
            for idx, text in enumerate(chunks):
                conn.execute(
                    "INSERT INTO chunks (doc_name, chunk_idx, text, metadata) VALUES (?, ?, ?, ?)",
                    (doc_name, idx, text, meta_str),
                )
            conn.commit()
        self._rebuild()
        return len(chunks)

    def _rebuild(self) -> None:
        with self._conn() as conn:
            rows = conn.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
        if not rows:
            return
        self._ids = [r["id"] for r in rows]
        texts = [r["text"] for r in rows]
        self._vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            max_features=50_000,
            ngram_range=(1, 2),
        )
        self._matrix = self._vectorizer.fit_transform(texts)
        self._save_index()
        log.info("RAG index rebuilt: %d chunks.", len(self._ids))

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, text: str, top_k: int = 4) -> List[Tuple[str, str, float]]:
        """Returns list of (doc_name, chunk_text, score)."""
        if self._vectorizer is None or self._matrix is None or not self._ids:
            return []
        q_vec = self._vectorizer.transform([text])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        results: List[Tuple[str, str, float]] = []
        with self._conn() as conn:
            for idx in top_indices:
                if scores[idx] < 1e-9:
                    continue
                chunk_id = self._ids[idx]
                row = conn.execute(
                    "SELECT doc_name, text FROM chunks WHERE id = ?", (chunk_id,)
                ).fetchone()
                if row:
                    results.append((row["doc_name"], row["text"], float(scores[idx])))
        return results

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._conn() as conn:
            total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            total_docs = conn.execute(
                "SELECT COUNT(DISTINCT doc_name) FROM chunks"
            ).fetchone()[0]
        return {"total_chunks": total_chunks, "total_docs": total_docs}

    def clear(self) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM chunks")
            conn.commit()
        self._vectorizer = None
        self._matrix = None
        self._ids = []
        for p in (self._tfidf_path, self._matrix_path):
            if p.exists():
                p.unlink()
        log.info("RAG index cleared.")
