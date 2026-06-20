import sqlite3
import os
import json
from datetime import datetime

DB_PATH = "./data/knowledge_base.db"


def init_db():
    """Initialize all SQLite tables."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT UNIQUE,
            summary     TEXT,
            chunk_count INTEGER DEFAULT 0,
            indexed_at  TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS query_history (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            query               TEXT,
            answer              TEXT,
            faithfulness        REAL,
            answer_relevancy    REAL,
            context_precision   REAL,
            context_recall      REAL,
            created_at          TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_entries (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            category    TEXT,
            title       TEXT,
            content     TEXT,
            source      TEXT,
            created_at  TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("[SQLite MCP] Database initialized.")


def register_document(filename: str, summary: str, chunk_count: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO documents (filename, summary, chunk_count, indexed_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(filename) DO UPDATE SET
            summary=excluded.summary,
            chunk_count=excluded.chunk_count,
            indexed_at=excluded.indexed_at
    """, (filename, summary, chunk_count, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_all_documents() -> list:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, filename, summary, chunk_count, indexed_at FROM documents")
    rows = c.fetchall()
    conn.close()
    return [
        {"id": r[0], "filename": r[1], "summary": r[2],
         "chunk_count": r[3], "indexed_at": r[4]}
        for r in rows
    ]


def save_query(query: str, answer: str, scores: dict = None):
    scores = scores or {}
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO query_history
        (query, answer, faithfulness, answer_relevancy,
         context_precision, context_recall, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        query, answer,
        scores.get("faithfulness"),
        scores.get("answer_relevancy"),
        scores.get("context_precision"),
        scores.get("context_recall"),
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()


def get_recent_queries(limit: int = 10) -> list:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT query, answer, created_at
        FROM query_history
        ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    return [{"query": r[0], "answer": r[1], "created_at": r[2]} for r in rows]


def search_query_history(keyword: str) -> list:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT query, answer, created_at
        FROM query_history
        WHERE query LIKE ? OR answer LIKE ?
        ORDER BY id DESC LIMIT 5
    """, (f"%{keyword}%", f"%{keyword}%"))
    rows = c.fetchall()
    conn.close()
    return [{"query": r[0], "answer": r[1], "created_at": r[2]} for r in rows]


def add_knowledge_entry(category: str, title: str, content: str, source: str = ""):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO knowledge_entries
        (category, title, content, source, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (category, title, content, source, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def search_knowledge_entries(keyword: str) -> list:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT category, title, content, source
        FROM knowledge_entries
        WHERE title LIKE ? OR content LIKE ? OR category LIKE ?
        ORDER BY id DESC LIMIT 5
    """, (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"))
    rows = c.fetchall()
    conn.close()
    return [
        {"category": r[0], "title": r[1], "content": r[2], "source": r[3]}
        for r in rows
    ]


def get_stats() -> dict:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM documents")
    doc_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM query_history")
    query_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM knowledge_entries")
    entry_count = c.fetchone()[0]
    c.execute("""
        SELECT
            AVG(faithfulness),
            AVG(answer_relevancy),
            AVG(context_precision),
            AVG(context_recall)
        FROM query_history
        WHERE faithfulness IS NOT NULL
    """)
    avg = c.fetchone()
    conn.close()
    return {
        "total_documents": doc_count,
        "total_queries": query_count,
        "total_entries": entry_count,
        "avg_faithfulness": round(avg[0], 3) if avg[0] else None,
        "avg_answer_relevancy": round(avg[1], 3) if avg[1] else None,
        "avg_context_precision": round(avg[2], 3) if avg[2] else None,
        "avg_context_recall": round(avg[3], 3) if avg[3] else None,
    }