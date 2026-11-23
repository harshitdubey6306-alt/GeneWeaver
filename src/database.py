import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("data/geneweaver.db")


def get_connection():
    """Creates the data directory if missing and returns a SQLite connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the alignment_runs table for tracking CRISPR jobs and metrics."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alignment_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                target_crispr TEXT NOT NULL,
                genome_length INTEGER NOT NULL,
                max_mismatches INTEGER NOT NULL,
                backend TEXT NOT NULL,
                execution_ms REAL NOT NULL,
                match_count INTEGER NOT NULL,
                match_indices TEXT NOT NULL
            )
        """)
        conn.commit()


def log_alignment_run(target_crispr, genome_length, max_mismatches, backend, execution_ms, match_indices):
    """Inserts a completed CRISPR alignment record and returns its row ID."""
    init_db()
    timestamp = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO alignment_runs (
                created_at, target_crispr, genome_length, max_mismatches,
                backend, execution_ms, match_count, match_indices
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                target_crispr,
                genome_length,
                max_mismatches,
                backend,
                round(execution_ms, 4),
                len(match_indices),
                json.dumps(match_indices),
            ),
        )
        conn.commit()
        return cursor.lastrowid


def fetch_recent_runs(limit=10):
    """Retrieves the most recent alignment job records from the database."""
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM alignment_runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


if __name__ == "__main__":
    init_db()
    run_id = log_alignment_run("CGATCGATCG", 5000, 1, "CPU Fallback (NumPy)", 1.42, [120, 845])
    print(f"Database initialized and test run logged with ID: {run_id}")