"""Create (or verify) the local SQLite schema. Idempotent — safe to re-run.

Usage:
    python -m src.db.init_db
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .. import config

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def get_connection() -> sqlite3.Connection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.SQLITE_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the table already existed on disk.
    CREATE TABLE IF NOT EXISTS in schema.sql is a no-op for an existing
    table, so new columns need an explicit, idempotent ALTER TABLE here."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(questions)")}
    if "formatted_json" not in existing:
        conn.execute("ALTER TABLE questions ADD COLUMN formatted_json TEXT")
        conn.commit()


def init_db() -> None:
    conn = get_connection()
    try:
        schema_sql = SCHEMA_PATH.read_text()
        conn.executescript(schema_sql)
        conn.commit()
        _migrate(conn)
        print(f"SQLite schema ready at {config.SQLITE_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
