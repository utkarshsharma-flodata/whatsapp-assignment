"""SQLite connection handling."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from core.config import get_settings
from database.schema import migrate


def open_connection(path: str | None = None) -> sqlite3.Connection:
    """Open a connection with foreign keys enabled and the schema migrated.

    Uses explicit transaction control (autocommit mode); repository code
    groups writes with ``transaction()``.
    """
    path = path or get_settings().database_path
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10, autocommit=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if path != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    migrate(conn)
    return conn


@contextmanager
def connect(path: str | None = None) -> Iterator[sqlite3.Connection]:
    """Open a connection for the duration of a ``with`` block, then close it."""
    conn = open_connection(path)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run the enclosed writes atomically. Nested calls join the outer transaction."""
    if conn.in_transaction:
        yield conn
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
