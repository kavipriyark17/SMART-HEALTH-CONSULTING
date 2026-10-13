"""SQLite data layer for the Smart Health Consulting app.

All queries are parameterized. Every consultation query also filters by
user_id, so one user can never read or delete another user's records.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def get_db_path() -> Path:
    """Return the database path (configurable through the DB_PATH env var)."""
    raw = os.getenv("DB_PATH", "data/health_app.db")
    path = Path(raw)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


@contextmanager
def get_connection():
    """Open a connection, commit on success, roll back on error, always close."""
    path = get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they do not exist yet."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash BLOB NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS consultations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                age_group TEXT,
                duration TEXT,
                symptoms TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_consultations_user "
            "ON consultations (user_id, created_at)"
        )


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------------------------------------------- users
def create_user(username: str, email: str, password_hash: bytes) -> bool:
    """Insert a user. Returns False if the username or email already exists."""
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO users (username, email, password_hash, created_at) "
                "VALUES (?, ?, ?, ?)",
                (username, email, password_hash, _now()),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def get_user_by_username(username: str) -> Optional[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT id, username, email, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()


# --------------------------------------------------------- consultations
def add_consultation(
    user_id: int, symptoms: str, ai_response: str, age_group: str, duration: str
) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO consultations "
            "(user_id, created_at, age_group, duration, symptoms, ai_response) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, _now(), age_group, duration, symptoms, ai_response),
        )
        return cur.lastrowid


def list_consultations(user_id: int) -> list:
    """All consultations of ONE user, newest first."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT id, created_at, age_group, duration, symptoms, ai_response "
            "FROM consultations WHERE user_id = ? ORDER BY id DESC",
            (user_id,),
        ).fetchall()


def get_consultation(user_id: int, consultation_id: int) -> Optional[sqlite3.Row]:
    """Fetch one record only if it belongs to user_id (otherwise None)."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT id, created_at, age_group, duration, symptoms, ai_response "
            "FROM consultations WHERE id = ? AND user_id = ?",
            (consultation_id, user_id),
        ).fetchone()


def delete_consultation(user_id: int, consultation_id: int) -> bool:
    """Delete one record only if it belongs to user_id."""
    with get_connection() as conn:
        cur = conn.execute(
            "DELETE FROM consultations WHERE id = ? AND user_id = ?",
            (consultation_id, user_id),
        )
        return cur.rowcount > 0


def delete_all_consultations(user_id: int) -> int:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM consultations WHERE user_id = ?", (user_id,))
        return cur.rowcount


def consultations_dataframe(user_id: int) -> pd.DataFrame:
    """History as a Pandas DataFrame (used for the table and CSV export)."""
    rows = list_consultations(user_id)
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return pd.DataFrame(
            columns=["id", "created_at", "age_group", "duration", "symptoms", "ai_response"]
        )
    return df
