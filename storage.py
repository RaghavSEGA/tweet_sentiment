import sqlite3
import json
import pandas as pd
from pathlib import Path

DB_PATH = Path("sentiment_tool.db")


def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                owner TEXT NOT NULL,
                name  TEXT NOT NULL,
                config TEXT NOT NULL,
                PRIMARY KEY (owner, name)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tweets (
                id TEXT NOT NULL,
                owner TEXT NOT NULL,
                project TEXT NOT NULL,
                created_at TEXT,
                author_id TEXT,
                username TEXT,
                name TEXT,
                text TEXT,
                lang TEXT,
                retweet_count INTEGER,
                like_count INTEGER,
                reply_count INTEGER,
                quote_count INTEGER,
                sentiment TEXT,
                category TEXT,
                score REAL,
                reasoning TEXT,
                PRIMARY KEY (id, owner, project)
            )
        """)
        _migrate_legacy(conn)
        conn.commit()


def _migrate_legacy(conn):
    """One-time migration: add owner column to old single-owner tables if absent."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(projects)").fetchall()}
    if "owner" not in cols:
        conn.execute("ALTER TABLE projects ADD COLUMN owner TEXT NOT NULL DEFAULT ''")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(tweets)").fetchall()}
    if "owner" not in cols:
        conn.execute("ALTER TABLE tweets ADD COLUMN owner TEXT NOT NULL DEFAULT ''")


def get_projects(owner: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT name, config FROM projects WHERE owner = ? ORDER BY name",
            (owner,)
        ).fetchall()
    return [{"name": r[0], **json.loads(r[1])} for r in rows]


def save_project(config: dict, owner: str):
    name = config["name"]
    data = json.dumps({k: v for k, v in config.items() if k != "name"})
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO projects (owner, name, config) VALUES (?, ?, ?)",
            (owner, name, data)
        )
        conn.commit()


def load_project_config(name: str, owner: str) -> dict:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT config FROM projects WHERE owner = ? AND name = ?",
            (owner, name)
        ).fetchone()
    if not row:
        return {"keywords": [], "handles": [], "categories": []}
    return json.loads(row[0])


def save_tweets(project: str, owner: str, tweets: list[dict], update: bool = False):
    if not tweets:
        return
    with get_conn() as conn:
        for t in tweets:
            if update:
                conn.execute("""
                    UPDATE tweets SET sentiment=?, category=?, score=?, reasoning=?
                    WHERE id=? AND owner=? AND project=?
                """, (
                    t.get("sentiment"), t.get("category"),
                    t.get("score"), t.get("reasoning"),
                    str(t["id"]), owner, project
                ))
            else:
                conn.execute("""
                    INSERT OR IGNORE INTO tweets
                    (id, owner, project, created_at, author_id, username, name, text, lang,
                     retweet_count, like_count, reply_count, quote_count,
                     sentiment, category, score, reasoning)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    str(t["id"]), owner, project,
                    t.get("created_at"), t.get("author_id"),
                    t.get("username"), t.get("name"), t.get("text"), t.get("lang"),
                    t.get("retweet_count", 0), t.get("like_count", 0),
                    t.get("reply_count", 0), t.get("quote_count", 0),
                    t.get("sentiment"), t.get("category"),
                    t.get("score"), t.get("reasoning")
                ))
        conn.commit()


def load_tweets(project: str, owner: str) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM tweets WHERE owner = ? AND project = ? ORDER BY created_at DESC",
            conn, params=(owner, project)
        )
    return df