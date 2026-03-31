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
                name TEXT PRIMARY KEY,
                config TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tweets (
                id TEXT NOT NULL,
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
                PRIMARY KEY (id, project)
            )
        """)
        conn.commit()


def get_projects() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT name, config FROM projects").fetchall()
    return [{"name": r[0], **json.loads(r[1])} for r in rows]


def save_project(config: dict):
    name = config["name"]
    data = json.dumps({k: v for k, v in config.items() if k != "name"})
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO projects (name, config) VALUES (?, ?)",
            (name, data)
        )
        conn.commit()


def load_project_config(name: str) -> dict:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT config FROM projects WHERE name = ?", (name,)
        ).fetchone()
    if not row:
        return {"keywords": [], "handles": [], "categories": []}
    return json.loads(row[0])


def save_tweets(project: str, tweets: list[dict], update: bool = False):
    if not tweets:
        return
    with get_conn() as conn:
        for t in tweets:
            if update:
                conn.execute("""
                    UPDATE tweets SET sentiment=?, category=?, score=?, reasoning=?
                    WHERE id=? AND project=?
                """, (
                    t.get("sentiment"), t.get("category"),
                    t.get("score"), t.get("reasoning"),
                    str(t["id"]), project
                ))
            else:
                conn.execute("""
                    INSERT OR IGNORE INTO tweets
                    (id, project, created_at, author_id, username, name, text, lang,
                     retweet_count, like_count, reply_count, quote_count,
                     sentiment, category, score, reasoning)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    str(t["id"]), project, t.get("created_at"), t.get("author_id"),
                    t.get("username"), t.get("name"), t.get("text"), t.get("lang"),
                    t.get("retweet_count", 0), t.get("like_count", 0),
                    t.get("reply_count", 0), t.get("quote_count", 0),
                    t.get("sentiment"), t.get("category"),
                    t.get("score"), t.get("reasoning")
                ))
        conn.commit()


def load_tweets(project: str) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM tweets WHERE project = ? ORDER BY created_at DESC",
            conn, params=(project,)
        )
    return df
