"""
storage.py — Postgres-backed persistence for the Tweet Sentiment Tool.

Connection is via DATABASE_URL in st.secrets (a standard Postgres DSN).
Example secrets.toml entry:
    DATABASE_URL = "postgresql://user:password@host:5432/dbname"

On Supabase: use the "Session mode" connection string from
Settings → Database → Connection string → URI (port 5432).
"""

import json
import pandas as pd
import streamlit as st
import psycopg2
import psycopg2.extras
from contextlib import contextmanager


# ── Connection ────────────────────────────────────────────────────────────────

@contextmanager
def _get_conn():
    """Yield a psycopg2 connection from the DATABASE_URL secret."""
    url = st.secrets.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError(
            "DATABASE_URL not set in secrets.toml. "
            "Add: DATABASE_URL = \"postgresql://user:pass@host:5432/db\""
        )
    conn = psycopg2.connect(url, sslmode="require")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db():
    """Create tables if they don't exist. Safe to call on every startup."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    owner  TEXT NOT NULL,
                    name   TEXT NOT NULL,
                    config JSONB NOT NULL DEFAULT '{}',
                    PRIMARY KEY (owner, name)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tweets (
                    id             TEXT    NOT NULL,
                    owner          TEXT    NOT NULL,
                    project        TEXT    NOT NULL,
                    created_at     TEXT,
                    author_id      TEXT,
                    username       TEXT,
                    name           TEXT,
                    text           TEXT,
                    lang           TEXT,
                    retweet_count  INTEGER DEFAULT 0,
                    like_count     INTEGER DEFAULT 0,
                    reply_count    INTEGER DEFAULT 0,
                    quote_count    INTEGER DEFAULT 0,
                    sentiment      TEXT,
                    category       TEXT,
                    score          REAL,
                    reasoning      TEXT,
                    PRIMARY KEY (id, owner, project)
                )
            """)
            # Index for fast per-user lookups
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_tweets_owner_project
                ON tweets (owner, project)
            """)


# ── Projects ──────────────────────────────────────────────────────────────────

def get_projects(owner: str) -> list[dict]:
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT name, config FROM projects WHERE owner = %s ORDER BY name",
                (owner,)
            )
            rows = cur.fetchall()
    return [{"name": r["name"], **r["config"]} for r in rows]


def save_project(config: dict, owner: str):
    name = config["name"]
    data = {k: v for k, v in config.items() if k != "name"}
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO projects (owner, name, config)
                VALUES (%s, %s, %s)
                ON CONFLICT (owner, name) DO UPDATE SET config = EXCLUDED.config
            """, (owner, name, json.dumps(data)))


def load_project_config(name: str, owner: str) -> dict:
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT config FROM projects WHERE owner = %s AND name = %s",
                (owner, name)
            )
            row = cur.fetchone()
    if not row:
        return {"keywords": [], "handles": [], "categories": []}
    return row["config"]


# ── Tweets ────────────────────────────────────────────────────────────────────

def save_tweets(project: str, owner: str, tweets: list[dict], update: bool = False):
    if not tweets:
        return
    with _get_conn() as conn:
        with conn.cursor() as cur:
            for t in tweets:
                if update:
                    cur.execute("""
                        UPDATE tweets
                        SET sentiment = %s, category = %s, score = %s, reasoning = %s
                        WHERE id = %s AND owner = %s AND project = %s
                    """, (
                        t.get("sentiment"), t.get("category"),
                        t.get("score"), t.get("reasoning"),
                        str(t["id"]), owner, project
                    ))
                else:
                    cur.execute("""
                        INSERT INTO tweets
                            (id, owner, project, created_at, author_id, username, name,
                             text, lang, retweet_count, like_count, reply_count,
                             quote_count, sentiment, category, score, reasoning)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (id, owner, project) DO NOTHING
                    """, (
                        str(t["id"]), owner, project,
                        t.get("created_at"), t.get("author_id"),
                        t.get("username"), t.get("name"),
                        t.get("text"), t.get("lang"),
                        t.get("retweet_count", 0), t.get("like_count", 0),
                        t.get("reply_count", 0), t.get("quote_count", 0),
                        t.get("sentiment"), t.get("category"),
                        t.get("score"), t.get("reasoning")
                    ))


def load_tweets(project: str, owner: str) -> pd.DataFrame:
    with _get_conn() as conn:
        df = pd.read_sql(
            "SELECT * FROM tweets WHERE owner = %s AND project = %s ORDER BY created_at DESC",
            conn, params=(owner, project)
        )
    return df