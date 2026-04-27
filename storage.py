"""
storage.py — Postgres-backed persistence for the X Sentiment Tool.

Database credentials are loaded from AWS Secrets Manager at startup.
The ECS task role grants access — no hardcoded credentials needed.

Secret name : soa-tools/xsentiment/db
Secret value: {"url": "postgresql://user:password@host:5432/dbname"}
"""

import json
import os
import boto3
import pandas as pd
import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from functools import lru_cache


# ── Connection ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_database_url() -> str:
    """
    Fetch the DATABASE_URL from Secrets Manager once and cache it.
    Falls back to the DATABASE_URL environment variable for local development.
    """
    # Local dev override
    local_url = os.environ.get("DATABASE_URL", "")
    if local_url:
        return local_url

    client = boto3.client("secretsmanager", region_name="us-east-2")
    resp   = client.get_secret_value(SecretId="soa-tools/xsentiment/db")
    secret = json.loads(resp["SecretString"])
    return secret["url"]


@contextmanager
def _get_conn():
    """Yield a psycopg2 connection using the DATABASE_URL from Secrets Manager."""
    url  = _get_database_url()
    conn = psycopg2.connect(url, sslmode="require", options="-c statement_timeout=30000")
    conn.autocommit = False
    psycopg2.extras.register_default_jsonb(conn)
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
