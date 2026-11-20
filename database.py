"""
database.py - PostgreSQL storage for the face authorization app.

Multi-user version: every person and detection can belong to a user (owner_id).
Calling a function without owner_id uses owner_id = NULL, which is the old
single-user data, so the Streamlit prototype keeps working unchanged.

Connection settings come from a .env file in the project folder:
    DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/face_auth

Requires PostgreSQL 15 or newer (uses UNIQUE ... NULLS NOT DISTINCT).
"""

import os

import numpy as np
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


def get_conn():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. Create a .env file in the project folder with:\n"
            "DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/face_auth"
        )
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def init_db():
    """Create tables, and upgrade an older single-user database in place."""
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id             SERIAL PRIMARY KEY,
                email          TEXT NOT NULL UNIQUE,
                password_hash  TEXT NOT NULL,
                created_at     TIMESTAMP NOT NULL DEFAULT now()
            );

            CREATE TABLE IF NOT EXISTS people (
                id          SERIAL PRIMARY KEY,
                owner_id    INTEGER REFERENCES users(id) ON DELETE CASCADE,
                name        TEXT NOT NULL,
                photo_path  TEXT,
                created_at  TIMESTAMP NOT NULL DEFAULT now()
            );

            CREATE TABLE IF NOT EXISTS encodings (
                id          SERIAL PRIMARY KEY,
                person_id   INTEGER NOT NULL
                            REFERENCES people(id) ON DELETE CASCADE,
                encoding    BYTEA NOT NULL
            );

            CREATE TABLE IF NOT EXISTS detections (
                id           SERIAL PRIMARY KEY,
                owner_id     INTEGER REFERENCES users(id) ON DELETE CASCADE,
                detected_at  TIMESTAMP NOT NULL DEFAULT now(),
                name         TEXT NOT NULL,
                authorized   BOOLEAN NOT NULL
            );

            -- upgrade path for databases created before multi-user support
            ALTER TABLE people     ADD COLUMN IF NOT EXISTS owner_id INTEGER
                REFERENCES users(id) ON DELETE CASCADE;
            ALTER TABLE detections ADD COLUMN IF NOT EXISTS owner_id INTEGER
                REFERENCES users(id) ON DELETE CASCADE;
            ALTER TABLE people DROP CONSTRAINT IF EXISTS people_name_key;

            -- a name must be unique per user (case-insensitive), NULL owner counts too
            CREATE UNIQUE INDEX IF NOT EXISTS people_owner_name_key
                ON people (owner_id, lower(name)) NULLS NOT DISTINCT;

            CREATE INDEX IF NOT EXISTS people_owner_idx ON people (owner_id);
            CREATE INDEX IF NOT EXISTS detections_owner_idx ON detections (owner_id);
            """
        )


# ---------- users ----------

def create_user(email, password_hash):
    """Create a user. Returns the new user id. Raises ValueError if the email exists."""
    email = email.strip().lower()
    if not email:
        raise ValueError("Email cannot be empty.")
    try:
        with get_conn() as conn:
            row = conn.execute(
                "INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id",
                (email, password_hash),
            ).fetchone()
            return row["id"]
    except psycopg.errors.UniqueViolation:
        raise ValueError("That email is already registered.")


def get_user_by_email(email):
    with get_conn() as conn:
        return conn.execute(
            "SELECT id, email, password_hash, created_at FROM users WHERE email = %s",
            (email.strip().lower(),),
        ).fetchone()


def get_user_by_id(user_id):
    with get_conn() as conn:
        return conn.execute(
            "SELECT id, email, created_at FROM users WHERE id = %s", (user_id,)
        ).fetchone()


def delete_user(user_id):
    """Delete a user and ALL their data (people, encodings, detections)."""
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM users WHERE id = %s", (user_id,))
        return cur.rowcount > 0


# ---------- people ----------

def add_person(name, encoding, photo_path=None, owner_id=None):
    """Create a person with their first encoding. Returns the new person id.
    Raises ValueError if this owner already has someone with that name."""
    name = name.strip()
    if not name:
        raise ValueError("Name cannot be empty.")
    try:
        with get_conn() as conn:
            row = conn.execute(
                "INSERT INTO people (owner_id, name, photo_path) "
                "VALUES (%s, %s, %s) RETURNING id",
                (owner_id, name, photo_path),
            ).fetchone()
            person_id = row["id"]
            conn.execute(
                "INSERT INTO encodings (person_id, encoding) VALUES (%s, %s)",
                (person_id, np.asarray(encoding, dtype=np.float64).tobytes()),
            )
            return person_id
    except psycopg.errors.UniqueViolation:
        raise ValueError(f"'{name}' already exists.")


def add_encoding(person_id, encoding, owner_id=None):
    """Add another photo's encoding to an existing person.
    Raises ValueError if that person doesn't belong to owner_id."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM people WHERE id = %s AND owner_id IS NOT DISTINCT FROM %s",
            (person_id, owner_id),
        ).fetchone()
        if not row:
            raise ValueError("Person not found.")
        conn.execute(
            "INSERT INTO encodings (person_id, encoding) VALUES (%s, %s)",
            (person_id, np.asarray(encoding, dtype=np.float64).tobytes()),
        )


def remove_person(person_id, owner_id=None):
    """Delete a person and their encodings. Returns True if something was deleted
    (False if the person doesn't exist or belongs to someone else)."""
    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM people WHERE id = %s AND owner_id IS NOT DISTINCT FROM %s",
            (person_id, owner_id),
        )
        return cur.rowcount > 0


def list_people(owner_id=None):
    """Returns a list of dicts: id, name, photo_path, created_at, photo_count."""
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT p.id, p.name, p.photo_path, p.created_at,
                   COUNT(e.id) AS photo_count
            FROM people p
            LEFT JOIN encodings e ON e.person_id = p.id
            WHERE p.owner_id IS NOT DISTINCT FROM %s
            GROUP BY p.id
            ORDER BY p.name
            """,
            (owner_id,),
        ).fetchall()


def load_known_faces(owner_id=None):
    """Returns (encodings, names): two parallel lists for the recognizer."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT p.name, e.encoding
            FROM encodings e JOIN people p ON p.id = e.person_id
            WHERE p.owner_id IS NOT DISTINCT FROM %s
            """,
            (owner_id,),
        ).fetchall()
    encodings = [np.frombuffer(bytes(r["encoding"]), dtype=np.float64) for r in rows]
    names = [r["name"] for r in rows]
    return encodings, names


# ---------- detection log ----------

def log_detection(name, authorized, owner_id=None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO detections (owner_id, name, authorized) VALUES (%s, %s, %s)",
            (owner_id, name, bool(authorized)),
        )


def recent_detections(limit=20, owner_id=None):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT detected_at, name, authorized FROM detections "
            "WHERE owner_id IS NOT DISTINCT FROM %s ORDER BY id DESC LIMIT %s",
            (owner_id, limit),
        ).fetchall()
    return [
        {
            "timestamp": r["detected_at"].strftime("%Y-%m-%d %H:%M:%S"),
            "name": r["name"],
            "authorized": r["authorized"],
        }
        for r in rows
    ]


if __name__ == "__main__":
    try:
        init_db()
        print("PostgreSQL database ready.")
    except Exception as e:
        print("Could not connect or set up the database:\n", e)