#import packages

import sqlite3
from pathlib import Path
import bcrypt

#set path of db
DB_PATH = Path(__file__).resolve().parent / "users.db"

#open db
def get_connection():
    return sqlite3.connect(DB_PATH)

#creates users table 
def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash BLOB NOT NULL
        )
    """)
    conn.commit()
    conn.close()

#stores new user with crypted password, if username doesn't already exist in db
def create_user(username, password):
    conn = get_connection()
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    try:
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

#checks validity of login attempt
def authenticate_user(username, password):
    conn = get_connection()
    row = conn.execute(
        "SELECT password_hash FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    conn.close()

    if row is None:
        return False

    return bcrypt.checkpw(password.encode(), row[0])
