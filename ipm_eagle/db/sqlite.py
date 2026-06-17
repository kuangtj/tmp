import sqlite3
from pathlib import Path

DB_PATH = Path("data/db/ipm_eagle.sqlite")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
