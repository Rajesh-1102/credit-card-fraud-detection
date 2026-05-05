import os
import sqlite3
from datetime import datetime

from src.logger import get_logger

logger = get_logger(__name__)

os.makedirs("database", exist_ok=True)

DB_PATH = "database/fraud.db"


def create_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn


def create_table():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction INTEGER,
            probability REAL,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()

    logger.info("Database table ready.")


def insert_transaction(prediction, probability):
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO transactions (prediction, probability, timestamp)
        VALUES (?, ?, ?)
    """, (prediction, probability, datetime.now().isoformat()))

    conn.commit()
    conn.close()

    logger.info("Transaction saved to database.")