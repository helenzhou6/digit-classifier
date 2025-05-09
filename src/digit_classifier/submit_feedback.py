import psycopg2
import os
from dotenv import load_dotenv

def _connect_to_db():
    load_dotenv()
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
    POSTGRES_USERNAME = os.getenv('POSTGRES_USERNAME')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    conn = psycopg2.connect(f"user={POSTGRES_USERNAME} password={POSTGRES_PASSWORD} port={DB_PORT} host={DB_HOST}")
    return conn

def _close_db_connection(conn):
    conn.commit()
    conn.close()

def add_feedback_record():
    conn =  _connect_to_db()
    cur = conn.cursor()
    cur.execute(f"INSERT INTO feedback (timestamp, predicted_digit, true_digit, conf_percent) VALUES (NOW(), '3', '4', '80.0');")
    _close_db_connection(conn)

