import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_USERNAME = os.getenv('POSTGRES_USERNAME')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_CONNECTION_STRING=f"user={POSTGRES_USERNAME} password={POSTGRES_PASSWORD} port={DB_PORT} host={DB_HOST}"

def connect_to_db():
    conn = psycopg2.connect(DB_CONNECTION_STRING)
    return conn

def close_db_connection(conn):
    conn.commit()
    conn.close()

def create_table_with_dummy_data():
    conn =  connect_to_db()
    cur = conn.cursor()
    cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                timestamp TIMESTAMP,
                predicted_digit BIGINT,
                true_digit BIGINT,
                conf_percent FLOAT
            )
            """)
    close_db_connection(conn)

def _drop_table(): # to drop the table when needed
    conn =  connect_to_db()
    cur = conn.cursor()
    cur.execute("DROP TABLE feedback")
    close_db_connection(conn)

create_table_with_dummy_data()