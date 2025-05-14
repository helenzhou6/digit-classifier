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