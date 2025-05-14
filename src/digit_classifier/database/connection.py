import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_USERNAME = os.getenv('POSTGRES_USERNAME')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')

def connect_to_db():
    conn = psycopg2.connect(
        user=POSTGRES_USERNAME, password=POSTGRES_PASSWORD, host=DB_HOST, port=DB_PORT
    )

    return conn

def close_db_connection(conn):
    conn.commit()
    conn.close()