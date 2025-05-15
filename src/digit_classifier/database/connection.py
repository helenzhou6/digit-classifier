from psycopg2 import connect
from dotenv import load_dotenv
from os import getenv

load_dotenv()

POSTGRES_PASSWORD = getenv('POSTGRES_PASSWORD')
POSTGRES_USERNAME = getenv('POSTGRES_USERNAME')
DB_HOST = getenv('DB_HOST')
DB_PORT = getenv('DB_PORT')

def connect_to_db():
    conn = connect(
        user=POSTGRES_USERNAME, password=POSTGRES_PASSWORD, host=DB_HOST, port=DB_PORT
    )

    return conn

def close_db_connection(conn):
    conn.commit()
    conn.close()