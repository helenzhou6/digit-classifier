import psycopg2
import os
from dotenv import load_dotenv

# Connect to your postgres DB
load_dotenv()
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_USERNAME = os.getenv('POSTGRES_USERNAME')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
conn = psycopg2.connect(f"user={POSTGRES_USERNAME} password={POSTGRES_PASSWORD} port={DB_PORT} host={DB_HOST}")

# Open a cursor to perform database operations
cur = conn.cursor()

# Execute a query
cur.execute("CREATE TABLE IF NOT EXISTS test_table (id SERIAL PRIMARY KEY, name VARCHAR(100))")
cur.execute("INSERT INTO test_table (name) VALUES ('John Doe');")

# Retrieve query results
cur.execute("SELECT * FROM test_table")

records = cur.fetchall()
for record in records:
    print(record)

cur.execute("DROP TABLE test_table")

conn.commit()
conn.close()