import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_USERNAME = os.getenv('POSTGRES_USERNAME')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
conn = psycopg2.connect(f"user={POSTGRES_USERNAME} password={POSTGRES_PASSWORD} port={DB_PORT} host={DB_HOST}")

cur = conn.cursor()
cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                predicted_digit BIGINT,
                true_digit BIGINT,
                conf_percent FLOAT,
                timestamp TIMESTAMP
            )
            """)
cur.execute("INSERT INTO feedback (predicted_digit, true_digit, conf_percent, timestamp) VALUES ('1', '4', '90.0', NOW());")
cur.execute("SELECT * FROM feedback")
records = cur.fetchall()
for record in records:
    print(record)

cur.execute("DROP TABLE feedback")

conn.commit()
conn.close()