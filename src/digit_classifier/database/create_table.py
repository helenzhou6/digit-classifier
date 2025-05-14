from digit_classifier.database.connection import connect_to_db, close_db_connection

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