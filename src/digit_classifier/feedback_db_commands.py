from digit_classifier.database import connect_to_db, close_db_connection

def add_feedback_record():
    conn =  connect_to_db()
    cur = conn.cursor()
    cur.execute(f"INSERT INTO feedback (timestamp, predicted_digit, true_digit, conf_percent) VALUES (NOW(), '3', '4', '80.0');")
    close_db_connection(conn)

def get_feedback_records():
    conn =  connect_to_db()
    cur = conn.cursor()
    cur.execute("SELECT timestamp, predicted_digit, true_digit, conf_percent FROM feedback ORDER BY timestamp DESC")
    records = cur.fetchall()
    close_db_connection(conn)
    return records