from psycopg2 import sql

from digit_classifier.database.connection import connect_to_db, close_db_connection

def add_feedback_record(predicted_digit: int, true_digit: int, conf_percent: float):
    conn = connect_to_db()
    cur = conn.cursor()
    cur.execute(
        sql.SQL("INSERT INTO {} (timestamp, predicted_digit, true_digit, conf_percent) VALUES (NOW(), %s, %s, %s)")
            .format(sql.Identifier("feedback")),
            [predicted_digit, true_digit, conf_percent])
    close_db_connection(conn)

def get_feedback_records():
    conn =  connect_to_db()
    cur = conn.cursor()
    cur.execute("SELECT TO_CHAR(timestamp, 'YYYY-MM-DD HH24:MI:SS') AS formatted_datetime, predicted_digit, true_digit, conf_percent FROM feedback ORDER BY formatted_datetime DESC")
    records = cur.fetchall()
    close_db_connection(conn)
    return records