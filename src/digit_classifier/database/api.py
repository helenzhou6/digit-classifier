from fastapi import FastAPI
from pydantic import BaseModel

from digit_classifier.database.feedback_cmd import get_feedback_records, add_feedback_record
from digit_classifier.database.create_table import create_table_with_dummy_data

appdb = FastAPI()

create_table_with_dummy_data()

@appdb.get("/getfeedbackrecords")
async def getfeedbackrecords():
      return get_feedback_records()

class FeedbackRecord(BaseModel):
    predicted_digit: int
    true_digit: int
    conf_level: float

@appdb.post("/postfeedbackrecord")
async def postfeedbackrecord(feedback: FeedbackRecord):
      add_feedback_record(feedback.predicted_digit, feedback.true_digit, feedback.conf_level)
      return "posted feedback"

@appdb.get("/healthcheck")
async def healthcheck():
      return "The database API is up and running"