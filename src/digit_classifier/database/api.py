from fastapi import FastAPI
from digit_classifier.database.feedback_cmd import get_feedback_records
from digit_classifier.database.create_table import create_table_with_dummy_data

appdb = FastAPI()

create_table_with_dummy_data()

@appdb.get("/getrecords")
async def getrecords():
      return get_feedback_records()

@appdb.get("/healthcheck")
async def healthcheck():
      return "The database API is up and running"