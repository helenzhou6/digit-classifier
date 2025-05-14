import requests
from dotenv import load_dotenv
import os

load_dotenv()
PREDICT_URL = os.getenv('PREDICT_URL')
DATABASE_URL = os.getenv('DATABASE_URL')

def predict_digit_api(canvas_img_data):
    data_bytes = canvas_img_data.tobytes()
    files = {'file': ('data.npy', data_bytes, 'application/octet-stream')}
    response = requests.post(PREDICT_URL, files=files)
    return response.json()

def get_feedback_records():
    response = requests.get(DATABASE_URL)
    return response.json()