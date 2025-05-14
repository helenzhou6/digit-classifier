import requests
from dotenv import load_dotenv
import os

load_dotenv()
MODEL_API_URL = os.getenv('MODEL_API_URL')
DATABASE_API_URL = os.getenv('DATABASE_API_URL')

def predict_digit_api(canvas_img_data):
    data_bytes = canvas_img_data.tobytes()
    files = {'file': ('data.npy', data_bytes, 'application/octet-stream')}
    response = requests.post(f"{MODEL_API_URL}/predict", files=files)
    return response.json()

def get_feedback_records():
    response = requests.get(f"{DATABASE_API_URL}/getfeedbackrecords")
    return response.json()

def add_feedback_record(predicted_digit: int, true_digit: int, conf_level: float):
    data = {
        "predicted_digit": predicted_digit,
        "true_digit": true_digit,
        "conf_level": conf_level
    }
    response = requests.post(f"{DATABASE_API_URL}/postfeedbackrecord", json=data)
    return response.json()