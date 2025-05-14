import requests
from dotenv import load_dotenv
import os

load_dotenv()
PREDICT_URL = os.getenv('PREDICT_URL')

def call_model_api(canvas_img_data):
    data_bytes = canvas_img_data.tobytes()
    files = {'file': ('data.npy', data_bytes, 'application/octet-stream')}
    response = requests.post(PREDICT_URL, files=files)
    return response.json()