from fastapi import FastAPI, File, UploadFile
from numpy import frombuffer, uint8

from digit_classifier.model.run_model import predict_digit
from digit_classifier.model.init_model import init_model

app = FastAPI()
model = init_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
      content = await file.read()
      data = frombuffer(content, dtype=uint8).reshape((280, 280, 4))
      predicted_digit, conf_percent = predict_digit(data, model)
      return {
            "predicted_digit": predicted_digit,
            "conf_percent": conf_percent
      }

@app.get("/healthcheck")
async def healthcheck():
      return "The model API is up and running"