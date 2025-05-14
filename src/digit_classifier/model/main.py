from fastapi import FastAPI, File, UploadFile
import numpy

from digit_classifier.model.run_model import predict_digit

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
      content = await file.read()
      data = numpy.frombuffer(content, dtype=numpy.uint8)
      predicted_digit, conf_percent = predict_digit(data)
      return {
            "predicted_digit": predicted_digit,
            "conf_percent": conf_percent
      }

@app.get("/healthcheck")
async def hello():
      return "The model API is up and running"