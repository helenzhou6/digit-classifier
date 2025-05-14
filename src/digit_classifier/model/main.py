from fastapi import FastAPI
from digit_classifier.model.run_model import predict_digit

app = FastAPI()
@app.post("/predict")
async def predict(uint8_img):
      return predict_digit(uint8_img)

@app.get("/healthcheck")
async def hello():
      return "The model API is up and running"