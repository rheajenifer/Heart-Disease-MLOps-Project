from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="Heart Disease Prediction API"
)

# Load trained model
model = joblib.load("models/model.pkl")

# Input schema
class HeartData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}

@app.post("/predict")
def predict(data: HeartData):

    df = pd.DataFrame([data.dict()])

    prediction = model.predict(df)[0]

    return {
        "prediction": int(prediction)
    }