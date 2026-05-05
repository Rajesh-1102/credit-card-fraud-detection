from fastapi import FastAPI
import joblib
import pandas as pd

from src.config import MODEL_PATH, SCALER_PATH

app = FastAPI(title="Fraud Detection API")

# Load model + scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    # scale
    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }