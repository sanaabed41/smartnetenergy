from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(
    title="SmartNetEnergy API",
    description="API de prédiction de consommation énergétique des tours 5G",
    version="1.0.0"
)

# Charger le modèle
model = None

@app.on_event("startup")
async def load_model():
    global model
    model_path = os.getenv("MODEL_PATH", "energy_model.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Modèle chargé avec succès !")
    else:
        print("Modèle non trouvé, mode demo activé")

# Schema de la requête
class PredictionRequest(BaseModel):
    traffic_load: float
    antenna_count: int
    bandwidth: float
    tx_power: float
    frequency: float
    esmode_sum: int = 0

    class Config:
        json_schema_extra = {
            "example": {
                "traffic_load": 1200,
                "antenna_count": 64,
                "bandwidth": 100,
                "tx_power": 43,
                "frequency": 3500,
                "esmode_sum": 2
            }
        }

# Schema de la réponse
class PredictionResponse(BaseModel):
    energy_prediction: float
    unit: str = "watts"
    status: str = "success"

@app.get("/")
def root():
    return {
        "project": "SmartNetEnergy",
        "description": "5G Energy Consumption Prediction API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        features = np.array([[
            request.traffic_load,
            request.tx_power,
            request.frequency,
            request.bandwidth,
            request.antenna_count,
            request.esmode_sum,
            request.tx_power * request.traffic_load
        ]])

        if model is not None:
            prediction = model.predict(features)[0]
        else:
            # Mode demo si pas de modèle
            prediction = (
                request.traffic_load * 0.15 +
                request.antenna_count * 2.5 +
                request.bandwidth * 0.8 +
                request.tx_power * 1.2
            )

        return PredictionResponse(
            energy_prediction=round(float(prediction), 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    return {
        "model_loaded": model is not None,
        "api_version": "1.0.0",
        "endpoint": "/predict"
    }
