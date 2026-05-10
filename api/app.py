from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SmartNetEnergy API",
    description="5G Energy Consumption Prediction API",
    version="1.0.0"
)

model = None
scaler = None

@app.on_event("startup")
async def load_model():
    global model, scaler
    model_path = os.getenv("MODEL_PATH", "models/energy_model.pkl")
    scaler_path = os.getenv("SCALER_PATH", "models/scaler.pkl")
    try:
        import joblib
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path} — demo mode")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

class PredictionRequest(BaseModel):
    traffic_load: float
    antenna_count: int
    bandwidth: float
    tx_power: float
    frequency: float
    esmode_sum: int = 0

    class Config:
        json_schema_extra = {"example": {
            "traffic_load": 1200, "antenna_count": 64,
            "bandwidth": 100, "tx_power": 43,
            "frequency": 3500, "esmode_sum": 2
        }}

class PredictionResponse(BaseModel):
    energy_prediction: float
    unit: str = "watts"
    status: str = "success"
    model_type: str = "demo"
    confidence: float = 0.0

@app.get("/")
def root():
    return {
        "project": "SmartNetEnergy",
        "description": "5G Energy Consumption Prediction API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else "none",
        "demo_mode": model is None
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

        if scaler is not None:
            features = scaler.transform(features)

        if model is not None:
            prediction = float(model.predict(features)[0])
            model_type = type(model).__name__
            confidence = 0.95
        else:
            prediction = (
                request.traffic_load * 0.15 +
                request.antenna_count * 2.5 +
                request.bandwidth * 0.8 +
                request.tx_power * 1.2
            )
            model_type = "demo"
            confidence = 0.0

        return PredictionResponse(
            energy_prediction=round(prediction, 2),
            model_type=model_type,
            confidence=confidence
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    return {
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else "none",
        "api_version": "1.0.0",
        "demo_mode": model is None
    }
