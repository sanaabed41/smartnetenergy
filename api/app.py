from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os, logging, joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SmartNetEnergy API",
    description="5G Energy Consumption Prediction — Stacking Model",
    version="2.0.0"
)

model = None
scaler = None
encoders = None

@app.on_event("startup")
async def load_model():
    global model, scaler, encoders
    try:
        model_path = os.getenv("MODEL_PATH", "models/energy_model.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"✅ Model loaded: {type(model).__name__}")
        else:
            logger.warning("⚠️ Model not found — demo mode")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

class PredictionRequest(BaseModel):
    # Features de base
    traffic_load: float          # load
    tx_power: float              # TXpower
    frequency: float             # Frequency
    bandwidth: float             # Bandwidth
    antenna_count: int           # Antennas
    rutype: int = 1              # RUType encodé
    # Features temporelles
    hour: int = 12               # heure (0-23)
    # ESModes (4 et 5 supprimés)
    esmode1: int = 0
    esmode2: int = 0
    esmode3: int = 0
    esmode6: int = 0

    class Config:
        json_schema_extra = {"example": {
            "traffic_load": 0.7, "tx_power": 43,
            "frequency": 3500, "bandwidth": 100,
            "antenna_count": 64, "rutype": 1,
            "hour": 14, "esmode1": 1, "esmode2": 0,
            "esmode3": 0, "esmode6": 0
        }}

class PredictionResponse(BaseModel):
    energy_prediction: float
    energy_kwh: float
    unit: str = "watts"
    status: str = "success"
    model_type: str = "demo"

@app.get("/")
def root():
    return {
        "project": "SmartNetEnergy",
        "version": "2.0.0",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else "demo"
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
def predict(req: PredictionRequest):
    try:
        load = req.traffic_load
        hour_rad = (req.hour / 24) * 2 * np.pi
        heure_sin = np.sin(hour_rad)
        heure_cos = np.cos(hour_rad)
        est_nuit = 1 if req.hour in range(0, 6) else 0
        est_pic  = 1 if req.hour in range(9, 12) or req.hour in range(18, 22) else 0
        p_dynamic = req.antenna_count * req.tx_power * load

        features = np.array([[
            load,
            load**2,
            load**3,
            req.tx_power,
            req.frequency,
            req.bandwidth,
            req.antenna_count,
            req.rutype,
            heure_sin,
            heure_cos,
            est_nuit,
            est_pic,
            p_dynamic,
            req.esmode1,
            req.esmode2,
            req.esmode3,
            req.esmode6,
        ]])

        if model is not None:
            log_pred = model.predict(features)[0]
            prediction = float(np.expm1(log_pred))
            model_type = type(model).__name__
        else:
            prediction = (
                load * 200 +
                req.antenna_count * 2.5 +
                req.bandwidth * 0.8 +
                req.tx_power * 1.2
            )
            model_type = "demo"

        return PredictionResponse(
            energy_prediction=round(prediction, 2),
            energy_kwh=round(prediction/1000, 4),
            model_type=model_type
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    return {
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else "none",
        "api_version": "2.0.0",
        "features": 17,
        "algorithm": "Stacking (LightGBM + CatBoost + RF → Ridge)"
    }
