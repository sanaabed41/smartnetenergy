from fastapi import FastAPI, HTTPException, Security, Request, Depends
from fastapi.security import APIKeyHeader, APIKeyQuery
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import os, logging, time
from collections import defaultdict
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════
# SECURITY CONFIG
# ══════════════════════════════════════
API_KEYS = {
    os.getenv("API_KEY_ADMIN", "sne-admin-key-2024"):    "admin",
    os.getenv("API_KEY_ENGINEER", "sne-eng-key-2024"):   "engineer",
    os.getenv("API_KEY_READONLY", "sne-read-key-2024"):  "readonly",
}

ALLOWED_IPS = os.getenv("ALLOWED_IPS", "").split(",") if os.getenv("ALLOWED_IPS") else []

# Rate limiting: max requests per minute per IP
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
rate_store = defaultdict(list)

# ══════════════════════════════════════
# APP INIT
# ══════════════════════════════════════
app = FastAPI(
    title="SmartNetEnergy API",
    description="5G Energy Consumption Prediction — Secured",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sanaabed41.github.io"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

model = None
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query  = APIKeyQuery(name="api_key",    auto_error=False)

# ══════════════════════════════════════
# SECURITY MIDDLEWARE
# ══════════════════════════════════════
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    client_ip = request.client.host

    # 1. IP Whitelisting (skip for /health and /docs)
    if ALLOWED_IPS and request.url.path not in ["/health", "/docs", "/redoc", "/openapi.json"]:
        if client_ip not in ALLOWED_IPS:
            logger.warning(f"Blocked IP: {client_ip}")
            return JSONResponse(
                status_code=403,
                content={"error": "Access denied", "ip": client_ip}
            )

    # 2. Rate Limiting
    now = datetime.now()
    window_start = now - timedelta(minutes=1)
    rate_store[client_ip] = [t for t in rate_store[client_ip] if t > window_start]

    if len(rate_store[client_ip]) >= RATE_LIMIT:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return JSONResponse(
            status_code=429,
            content={
                "error": "Too many requests",
                "limit": f"{RATE_LIMIT} requests/minute",
                "retry_after": "60 seconds"
            }
        )

    rate_store[client_ip].append(now)

    # 3. Log all requests
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s - {client_ip}")

    response.headers["X-Process-Time"] = str(process_time)
    return response

# ══════════════════════════════════════
# API KEY VALIDATION
# ══════════════════════════════════════
async def verify_api_key(
    key_header: str = Security(api_key_header),
    key_query:  str = Security(api_key_query)
):
    api_key = key_header or key_query
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API Key required. Pass it via X-API-Key header or ?api_key= query param"
        )
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key"
        )
    return {"key": api_key, "role": API_KEYS[api_key]}

# ══════════════════════════════════════
# MODEL LOAD
# ══════════════════════════════════════
@app.on_event("startup")
async def load_model():
    global model
    try:
        import joblib
        model_path = os.getenv("MODEL_PATH", "models/energy_model.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"Model loaded: {type(model).__name__}")
        else:
            logger.warning("Model not found — demo mode")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

# ══════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════
class PredictionRequest(BaseModel):
    traffic_load:  float
    antenna_count: int
    bandwidth:     float
    tx_power:      float
    frequency:     float
    esmode_sum:    int = 0
    hour:          int = 12

    class Config:
        json_schema_extra = {"example": {
            "traffic_load": 1200, "antenna_count": 64,
            "bandwidth": 100, "tx_power": 43,
            "frequency": 3500, "esmode_sum": 2, "hour": 14
        }}

class PredictionResponse(BaseModel):
    energy_prediction: float
    energy_kwh:        float
    unit:              str = "watts"
    status:            str = "success"
    model_type:        str = "demo"
    timestamp:         str = ""

# ══════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════
@app.get("/")
def root():
    return {
        "project": "SmartNetEnergy",
        "version": "3.0.0",
        "security": ["API Key", "Rate Limiting", "IP Filtering", "CORS"],
        "model_loaded": model is not None,
        "docs": "/docs"
    }

@app.get("/health")
def health():
    # No auth required for health check
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else "none",
        "demo_mode": model is None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    auth: dict = Depends(verify_api_key)
):
    logger.info(f"Prediction request by role: {auth['role']}")
    try:
        load = request.traffic_load
        hour_rad  = (request.hour / 24) * 2 * np.pi
        heure_sin = np.sin(hour_rad)
        heure_cos = np.cos(hour_rad)
        est_nuit  = 1 if request.hour in range(0, 6) else 0
        est_pic   = 1 if request.hour in range(9, 12) or request.hour in range(18, 22) else 0
        p_dynamic = request.antenna_count * request.tx_power * load

        features = np.array([[
            load, load**2, load**3,
            request.tx_power, request.frequency,
            request.bandwidth, request.antenna_count,
            1,  # rutype default
            heure_sin, heure_cos, est_nuit, est_pic,
            p_dynamic,
            request.esmode_sum, 0, 0, 0
        ]])

        if model is not None:
            pred = float(np.expm1(model.predict(features)[0]))
            model_type = type(model).__name__
        else:
            pred = (load * 0.15 + request.antenna_count * 2.5 +
                    request.bandwidth * 0.8 + request.tx_power * 1.2)
            model_type = "demo"

        return PredictionResponse(
            energy_prediction=round(pred, 2),
            energy_kwh=round(pred / 1000, 4),
            model_type=model_type,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics(auth: dict = Depends(verify_api_key)):
    return {
        "model_loaded": model is not None,
        "model_type":   type(model).__name__ if model else "none",
        "api_version":  "3.0.0",
        "role":         auth["role"],
        "security":     ["API Key", "Rate Limiting", "IP Filtering"]
    }

@app.get("/admin/keys", tags=["Admin"])
async def list_keys(auth: dict = Depends(verify_api_key)):
    if auth["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return {
        "keys": [{"role": role, "masked": f"{key[:8]}..."}
                 for key, role in API_KEYS.items()]
    }
