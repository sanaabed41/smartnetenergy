import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)
VALID_KEY = os.getenv("API_KEY_ADMIN", "sne-admin-key-2024")

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["project"] == "SmartNetEnergy"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    response = client.post("/predict",
        headers={"X-API-Key": VALID_KEY},
        json={
            "traffic_load": 1200,
            "antenna_count": 64,
            "bandwidth": 100,
            "tx_power": 43,
            "frequency": 3500,
            "esmode_sum": 2
        }
    )
    assert response.status_code == 200
    assert "energy_prediction" in response.json()
    assert response.json()["status"] == "success"

def test_predict_zero_load():
    response = client.post("/predict",
        headers={"X-API-Key": VALID_KEY},
        json={
            "traffic_load": 0,
            "antenna_count": 1,
            "bandwidth": 10,
            "tx_power": 10,
            "frequency": 1800,
            "esmode_sum": 0
        }
    )
    assert response.status_code == 200
    assert response.json()["energy_prediction"] >= 0
