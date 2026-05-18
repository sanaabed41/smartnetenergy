from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from app import app

client = TestClient(app)

VALID_KEY   = "sne-admin-key-2024"
INVALID_KEY = "wrong-key-123"

def test_health_no_auth():
    """Health check should work without auth"""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_predict_no_key():
    """Predict without API key should return 401"""
    r = client.post("/predict", json={
        "traffic_load": 1200, "antenna_count": 64,
        "bandwidth": 100, "tx_power": 43,
        "frequency": 3500, "esmode_sum": 2
    })
    assert r.status_code == 401

def test_predict_invalid_key():
    """Predict with wrong key should return 403"""
    r = client.post("/predict",
        headers={"X-API-Key": INVALID_KEY},
        json={"traffic_load": 1200, "antenna_count": 64,
              "bandwidth": 100, "tx_power": 43,
              "frequency": 3500, "esmode_sum": 2}
    )
    assert r.status_code == 403

def test_predict_valid_key():
    """Predict with valid key should return 200"""
    r = client.post("/predict",
        headers={"X-API-Key": VALID_KEY},
        json={"traffic_load": 1200, "antenna_count": 64,
              "bandwidth": 100, "tx_power": 43,
              "frequency": 3500, "esmode_sum": 2}
    )
    assert r.status_code == 200
    assert "energy_prediction" in r.json()

def test_predict_key_query_param():
    """Predict with key as query param should work"""
    r = client.post(f"/predict?api_key={VALID_KEY}",
        json={"traffic_load": 1200, "antenna_count": 64,
              "bandwidth": 100, "tx_power": 43,
              "frequency": 3500, "esmode_sum": 2}
    )
    assert r.status_code == 200

def test_admin_endpoint_non_admin():
    """Admin endpoint with engineer key should return 403"""
    r = client.get("/admin/keys",
        headers={"X-API-Key": "sne-eng-key-2024"}
    )
    assert r.status_code == 403

def test_admin_endpoint_admin():
    """Admin endpoint with admin key should return 200"""
    r = client.get("/admin/keys",
        headers={"X-API-Key": VALID_KEY}
    )
    assert r.status_code == 200
