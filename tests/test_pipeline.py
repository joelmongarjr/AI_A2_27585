from fastapi.testclient import TestClient
from app import app
import os

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["verification_code"] == "AI-A2-7585"

def test_valid_predict():
    payload = {
        "rainfall_mm": 950.5, "avg_temp_c": 20.2, "soil_moisture_pct": 55.0,
        "fertilizer_kg": 120.0, "field_size_ha": 1.5, "season": "A", "cooperative_support": 1
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    assert "yield_estimate_tons" in res.json()
