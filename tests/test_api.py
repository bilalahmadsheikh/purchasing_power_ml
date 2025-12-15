"""
API Tests for PPP-Q Classifier
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_root():
    """Test health check"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_bitcoin():
    """Test Bitcoin prediction"""
    response = client.post(
        "/predict",
        json={"asset": "Bitcoin", "horizon_years": 5}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["asset"] == "Bitcoin"
    assert "predicted_class" in data
    assert "confidence" in data
    assert "strengths" in data

def test_list_assets():
    """Test asset listing"""
    response = client.get("/assets")
    assert response.status_code == 200
    assert len(response.json()["assets"]) == 15

def test_model_info():
    """Test model info"""
    response = client.get("/model/info")
    assert response.status_code == 200
    assert response.json()["model_type"] == "LightGBM Gradient Boosting"