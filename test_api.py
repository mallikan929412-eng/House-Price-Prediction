"""
Pytest Suite for NammaMane API and ML Pipeline
----------------------------------------------
Tests health check, options, prediction, what-if, compare, deal-check,
locality-stats, model-report, and validation errors (422).
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app, load_artifacts

# Ensure artifacts are loaded in test context
load_artifacts()
client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["is_synthetic_data"] is True
    assert "HistGradientBoostingRegressor" in data["best_model"]

def test_options():
    response = client.get("/options")
    assert response.status_code == 200
    data = response.json()
    assert "localities" in data
    assert len(data["localities"]) == 25
    assert "Indiranagar" in data["localities"]
    assert "ranges" in data
    assert "bhk" in data["ranges"]
    assert "area_sqft" in data["ranges"]

def test_predict_valid():
    payload = {
        "locality": "Indiranagar",
        "property_type": "Apartment",
        "bhk": 3,
        "area_sqft": 1500,
        "bathrooms": 3,
        "balconies": 2.0,
        "floor": 4,
        "total_floors": 12,
        "age_years": 3,
        "furnishing": "Semi-Furnished",
        "parking_slots": 1,
        "lift_available": "Yes",
        "gated_community": "Yes",
        "amenities_score": 8.5,
        "metro_distance_km": 1.2,
        "school_distance_km": 0.8,
        "tech_park_distance_km": 4.5,
        "facing": "East",
        "rera_approved": "Yes"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_price_lakhs"] > 50.0
    assert data["confidence_range_lakhs"] > 0
    assert data["price_range_low_lakhs"] < data["price_range_high_lakhs"]
    assert data["price_per_sqft"] > 3000
    assert len(data["why_this_price"]) > 0
    assert data["is_synthetic_data"] is True

def test_predict_validation_error_invalid_locality():
    payload = {
        "locality": "FakeLocalityNotInBengaluru",
        "property_type": "Apartment",
        "bhk": 2,
        "area_sqft": 1100
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "Invalid locality" in response.text

def test_predict_validation_error_negative_area():
    payload = {
        "locality": "Whitefield",
        "property_type": "Apartment",
        "bhk": 2,
        "area_sqft": -500
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_whatif_analysis():
    payload = {
        "base_house": {
            "locality": "Whitefield",
            "property_type": "Apartment",
            "bhk": 2,
            "area_sqft": 1100,
            "age_years": 8,
            "metro_distance_km": 6.0
        },
        "scenarios": [
            {
                "label": "Closer to Namma Metro (0.8 km)",
                "changes": {"metro_distance_km": 0.8}
            },
            {
                "label": "Brand New Construction (1 yr)",
                "changes": {"age_years": 1}
            }
        ]
    }
    response = client.post("/whatif", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["base_price_lakhs"] > 0
    assert len(data["results"]) == 2
    for r in data["results"]:
        assert "predicted_price_lakhs" in r
        assert "difference_lakhs" in r
        assert "percentage_change" in r

def test_compare_houses_duel():
    payload = {
        "house_a": {
            "locality": "Indiranagar",
            "property_type": "Apartment",
            "bhk": 2,
            "area_sqft": 1000,
            "age_years": 4,
            "metro_distance_km": 1.0,
            "amenities_score": 8.0
        },
        "house_b": {
            "locality": "Electronic City",
            "property_type": "Apartment",
            "bhk": 3,
            "area_sqft": 1500,
            "age_years": 2,
            "metro_distance_km": 4.0,
            "amenities_score": 8.5
        },
        "label_a": "Indiranagar 2BHK",
        "label_b": "Electronic City 3BHK"
    }
    response = client.post("/compare", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "price_difference_lakhs" in data
    assert "winner_banner" in data
    assert "category_winners" in data
    assert "price" in data["category_winners"]
    assert "spaciousness" in data["category_winners"]

def test_deal_check_bargain_and_overpriced():
    house = {
        "locality": "Koramangala",
        "property_type": "Apartment",
        "bhk": 3,
        "area_sqft": 1400,
        "age_years": 3,
        "metro_distance_km": 2.0
    }
    # Predict to get baseline
    pred_res = client.post("/predict", json=house).json()
    fair_price = pred_res["predicted_price_lakhs"]

    # 1. Test Bargain (asking 25% lower)
    bargain_payload = {
        "house": house,
        "asking_price_lakhs": round(fair_price * 0.75, 2)
    }
    res_bargain = client.post("/deal-check", json=bargain_payload)
    assert res_bargain.status_code == 200
    data_b = res_bargain.json()
    assert data_b["status"] == "Bargain"
    assert data_b["gauge_percentage"] <= 40.0

    # 2. Test Overpriced (asking 30% higher)
    overpriced_payload = {
        "house": house,
        "asking_price_lakhs": round(fair_price * 1.30, 2)
    }
    res_over = client.post("/deal-check", json=overpriced_payload)
    assert res_over.status_code == 200
    data_o = res_over.json()
    assert data_o["status"] == "Overpriced"
    assert data_o["gauge_percentage"] >= 60.0

def test_locality_stats():
    response = client.get("/locality-stats")
    assert response.status_code == 200
    data = response.json()
    assert "localities" in data
    assert len(data["localities"]) == 25
    first = data["localities"][0]
    assert "locality" in first
    assert "avg_price_per_sqft" in first
    assert "lat" in first
    assert "lng" in first

def test_model_report():
    response = client.get("/model-report")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "models" in data["metrics"]
    models = data["metrics"]["models"]
    assert "LinearRegression" in models
    assert "RandomForestRegressor" in models
    assert "HistGradientBoostingRegressor" in models
    assert len(data["feature_importance"]) > 0
