"""
NammaMane FastAPI Application
-----------------------------
Production-ready backend for Bengaluru house price predictions,
what-if explorations, property duels, and deal fairness checks.
"""

import os
import json
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

from backend.app.schemas import (
    HouseInput,
    PredictionResponse,
    WhyThisPriceItem,
    WhatIfRequest,
    WhatIfResponse,
    WhatIfResult,
    CompareRequest,
    CompareResponse,
    DealCheckRequest,
    DealCheckResponse
)

# Base Paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(APP_DIR)
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
DATA_DIR = os.path.join(PROJECT_DIR, "data")

MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
OPTIONS_PATH = os.path.join(MODELS_DIR, "options.json")
LOCALITY_STATS_PATH = os.path.join(MODELS_DIR, "locality_stats.json")
COORDINATES_PATH = os.path.join(DATA_DIR, "bengaluru_coordinates.json")

# Global in-memory cache for artifacts
artifacts: Dict[str, Any] = {}

def load_artifacts():
    """Loads trained pipeline model and metadata into memory."""
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Trained model not found at {MODEL_PATH}. Run backend/ml/train.py first.")
    
    artifacts["model"] = joblib.load(MODEL_PATH)

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        artifacts["metrics"] = json.load(f)

    with open(OPTIONS_PATH, "r", encoding="utf-8") as f:
        artifacts["options"] = json.load(f)

    with open(LOCALITY_STATS_PATH, "r", encoding="utf-8") as f:
        artifacts["locality_stats"] = json.load(f)

    with open(os.path.join(MODELS_DIR, "feature_importance.json"), "r", encoding="utf-8") as f:
        artifacts["feature_importance"] = json.load(f)

    if os.path.exists(COORDINATES_PATH):
        with open(COORDINATES_PATH, "r", encoding="utf-8") as f:
            artifacts["coordinates"] = json.load(f)
    else:
        artifacts["coordinates"] = {}

    print("NammaMane artifacts successfully loaded into memory!")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield

# Initialize FastAPI App
app = FastAPI(
    title="NammaMane API",
    description="Bengaluru House Price Prediction & Real-Estate Analytics API (Synthetic Dataset of 15,000 listings)",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def run_single_prediction(house: HouseInput) -> float:
    """Invokes pipeline on a single HouseInput and returns predicted price in Lakhs."""
    model = artifacts.get("model")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently loading or unavailable."
        )
    df_input = pd.DataFrame([house.to_df_dict()])
    pred = float(model.predict(df_input)[0])
    return max(5.0, round(pred, 2))

def compute_why_this_price(house: HouseInput, predicted_price: float) -> list:
    """Generates beginner-friendly factor impact breakdown."""
    items = []
    stats = artifacts.get("locality_stats", {})
    loc_stat = stats.get(house.locality, {})
    avg_pps = loc_stat.get("avg_price_per_sqft", 8500)
    
    # 1. Locality factor
    city_avg_pps = 9200
    loc_diff_pct = ((avg_pps - city_avg_pps) / city_avg_pps) * 100
    if loc_diff_pct > 15:
        items.append(WhyThisPriceItem(
            factor=f"Prime Locality ({house.locality})",
            impact=f"+{round(loc_diff_pct * 0.4, 1)}%",
            direction="positive",
            description=f"{house.locality} commands higher land value averaging ₹{avg_pps:,.0f}/sqft."
        ))
    elif loc_diff_pct < -15:
        items.append(WhyThisPriceItem(
            factor=f"Affordable Locality ({house.locality})",
            impact=f"-{round(abs(loc_diff_pct) * 0.4, 1)}%",
            direction="negative",
            description=f"{house.locality} offers attractive value averaging ₹{avg_pps:,.0f}/sqft."
        ))

    # 2. Area factor
    if house.area_sqft > 1600:
        items.append(WhyThisPriceItem(
            factor="Spacious Area",
            impact=f"+₹{round((house.area_sqft - 1200) * (avg_pps / 100000) * 0.8, 1)}L",
            direction="positive",
            description=f"Expansive {int(house.area_sqft)} sq.ft layout increases total valuation."
        ))
    elif house.area_sqft < 900:
        items.append(WhyThisPriceItem(
            factor="Compact Footprint",
            impact=f"-₹{round((1200 - house.area_sqft) * (avg_pps / 100000) * 0.7, 1)}L",
            direction="negative",
            description=f"Compact {int(house.area_sqft)} sq.ft size keeps total price accessible."
        ))

    # 3. Metro Proximity
    if house.metro_distance_km <= 1.5:
        items.append(WhyThisPriceItem(
            factor="Close to Namma Metro",
            impact="+₹5.5L - ₹8.0L",
            direction="positive",
            description=f"Under {house.metro_distance_km:.1f} km from metro station adds substantial transit convenience."
        ))
    elif house.metro_distance_km >= 8.0:
        items.append(WhyThisPriceItem(
            factor="Distance to Metro",
            impact="-₹4.0L - ₹6.5L",
            direction="negative",
            description=f"{house.metro_distance_km:.1f} km from nearest metro station slightly discounts appreciation."
        ))

    # 4. Property Age
    if house.age_years <= 2:
        items.append(WhyThisPriceItem(
            factor="New Construction",
            impact="+₹4.5L",
            direction="positive",
            description=f"{house.age_years} year old build offers modern architectural standards & lower maintenance."
        ))
    elif house.age_years >= 12:
        items.append(WhyThisPriceItem(
            factor="Property Age",
            impact=f"-₹{round(house.age_years * 0.5, 1)}L",
            direction="negative",
            description=f"{house.age_years} years of age incurs standard structural depreciation."
        ))

    # 5. Gated Community & Amenities
    if house.gated_community == "Yes" and (house.amenities_score or 0) >= 7.5:
        items.append(WhyThisPriceItem(
            factor="Premium Gated Community",
            impact="+₹8.0L - ₹12.0L",
            direction="positive",
            description=f"High amenities rating of {house.amenities_score}/10 with round-the-clock security and club features."
        ))

    return items[:4]

# --- Endpoints ---

@app.get("/health", tags=["System"])
def health_check():
    """Health status and model readiness check."""
    return {
        "status": "ok",
        "app_name": "NammaMane",
        "model_loaded": "model" in artifacts,
        "best_model": artifacts.get("metrics", {}).get("best_model", "HistGradientBoostingRegressor"),
        "dataset": "15,000 synthetic Bengaluru listings",
        "is_synthetic_data": True
    }

@app.get("/options", tags=["Metadata"])
def get_options():
    """Returns dropdown choices and slider min/max ranges for the frontend forms."""
    if "options" not in artifacts:
        raise HTTPException(status_code=500, detail="Options metadata not loaded.")
    return artifacts["options"]

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_price(house: HouseInput):
    """
    Predicts Bengaluru house price in lakhs with +/- range based on cross-validated MAE.
    Also returns price per sqft and 'Why this price?' explanation breakdown.
    """
    predicted_price = run_single_prediction(house)
    
    # Retrieve cross-validated MAE or test MAE for confidence range
    metrics = artifacts.get("metrics", {})
    confidence_margin = metrics.get("cross_val_mae", 8.0)
    
    price_low = max(5.0, round(predicted_price - confidence_margin, 2))
    price_high = round(predicted_price + confidence_margin, 2)
    price_per_sqft = round((predicted_price * 100000.0) / house.area_sqft, 1)

    why_breakdown = compute_why_this_price(house, predicted_price)

    return PredictionResponse(
        predicted_price_lakhs=predicted_price,
        confidence_range_lakhs=round(confidence_margin, 2),
        price_range_low_lakhs=price_low,
        price_range_high_lakhs=price_high,
        price_per_sqft=price_per_sqft,
        model_used=metrics.get("best_model", "HistGradientBoostingRegressor"),
        why_this_price=why_breakdown,
        is_synthetic_data=True
    )

@app.post("/whatif", response_model=WhatIfResponse, tags=["Analytics"])
def what_if_analysis(request: WhatIfRequest):
    """
    Simulates variations against a base house, returning price differences and percentage impacts.
    """
    base_price = run_single_prediction(request.base_house)
    results = []

    for scenario in request.scenarios:
        # Clone base house attributes and update with scenario changes
        house_dict = request.base_house.model_dump()
        house_dict.update(scenario.changes)
        
        try:
            modified_house = HouseInput(**house_dict)
            scenario_price = run_single_prediction(modified_house)
            diff = round(scenario_price - base_price, 2)
            pct = round((diff / base_price) * 100.0, 2) if base_price > 0 else 0.0

            results.append(WhatIfResult(
                label=scenario.label,
                predicted_price_lakhs=scenario_price,
                difference_lakhs=diff,
                percentage_change=pct,
                changes_applied=scenario.changes
            ))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Error applying scenario '{scenario.label}': {str(e)}"
            )

    return WhatIfResponse(
        base_price_lakhs=base_price,
        results=results,
        is_synthetic_data=True
    )

@app.post("/compare", response_model=CompareResponse, tags=["Comparison"])
def compare_houses(request: CompareRequest):
    """
    House Duel: compares two properties side-by-side with winning category badges.
    """
    price_a = run_single_prediction(request.house_a)
    price_b = run_single_prediction(request.house_b)
    
    pps_a = round((price_a * 100000.0) / request.house_a.area_sqft, 1)
    pps_b = round((price_b * 100000.0) / request.house_b.area_sqft, 1)

    diff = round(abs(price_a - price_b), 2)
    cheaper = request.label_a if price_a < price_b else request.label_b

    # Category comparisons
    winners = {}
    winners["price"] = request.label_a if price_a < price_b else request.label_b
    winners["price_per_sqft"] = request.label_a if pps_a < pps_b else request.label_b
    winners["spaciousness"] = request.label_a if request.house_a.area_sqft > request.house_b.area_sqft else request.label_b
    winners["metro_proximity"] = request.label_a if (request.house_a.metro_distance_km or 99) < (request.house_b.metro_distance_km or 99) else request.label_b
    winners["newness"] = request.label_a if (request.house_a.age_years or 99) < (request.house_b.age_years or 99) else request.label_b
    winners["amenities"] = request.label_a if (request.house_a.amenities_score or 0) > (request.house_b.amenities_score or 0) else request.label_b

    a_wins = sum(1 for v in winners.values() if v == request.label_a)
    b_wins = len(winners) - a_wins

    if a_wins > b_wins:
        banner = f"🏆 {request.label_a} leads with {a_wins} category advantages!"
    elif b_wins > a_wins:
        banner = f"🏆 {request.label_b} leads with {b_wins} category advantages!"
    else:
        banner = "⚖️ Both properties offer competitive, well-matched tradeoffs!"

    prop_a_data = request.house_a.model_dump()
    prop_a_data["predicted_price_lakhs"] = price_a
    prop_a_data["price_per_sqft"] = pps_a

    prop_b_data = request.house_b.model_dump()
    prop_b_data["predicted_price_lakhs"] = price_b
    prop_b_data["price_per_sqft"] = pps_b

    return CompareResponse(
        property_a=prop_a_data,
        property_b=prop_b_data,
        price_difference_lakhs=diff,
        cheaper_option=cheaper,
        winner_banner=banner,
        category_winners=winners,
        is_synthetic_data=True
    )

@app.post("/deal-check", response_model=DealCheckResponse, tags=["Valuation"])
def deal_checker(request: DealCheckRequest):
    """
    Evaluates whether an asking price represents a 'Bargain', 'Fair', or 'Overpriced' deal.
    """
    predicted = run_single_prediction(request.house)
    asking = request.asking_price_lakhs
    
    diff_lakhs = round(asking - predicted, 2)
    pct_diff = round(((asking - predicted) / predicted) * 100.0, 1)

    # Thresholds: <= -7% Bargain, >= +7% Overpriced, else Fair
    if pct_diff <= -7.0:
        deal_status = "Bargain"
        badge_color = "#10B981"  # Emerald green
        explanation = f"Asking price is {abs(pct_diff):.1f}% below predicted valuation of ₹{predicted:.2f}L. Excellent buyer opportunity!"
        # Map pct_diff (-30% to -7%) to gauge (10 to 35)
        gauge_val = max(10.0, 35.0 - (abs(pct_diff) - 7.0) * 1.5)
    elif pct_diff >= 7.0:
        deal_status = "Overpriced"
        badge_color = "#EF4444"  # Rose red
        explanation = f"Asking price is {pct_diff:.1f}% higher than predicted valuation of ₹{predicted:.2f}L. Room for negotiation recommended."
        # Map pct_diff (7% to 40%) to gauge (65 to 95)
        gauge_val = min(95.0, 65.0 + (pct_diff - 7.0) * 1.5)
    else:
        deal_status = "Fair"
        badge_color = "#F5A524"  # Marigold golden
        explanation = f"Asking price closely aligns within {abs(pct_diff):.1f}% of predicted valuation (₹{predicted:.2f}L)."
        gauge_val = 50.0 + (pct_diff * 1.5)

    return DealCheckResponse(
        asking_price_lakhs=asking,
        predicted_price_lakhs=predicted,
        difference_lakhs=diff_lakhs,
        difference_percentage=pct_diff,
        status=deal_status,
        badge_color=badge_color,
        gauge_percentage=round(gauge_val, 1),
        explanation=explanation,
        is_synthetic_data=True
    )

@app.get("/locality-stats", tags=["Market Data"])
def get_locality_stats():
    """
    Returns average price per sqft, listing count, and approximate coordinates for all 25 localities.
    """
    stats = artifacts.get("locality_stats", {})
    coords = artifacts.get("coordinates", {}).get("localities", {})

    combined = []
    for loc, data in stats.items():
        loc_coords = coords.get(loc, {"lat": 12.9716, "lng": 77.5946, "zone": data.get("zone", "South")})
        combined.append({
            "locality": loc,
            "zone": data.get("zone", "South"),
            "listing_count": data.get("listing_count", 0),
            "avg_price_per_sqft": data.get("avg_price_per_sqft", 0),
            "avg_price_lakhs": data.get("avg_price_lakhs", 0),
            "median_price_lakhs": data.get("median_price_lakhs", 0),
            "lat": loc_coords.get("lat"),
            "lng": loc_coords.get("lng"),
            "is_approximate_coords": True
        })

    # Sort from highest to lowest price per sqft
    combined.sort(key=lambda x: x["avg_price_per_sqft"], reverse=True)
    return {
        "localities": combined,
        "is_synthetic_data": True,
        "notice": "Trained on synthetic listings. Coordinates are approximate centers for map visualization."
    }

@app.get("/model-report", tags=["ML Metrics"])
def get_model_report():
    """
    Returns metrics comparison between 3 models, 5-fold CV statistics,
    and permutation feature importances.
    """
    return {
        "metrics": artifacts.get("metrics", {}),
        "feature_importance": artifacts.get("feature_importance", []),
        "is_synthetic_data": True
    }
