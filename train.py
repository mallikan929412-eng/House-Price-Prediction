"""
NammaMane Machine Learning Training Pipeline
-------------------------------------------
This module trains and benchmarks three regression models for predicting
Bengaluru house prices (in Lakhs of INR):
1. LinearRegression (Baseline)
2. RandomForestRegressor
3. HistGradientBoostingRegressor (Gradient Boosting)

Trained on a synthetic dataset of 15,000 listings across 25 Bengaluru localities.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "bengaluru_house_prices_15000.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def load_and_clean_data(csv_path: str):
    """
    Loads dataset, drops property_id, handles missing values & removes price-per-sqft outliers.
    """
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    initial_rows = len(df)
    print(f"Initial listings count: {initial_rows}")

    # Drop non-predictive property_id
    if "property_id" in df.columns:
        df = df.drop(columns=["property_id"])

    # Outlier detection on Price per SqFt using IQR per locality
    pps = (df["price_lakhs"] * 100000.0) / df["area_sqft"]
    df["_pps"] = pps

    outlier_mask = pd.Series(False, index=df.index)
    for loc, group in df.groupby("locality"):
        q1 = group["_pps"].quantile(0.25)
        q3 = group["_pps"].quantile(0.75)
        iqr = q3 - q1
        # Multiplier of 2.5 identifies clearly broken, artificially injected price anomalies
        lower_bound = max(0, q1 - 2.5 * iqr)
        upper_bound = q3 + 2.5 * iqr
        broken_rows = group[(group["_pps"] < lower_bound) | (group["_pps"] > upper_bound)].index
        outlier_mask.loc[broken_rows] = True

    removed_count = int(outlier_mask.sum())
    print(f"Clearly broken outlier rows removed via IQR rule: {removed_count}")
    df_cleaned = df[~outlier_mask].drop(columns=["_pps"]).reset_index(drop=True)
    print(f"Final clean listings count: {len(df_cleaned)}")

    return df_cleaned, removed_count, initial_rows

def build_preprocessor(cat_cols, num_cols):
    """
    Creates ColumnTransformer:
    - Median imputer for numbers
    - Most frequent imputer + OneHotEncoder for categories
    """
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_cols)
        ]
    )

def train_and_evaluate():
    df, removed_count, initial_rows = load_and_clean_data(DATA_PATH)

    # Separate target and features
    X = df.drop(columns=["price_lakhs"])
    y = df["price_lakhs"]

    cat_cols = [
        "locality", "zone", "property_type", "furnishing",
        "lift_available", "gated_community", "facing", "rera_approved"
    ]
    num_cols = [c for c in X.columns if c not in cat_cols]

    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
    print(f"Numerical features ({len(num_cols)}): {num_cols}")

    # Compute Locality Statistics for API & Map
    locality_stats = {}
    for loc, group in df.groupby("locality"):
        loc_pps = (group["price_lakhs"] * 100000.0) / group["area_sqft"]
        locality_stats[loc] = {
            "listing_count": int(len(group)),
            "avg_price_lakhs": round(float(group["price_lakhs"].mean()), 2),
            "median_price_lakhs": round(float(group["price_lakhs"].median()), 2),
            "avg_price_per_sqft": round(float(loc_pps.mean()), 1),
            "zone": str(group["zone"].iloc[0])
        }

    with open(os.path.join(MODELS_DIR, "locality_stats.json"), "w", encoding="utf-8") as f:
        json.dump(locality_stats, f, indent=2)
    print("Saved locality_stats.json")

    # 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidate_models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100, max_depth=16, random_state=42, n_jobs=-1
        ),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            random_state=42, max_iter=200, min_samples_leaf=15
        )
    }

    all_metrics = {}
    fitted_pipelines = {}

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, raw_regressor in candidate_models.items():
        print(f"\nEvaluating candidate model: {name}...")
        preprocessor = build_preprocessor(cat_cols, num_cols)
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", raw_regressor)
        ])

        # TransformedTargetRegressor ensures log1p training and expm1 prediction
        ttr = TransformedTargetRegressor(
            regressor=pipe,
            func=np.log1p,
            inverse_func=np.expm1
        )

        # 5-Fold Cross Validation on log-transformed space
        cv_scores_r2 = cross_val_score(ttr, X_train, y_train, cv=cv, scoring="r2")
        cv_scores_mae = -cross_val_score(ttr, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")

        # Fit on 80% train set
        ttr.fit(X_train, y_train)
        fitted_pipelines[name] = ttr

        # Predict on 20% test set
        preds = ttr.predict(X_test)
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(root_mean_squared_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        all_metrics[name] = {
            "test_mae_lakhs": round(mae, 2),
            "test_rmse_lakhs": round(rmse, 2),
            "test_r2": round(r2, 4),
            "cv_r2_mean": round(float(cv_scores_r2.mean()), 4),
            "cv_r2_std": round(float(cv_scores_r2.std()), 4),
            "cv_mae_mean": round(float(cv_scores_mae.mean()), 2),
            "cv_mae_std": round(float(cv_scores_mae.std()), 2)
        }
        print(f" -> {name} Test Results: MAE=Rs {mae:.2f}L | RMSE=Rs {rmse:.2f}L | R2={r2:.4f}")
        print(f" -> 5-Fold CV Mean R2: {cv_scores_r2.mean():.4f} (+/- {cv_scores_r2.std():.4f})")

    # Select Best Model based on Test R2 (or lowest MAE)
    best_name = max(all_metrics.keys(), key=lambda k: all_metrics[k]["test_r2"])
    print(f"\n==========================================")
    print(f"Best Model Selected: {best_name}")
    print(f"==========================================")

    best_pipeline = fitted_pipelines[best_name]

    # Save Best Model Pipeline
    model_save_path = os.path.join(MODELS_DIR, "model.joblib")
    joblib.dump(best_pipeline, model_save_path)
    print(f"Saved best model pipeline to: {model_save_path}")

    # Compute Permutation Feature Importance for Best Model on Test Set
    print("\nComputing Permutation Feature Importance for best model...")
    perm_result = permutation_importance(
        best_pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )

    feature_names = list(X.columns)
    importances = []
    for idx, col in enumerate(feature_names):
        importances.append({
            "feature": col,
            "importance_mean": round(float(perm_result.importances_mean[idx]), 5),
            "importance_std": round(float(perm_result.importances_std[idx]), 5)
        })

    importances.sort(key=lambda x: x["importance_mean"], reverse=True)

    with open(os.path.join(MODELS_DIR, "feature_importance.json"), "w", encoding="utf-8") as f:
        json.dump(importances, f, indent=2)
    print("Saved feature_importance.json")

    # Generate Actual vs Predicted Scatter Sample (300 points for frontend chart)
    best_preds = best_pipeline.predict(X_test)
    sample_indices = np.random.RandomState(42).choice(len(y_test), size=min(300, len(y_test)), replace=False)
    scatter_data = [
        {
            "actual": round(float(y_test.iloc[i]), 2),
            "predicted": round(float(best_preds[i]), 2),
            "locality": str(X_test["locality"].iloc[i]),
            "bhk": int(X_test["bhk"].iloc[i]),
            "area_sqft": int(X_test["area_sqft"].iloc[i])
        }
        for i in sample_indices
    ]

    # Model Summary metrics output
    final_output = {
        "best_model": best_name,
        "is_synthetic_data": True,
        "total_records_initial": initial_rows,
        "outliers_removed": removed_count,
        "training_records": len(X_train),
        "test_records": len(X_test),
        "best_mae_lakhs": all_metrics[best_name]["test_mae_lakhs"],
        "best_rmse_lakhs": all_metrics[best_name]["test_rmse_lakhs"],
        "best_r2": all_metrics[best_name]["test_r2"],
        "cross_val_mae": all_metrics[best_name]["cv_mae_mean"],
        "models": all_metrics,
        "scatter_sample": scatter_data
    }

    metrics_save_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"Saved metrics to: {metrics_save_path}")

    # Save Options and Ranges for Frontend Form Sliders & Dropdowns
    options_data = {
        "localities": sorted(df["locality"].unique().tolist()),
        "zones": sorted(df["zone"].unique().tolist()),
        "property_types": sorted(df["property_type"].unique().tolist()),
        "furnishing": ["Unfurnished", "Semi-Furnished", "Furnished"],
        "facings": ["East", "North", "West", "South", "North-East"],
        "ranges": {
            "bhk": {"min": int(df["bhk"].min()), "max": int(df["bhk"].max()), "default": 2},
            "area_sqft": {"min": int(df["area_sqft"].min()), "max": int(df["area_sqft"].max()), "default": 1200},
            "bathrooms": {"min": int(df["bathrooms"].min()), "max": int(df["bathrooms"].max()), "default": 2},
            "balconies": {"min": int(df["balconies"].dropna().min()), "max": int(df["balconies"].dropna().max()), "default": 1},
            "floor": {"min": int(df["floor"].min()), "max": int(df["floor"].max()), "default": 4},
            "total_floors": {"min": int(df["total_floors"].min()), "max": int(df["total_floors"].max()), "default": 15},
            "age_years": {"min": int(df["age_years"].min()), "max": int(df["age_years"].max()), "default": 5},
            "parking_slots": {"min": int(df["parking_slots"].min()), "max": int(df["parking_slots"].max()), "default": 1},
            "amenities_score": {"min": float(df["amenities_score"].dropna().min()), "max": float(df["amenities_score"].dropna().max()), "default": 7.0},
            "metro_distance_km": {"min": float(df["metro_distance_km"].min()), "max": float(df["metro_distance_km"].max()), "default": 2.5},
            "school_distance_km": {"min": float(df["school_distance_km"].dropna().min()), "max": float(df["school_distance_km"].dropna().max()), "default": 1.2},
            "tech_park_distance_km": {"min": float(df["tech_park_distance_km"].min()), "max": float(df["tech_park_distance_km"].max()), "default": 4.5}
        }
    }
    with open(os.path.join(MODELS_DIR, "options.json"), "w", encoding="utf-8") as f:
        json.dump(options_data, f, indent=2)
    print("Saved options.json")

    print("\nTraining & evaluation completed successfully!")

if __name__ == "__main__":
    train_and_evaluate()
