# Model Card: NammaMane Bengaluru House Price Predictor

## Model Details
- **Developer:** NammaMane Team (Bengaluru House Price Prediction)
- **Model Architecture:** Scikit-Learn `Pipeline` wrapping `ColumnTransformer` (median imputers for numerical attributes, most-frequent imputers and `OneHotEncoder` for categorical attributes) and `HistGradientBoostingRegressor` with target log-transformation (`TransformedTargetRegressor` with `np.log1p` and `np.expm1`).
- **Trained Target:** Property price in Lakhs of Indian Rupees (`price_lakhs`).
- **Benchmark Comparison:** Evaluated alongside baseline `LinearRegression` and `RandomForestRegressor`.

## Intended Use
- **Primary Use:** Educational demonstration, interactive property price estimation, what-if scenario testing, and property comparison for students and tech buyers in Bengaluru.
- **Out of Scope:** Financial speculation, formal bank mortgage underwriting, or legal property valuation.

## Training Data & Synthetic Disclosure
- **Dataset:** Synthetic dataset of 15,000 listings across 25 Bengaluru localities (`data/bengaluru_house_prices_15000.csv`).
- **Synthetic Statement:** **This dataset is entirely synthetic.** It was algorithmically generated with Bengaluru-aligned pricing patterns and is **not real market transaction data**.
- **Data Preprocessing & Cleaning:**
  - Non-predictive identifier `property_id` was dropped.
  - Missing values in `balconies` (476), `furnishing` (221), `amenities_score` (595), `school_distance_km` (309), and `facing` (734) were imputed systematically using median and most-frequent strategies.
  - Price per square foot outlier filtering (2.5 $\times$ IQR per locality) was applied to remove 47 artificially injected corrupt price rows.

## Quantitative Performance
| Model | Test MAE (₹ Lakhs) | Test RMSE (₹ Lakhs) | Test $R^2$ | 5-Fold CV $R^2$ |
| :--- | :--- | :--- | :--- | :--- |
| **HistGradientBoosting (Best)** | **₹ 8.00 L** | **₹ 11.15 L** | **0.9628** | **0.9614 $\pm$ 0.0018** |
| Linear Regression (Baseline) | ₹ 8.51 L | ₹ 12.07 L | 0.9565 | 0.9540 $\pm$ 0.0024 |
| Random Forest Regressor | ₹ 12.16 L | ₹ 16.94 L | 0.9142 | 0.9185 $\pm$ 0.0029 |

- **Top Predictive Features (Permutation Importance):**
  1. `locality` (Importance: 1.003)
  2. `area_sqft` (Importance: 0.717)
  3. `property_type` (Importance: 0.029)
  4. `age_years` (Importance: 0.021)
  5. `tech_park_distance_km` (Importance: 0.012)

## Known Limitations
- The model reflects synthetic heuristics and does not account for micro-factors such as legal title clarity, builder reputation, Cauvery water connection availability, or road width.
- Extrapolations beyond standard area boundaries (< 400 sq.ft or > 5000 sq.ft) have increased uncertainty.
