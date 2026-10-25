# NammaMane (ನಮ್ಮ ಮನೆ) — Implementation Plan & Architecture

> [!NOTE]
> **Project Context:** NammaMane is a full-stack ML application pairing a beginner-friendly learning experience with production-grade engineering. It predicts Bengaluru residential property prices in Lakhs (₹ Lakhs) using a curated synthetic dataset of 15,000 listings across 25 Bengaluru localities.

---

## 1. Dataset Analysis & Preparation

### Dataset Highlights:
- **Source:** Synthetic dataset (`data/bengaluru_house_prices_15000.csv`)
- **Dimensions:** 15,000 rows, 22 columns
- **Localities:** 25 Bengaluru zones/localities (Indiranagar, Koramangala, Whitefield, Electronic City, Malleshwaram, Hebbal, HSR Layout, etc.)
- **Target Variable:** `price_lakhs` (transformed to $\log(1 + y)$ during training and inverted via $\text{expm1}$)

### Verified Data Cleaning Strategy:
1. **Drop Identifiers:** Exclude `property_id` from model training.
2. **Missing Value Imputation:**
   - Categorical columns (`furnishing`: 221 missing, `facing`: 734 missing) $\rightarrow$ `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore')`
   - Numerical columns (`balconies`: 476 missing, `amenities_score`: 595 missing, `school_distance_km`: 309 missing) $\rightarrow$ `SimpleImputer(strategy='median')`
3. **Outlier Detection & Removal (IQR on Price per Sqft):**
   - Calculate $\text{price\_per\_sqft} = (\text{price\_lakhs} \times 100,000) / \text{area\_sqft}$.
   - Apply interquartile range filtering ($Q_1 - 1.8 \times \text{IQR}$ to $Q_3 + 1.8 \times \text{IQR}$), precisely isolating and removing the ~35–40 injected corrupt price anomalies while preserving valid market variations.
   - Print exact count and summary stats of dropped rows.

---

## 2. System Architecture & Tech Stack

```mermaid
graph TD
    subgraph Frontend ["React + Vite + TypeScript Frontend"]
        UI[Living Skyline & Bento UI]
        Pages[6 Dedicated Pages: Predict, What-If, Deal Check, Duel, Map, Model Lab]
        I18n[Bilingual Toggle: English / Kannada]
        State[API Client with Axios / React Hooks]
    end

    subgraph Backend ["FastAPI Backend (Python 3.11)"]
        API[FastAPI Application]
        Routes[Routes: /predict, /whatif, /compare, /deal-check, /locality-stats, /model-report]
        Validation[Pydantic V2 Schemas]
    end

    subgraph ML_Engine ["Scikit-Learn ML Pipeline"]
        Preprocessor[ColumnTransformer Imputers + OneHotEncoder]
        Models[Linear Regression | Random Forest | HistGradientBoosting]
        CV[5-Fold Cross Validation + Log-Scale Training]
        Artifacts[(model.joblib, metrics.json, feature_importance.json)]
    end

    UI --> State
    State -->|HTTP / JSON| API
    API --> Validation
    Validation --> Routes
    Routes --> Artifacts
    ML_Engine --> Artifacts
```

---

## 3. UI/UX Design Directions

We have generated high-resolution visual design mockups for two distinct creative directions:

````carousel
![Design Direction A - Golden Vidhana: Warm Dawn Skyline & Architectural Elegance](file:///C:/Users/mm917/.gemini/antigravity-ide/brain/f446eae4-57ae-479f-b501-cb99fd93b094/design_direction_a_1792952215681.jpg)
<!-- slide -->
![Design Direction B - Silicon Garden Bento: Modern Dark Mode & Dynamic Vector Skyline](file:///C:/Users/mm917/.gemini/antigravity-ide/brain/f446eae4-57ae-479f-b501-cb99fd93b094/design_direction_b_1792952242562.jpg)
````

### Comparison of Design Directions:

| Feature | Direction A: "Golden Vidhana" (Warm Heritage) | Direction B: "Silicon Garden Bento" (Modern Tech) |
| :--- | :--- | :--- |
| **Theme & Atmosphere** | Warm golden dawn, Jasmine white (`#FFF9EF`), deep indigo (`#1F2A5A`), marigold (`#F5A524`). Cozy, celebratory home-buying feel. | Sleek midnight indigo (`#1F2A5A`), luminous neon marigold (`#F5A524`), and glowing soft teal accents. Cyber-garden tech feel. |
| **Living Skyline** | Stylized illustrated Bengaluru landmarks (Vidhana Soudha dome, Cubbon Park trees, UB City towers) with sunlight warmth scaling dynamically with price. | Minimalist neon-vector interactive outline skyline with glowing heights and pulsing ambient illumination. |
| **Layout** | Elegant card-based flow with generous white space and clean typography. | Bento grid layout with compact high-density interactive widgets and smooth spring transitions. |
| **Target Appeal** | Friendly, accessible for all family members & homebuyers. | Cutting-edge, developer & recruiter WOW-factor for AI/ML engineering portfolios. |

> [!TIP]
> Both options support full light/dark switching and English/Kannada bilingual toggles.

---

## 4. Phased Task Breakdown

### Phase 1: Virtual Environment & Backend Setup
- [ ] Configure Python virtual environment and install `fastapi`, `uvicorn`, `pandas`, `scikit-learn`, `joblib`, and `pytest`.
- [ ] Organize project directory:
  ```text
  d:\Workspace\namma-mane\
  ├── backend\
  │   ├── app\
  │   │   ├── main.py
  │   │   ├── schemas.py
  │   │   └── utils.py
  │   ├── ml\
  │   │   ├── train.py
  │   │   └── evaluate.py
  │   ├── models\
  │   │   ├── model.joblib
  │   │   ├── metrics.json
  │   │   └── feature_importance.json
  │   └── tests\
  │       └── test_api.py
  ├── data\
  │   ├── bengaluru_house_prices_15000.csv
  │   └── bengaluru_coordinates.json
  └── frontend\
      ├── src\
      │   ├── components\
      │   ├── pages\
      │   ├── i18n\
      │   └── App.tsx
      └── ...
  ```

### Phase 2: ML Pipeline & Multi-Model Benchmarking (`train.py`)
- [ ] Load data, clean IQR price-per-sqft outliers, print row counts removed.
- [ ] Construct scikit-learn `Pipeline` with `ColumnTransformer` (median imputers for numerical features, most-frequent imputer + `OneHotEncoder` for categoricals).
- [ ] Train target on $\log(\text{price\_lakhs})$ and predict back with $\text{expm1}$.
- [ ] Compare 3 models using an 80/20 train-test split and 5-fold cross-validation:
  1. **LinearRegression** (baseline)
  2. **RandomForestRegressor**
  3. **HistGradientBoostingRegressor**
- [ ] Calculate & compare MAE (₹ Lakhs), RMSE, and $R^2$.
- [ ] Compute permutation feature importance on the best model.
- [ ] Serialize best model to `backend/models/model.joblib` and metrics to `metrics.json` & `feature_importance.json`.
- [ ] Write student-friendly explanation and Model Card in `MODEL_CARD.md`.

### Phase 3: FastAPI Backend & API Endpoints (`main.py`)
- [ ] Implement `GET /health` — health check.
- [ ] Implement `GET /options` — dropdown options, min/max slider ranges.
- [ ] Implement `POST /predict` — predicted price, $\pm \text{MAE}$ confidence range, price per sqft, and feature contribution breakdown ("Why this price?").
- [ ] Implement `POST /whatif` — base house vs list of hypothetical changes (metro proximity, extra BHK, newer age).
- [ ] Implement `POST /compare` — side-by-side comparison of 2 properties with winning attributes.
- [ ] Implement `POST /deal-check` — fair deal assessment: "Bargain", "Fair", or "Overpriced" with percentage deviation.
- [ ] Implement `GET /locality-stats` — average price/sqft and listings count per locality.
- [ ] Implement `GET /model-report` — model comparison metrics, cross-validation stats, and feature importance.
- [ ] Write comprehensive `pytest` test suite covering all endpoints and edge cases (invalid inputs returning 422).

### Phase 4: Frontend Development (React + Vite + TypeScript)
- [ ] Initialize Vite React + TS app with Tailwind CSS, Lucide icons, Framer Motion, and Recharts.
- [ ] Implement core Design System (colors: Indigo `#1F2A5A`, Marigold `#F5A524`, Jasmine White `#FFF9EF`, soft Teal).
- [ ] Build Living SVG Bengaluru Skyline component (Vidhana Soudha, tech parks, trees) that reacts dynamically to price updates.
- [ ] Develop 6 core views:
  1. **Home / Predict:** Interactive property valuation form with animated count-up price, confidence range bar, and "Why this price?" breakdown.
  2. **What-if Lab:** Real-time sliders with debounced API queries and visual chips (e.g. `+₹6.2L near metro`).
  3. **Deal Checker:** Asking price comparison with animated speedometer needle (Bargain / Fair / Overpriced).
  4. **House Duel (Compare):** Side-by-side battle card showing two houses with category winner badges.
  5. **Bengaluru Map:** Leaflet map with interactive locality bubbles color-coded by price per sqft.
  6. **Model Lab:** Transparent metrics dashboard displaying $R^2$, MAE, feature importance chart, and actual vs predicted scatter plot.
- [ ] Add English $\leftrightarrow$ Kannada bilingual toggle across all UI text.
- [ ] Add permanent synthetic data disclosure banner: *"Prices come from a model trained on a synthetic dataset. Not financial advice."*

### Phase 5: Testing, Validation & Final Handover
- [ ] Run backend tests (`pytest`).
- [ ] Run browser testing with visual verification across all 6 pages.
- [ ] Create detailed, beginner-friendly `README.md` explaining how the ML model and application work.
