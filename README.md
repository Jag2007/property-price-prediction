# Property Price Prediction

### Intelligent Real Estate Valuation & Investment Grade Classification via Feature Engineering and XGBoost

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Supported-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-0F9D58?style=flat-square)](https://xgboost.readthedocs.io/)
[![Live App](https://img.shields.io/badge/Live%20App-Open-1C1C1C?style=flat-square)](https://property-price-prediction-real-estate.streamlit.app/)

Hosted Streamlit URL: <https://property-price-prediction-real-estate.streamlit.app/>

---

## Overview

Real-estate decisions depend on two hard questions:

1. What is the **fair current market price** of a property?
2. Is this property a good **investment grade** option?

This project provides an end-to-end ML pipeline that automates both predictions using XGBoost:

- **Regression task:** predicts `Current_Market_Price`
- **Classification task:** predicts `Investment_Grade` (0, 1, 2)

The full workflow covers preprocessing, feature engineering, model training, evaluation, artifact persistence, and real-time inference through Streamlit.

### Problem Statement

| Challenge                        | Scope                                           |
| -------------------------------- | ----------------------------------------------- |
| Manual valuation inconsistencies | Property pricing varies by feature interactions |
| Multi-objective prediction       | Continuous price + categorical investment grade |
| Data quality issues              | Missing values, categorical features, outliers  |
| Deployment requirement           | Real-time predictions for unseen properties     |

---

## System Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                         PROPERTY PRICE PREDICTION SYSTEM                                 │
│                  Xgboost Regression + Classification Pipeline                            │
└──────────────────────────────────────────────────────────────────────────────────────────┘

                                  DATA SOURCE
                          ──────────────────────────
                               real_estate_raw.csv
                    (Property attributes + price + investment grade)

╔════════════════════════════════════ TRAINING PIPELINE ═══════════════════════════════════╗

        ┌────────────────────┐
        │ Data Loading       │
        │ real_estate_raw.csv
        └─────────┬──────────┘
                  ▼
        ┌─────────────────────────────────────────────────────┐
        │ Preprocessing                                       │
        │-----------------------------------------------------│
        │ • Median imputation for numeric missing values      │
        │ • Ordinal encoding: Furnishing_Status (0/1/2)       │
        │ • One-hot encoding: Neighborhood (drop-first)       │
        │ • IQR clipping for outliers (except target columns) │
        └─────────┬───────────────────────────────────────────┘
                  ▼
        ┌────────────────────────────────────────────────────────────┐
        │ Machine Learning Models                                   │
        │------------------------------------------------------------│
        │ XGBRegressor: Predict Current_Market_Price                │
        │ XGBClassifier: Predict Investment_Grade                   │
        └─────────┬──────────────────────────────────────────────────┘
                  ▼
        ┌────────────────────────────────────────────────────────────┐
        │ Model Persistence                                          │
        │ xgb_regression_model.joblib                               │
        │ regression_scaler.joblib                                  │
        │ xgb_classification_model.joblib                           │
        │ classification_scaler.joblib                              │
        └────────────────────────────────────────────────────────────┘

╚═══════════════════════════════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════ INFERENCE PIPELINE ══════════════════════════════════╗

        User Input (app/streamlit_app.py)
        (Manual form or CSV batch input)
                    │
                    ▼
        ┌──────────────────────┐
        │ Load Models & Scalers│
        │ from /models          │
        └─────────┬────────────┘
                  ▼
        ┌──────────────────────────────────────────┐
        │ Feature Construction                      │
        │------------------------------------------│
        │ • Numeric feature validation              │
        │ • Furnishing ordinal mapping              │
        │ • Neighborhood one-hot transformation     │
        └─────────┬────────────────────────────────┘
                  ▼
        ┌──────────────────────────────────────────┐
        │ Prediction Engine                         │
        │------------------------------------------│
        │ • scaler.transform(features)              │
        │ • reg_model.predict()                     │
        │ • clf_model.predict() + predict_proba()   │
        └─────────┬────────────────────────────────┘
                  ▼
        ┌──────────────────────────────────────────┐
        │ Outputs                                  │
        │------------------------------------------│
        │ • Predicted Current Market Price         │
        │ • Predicted Investment Grade             │
        │ • Class probability distribution          │
        └──────────────────────────────────────────┘

╚═══════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## Repository Structure

```text
property-price-prediction/
├── .streamlit/
│   └── config.toml
├── README.md
├── requirements.txt
├── runtime.txt
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   └── real_estate_raw.csv
│   └── processed/
│       └── real_estate_clean.csv
│
├── models/
│   ├── xgb_regression_model.joblib
│   ├── regression_scaler.joblib
│   ├── xgb_classification_model.joblib
│   └── classification_scaler.joblib
│
├── notebooks/
│   └── training_colab.ipynb
│
└── report/
    ├── property_price_prediction_report.pdf
    └── property_price_prediction_report.tex
```

---

## Quickstart

### 1. Enter the project

```bash
cd property-price-prediction
```

### 2. Set up environment

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launch the app

```bash
streamlit run app/streamlit_app.py
# Opens at http://localhost:8501
```

### 4. Re-train models (optional)

Use your training notebook/script to retrain, then export artifacts with these exact names in `models/`:

- `xgb_regression_model.joblib`
- `regression_scaler.joblib`
- `xgb_classification_model.joblib`
- `classification_scaler.joblib`

---

## Data Processing Pipeline

### 1. Data Loading and Initial Inspection

- Loaded `real_estate_raw.csv` into a pandas DataFrame
- Checked dataset shape and sample rows to verify schema

### 2. Data Preprocessing

- **Handling Missing Values**
  - Numeric columns were imputed with median values
- **Categorical Encoding**
  - `Furnishing_Status` ordinal mapping:
    - `Unfurnished -> 0`
    - `Semi-furnished -> 1`
    - `Fully-furnished -> 2`
  - `Neighborhood` one-hot encoded with drop-first strategy
- **Outlier Treatment**
  - IQR-based clipping applied to numeric columns excluding:
    - `Current_Market_Price`
    - `Investment_Grade`
- **Saving Cleaned Data**
  - Exported to `data/processed/real_estate_clean.csv`

### 3. XGBoost Regression (Current_Market_Price)

- **Target:** `Current_Market_Price`
- **Features:** all engineered predictors except `Investment_Grade`
- **Split:** 80% train / 20% test (`random_state=42`)
- **Scaling:** `MinMaxScaler`
- **Model:** `XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)`
- **Evaluation:** R2, MAE, RMSE

### 4. XGBoost Classification (Investment_Grade)

- **Target:** `Investment_Grade`
- **Features:** all engineered predictors excluding both:
  - `Investment_Grade`
  - `Current_Market_Price` (to prevent leakage)
- **Split:** 80% train / 20% test (`random_state=42`)
- **Scaling:** `MinMaxScaler`
- **Model:** `XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, eval_metric='mlogloss')`
- **Evaluation:** Accuracy, classification report, confusion matrix

### 5. Model Saving and Inference Readiness

- Saved model and scaler artifacts using `joblib`
- Artifacts are directly consumed by `app/streamlit_app.py`

---

## Models

### Regression Model

- **Algorithm:** XGBoost Regressor
- **Goal:** Predict market price from property attributes
- **Artifacts:**
  - `models/xgb_regression_model.joblib`
  - `models/regression_scaler.joblib`

### Classification Model

- **Algorithm:** XGBoost Classifier
- **Goal:** Predict investment grade class (0/1/2)
- **Artifacts:**
  - `models/xgb_classification_model.joblib`
  - `models/classification_scaler.joblib`

---

## Results

Metrics below are from a previous training run of this pipeline:

| Metric          | Regression       | Classification |
| --------------- | ---------------- | -------------- |
| **R2 Score**    | **0.9483**       | -              |
| **MAE**         | **550,850.82**   | -              |
| **RMSE**        | **1,010,636.01** | -              |
| **Accuracy**    | -                | **97.50%**     |
| **Weighted F1** | -                | **0.97**       |

> Note: Classification report in the notebook shows strong precision/recall across all three classes, with relatively lower recall on class `2` compared to classes `0` and `1`.

---

## Application

The Streamlit application (`app/streamlit_app.py`) includes:

### 1. Predict

- Manual property input form
- Simultaneous prediction of:
  - `Current_Market_Price`
  - `Investment_Grade`
- Class probability visualization

### 2. Batch Predict

- CSV upload for bulk inference
- Supports:
  - fully encoded feature columns, or
  - raw columns with `Furnishing_Status` and `Neighborhood`
- Downloadable predictions CSV

### 3. About

- Summary of preprocessing, model training, and deployment flow

---

## Inference Pipeline (Code-Aligned)

The deployed inference flow in `app/streamlit_app.py` uses these functions:

1. `load_artifacts(...)`
2. `default_values(df, feature_columns)`
3. `build_feature_row(...)` for manual input
4. `raw_to_feature_frame(...)` for CSV raw input
5. `run_predict(features, reg_model, reg_scaler, clf_model, clf_scaler)`
6. `page_manual(...)` and `page_csv(...)` UI handlers

Artifact constants used by inference:

- `REG_MODEL_PATH  -> models/xgb_regression_model.joblib`
- `REG_SCALER_PATH -> models/regression_scaler.joblib`
- `CLF_MODEL_PATH  -> models/xgb_classification_model.joblib`
- `CLF_SCALER_PATH -> models/classification_scaler.joblib`

---

## Limitations

- The model behavior depends on the training data distribution in the provided dataset.
- Investment grade labels are class IDs (`0/1/2`), so business meaning should be mapped explicitly in production.
- Extreme real-world market shifts may require frequent retraining.

---

## Future Enhancements

- Add model monitoring in the hosted Streamlit app (input drift, prediction drift, and confidence trends) with periodic retraining triggers.
- Expand feature set with location-intelligence signals (pincode/zone index, nearby amenities, transport score) to improve valuation robustness.
- Introduce explainability in predictions (feature contribution view for each estimate) so users can understand why a price/grade was predicted.
- Build an investor advisory layer on top of current outputs that converts predicted price + investment grade into buy/hold/avoid guidance with risk notes.
