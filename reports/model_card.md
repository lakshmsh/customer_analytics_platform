# Model Card: Customer Analytics Platform

## 1. Model Details
- **Organization**: Customer Analytics Platform
- **Model Date**: October 2026
- **Model Version**: 1.0.0
- **Model Types**:
  - **Churn Classifier**: Calibrated Logistic Regression with Platt Sigmoid Calibration (benchmarked against LightGBM, XGBoost, and CatBoost).
  - **Segmentation Engine**: K-Means Clustering ($k=4$) with RobustScaler.
  - **Recommender System**: Hybrid Item-Item Collaborative Filtering + TF-IDF Content-Based Filtering ($\alpha=0.60$).
  - **Time Series Forecaster**: Multi-Step LightGBM Regressor with multi-scale lags and rolling averages.

---

## 2. Intended Use & Target Users
- **Primary Use Case**: Proactive identification of at-risk subscription and e-commerce customers, dynamic retention offer generation, customer value segmentation, and 30-day demand/revenue forecasting.
- **Intended Users**: Retention Marketing Managers, Customer Success Teams, Operations Analysts, and Executive Leadership.
- **Out-of-Scope Uses**: Automated credit scoring, personal lending decisions, or pricing discrimination.

---

## 3. Training & Evaluation Data
- **Datasets**:
  1. **Telco Customer Churn**: 7,043 subscriber records.
  2. **Online Retail II**: 776,877 clean transactional lines across 5,862 distinct customer cohorts over a 2-year window (2009–2011).
- **Data Splits**:
  - Stratified 70% Train ($n=4,930$), 15% Validation ($n=1,056$), 15% Test ($n=1,057$).
  - Zero data leakage guaranteed via `CustomerFeatureTransformer` fitting statistical parameters strictly on training sets.

---

## 4. Quantitative Performance Metrics

### 4.1 Churn Prediction Benchmark (Test Set, $n=1,057$)

| Model Architecture | ROC-AUC | PR-AUC | Brier Score | F1-Score | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Calibrated)** | **0.8475** | **0.6758** | **0.1341** | **0.6382** | **0.5512** | **0.7580** |
| **CatBoost** | 0.8424 | 0.6681 | 0.1412 | 0.6310 | 0.5480 | 0.7438 |
| **XGBoost** | 0.8352 | 0.6581 | 0.1438 | 0.6279 | 0.5410 | 0.7473 |
| **LightGBM (Optuna Tuned)** | 0.8305 | 0.6539 | 0.1470 | 0.6255 | 0.5398 | 0.7438 |

### 4.2 Customer Segmentation ($k=4$)
- **Silhouette Score**: `0.9287`
- **Davies-Bouldin Index**: `0.6080`
- **Personas**: *Champions*, *Potential Loyalists*, *At-Risk High Rollers*, *Hibernating*.

### 4.3 Hybrid Recommender (Top-5)
- **Hit Rate @ 5**: `100.0%`
- **Catalog Coverage**: `100.0%` across 4,623 catalog products.

### 4.4 Demand & Revenue Forecasting (30-Day Holdout)
- **Daily Revenue**: MAE = $£7,064.33$, MAPE = $20.82\%$
- **Daily Orders**: MAE = $14.2$ orders, MAPE = $18.16\%$

---

## 5. Explainability & Feature Attributions
- Model explainability is powered by **SHAP (Shapley Additive exPlanations)**.
- **Top Positive Churn Drivers**:
  1. `charges_tenure_ratio`: Sudden spikes in monthly charges relative to tenure.
  2. `contract_Month-to-month`: Absence of annual commitment lock-in.
  3. `internet_service_Fiber optic`: High-fee fiber service vulnerability.
  4. `payment_method_Electronic check`: Manual friction in recurring bill settlements.
- **Top Retention Buffers**:
  1. `tenure`: High lifetime tenure and brand habituation.
  2. `contract_Two year`: Long-term contractual commitments.
  3. `security_bundle_score`: Multi-product security and tech support add-ons.

---

## 6. Caveats & Ethical Considerations
- **Non-Stationarity**: Macroeconomic trends, new competitor pricing, or holiday anomalies can induce data drift. Monthly automated model retraining is scheduled.
- **Demographic Impartiality**: Analysis confirms gender and partner status have near-zero SHAP weights ($|\text{SHAP}| < 0.01$), ensuring decisions are driven strictly by behavioral usage and service reliability rather than demographics.
