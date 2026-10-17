# Feature Catalog & Engineering Documentation

This document catalogs all engineered features produced by `src/features/build_features.py` and `src/features/encoding.py`, detailing mathematical definitions, intuition, and leakage-prevention guarantees.

---

## 1. Customer Behavioral Features (E-Commerce)

| Feature Name | Category | Type | Definition / Formula | Business Rationale & Signal |
| :--- | :--- | :--- | :--- | :--- |
| `recency_days` | RFM | Integer | $T_{\text{ref}} - \max(\text{InvoiceDate})$ | Strongest empirical churn predictor. Measures days since last transaction. |
| `frequency_orders` | RFM | Integer | $\text{nunique}(\text{Invoice})$ | Lifetime transaction count. Measures habituation and brand affinity. |
| `monetary_total` | RFM | Float | $\sum (\text{Quantity} \times \text{Price})$ | Cumulative gross customer revenue (£). Primary driver of CLV. |
| `r_score`, `f_score`, `m_score` | RFM | Categorical/Int | Quintile binning ($1 \dots 5$) | Normalized rank scores relative to training cohort. |
| `rfm_score` | RFM | String | $\text{str}(R) + \text{str}(F) + \text{str}(M)$ | Composite RFM signature (e.g. `'555'` = Champion, `'111'` = Lost). |
| `tenure_days` | Tenure | Integer | $\max(t) - \min(t)$ | Total account longevity in days. Distinguishes new trialists from veterans. |
| `tenure_months` | Tenure | Float | $\text{tenure\_days} / 30.44$ | Standardized monthly account age. |
| `spend_per_tenure_day`| Ratio | Float | $\text{monetary\_total} / \max(\text{tenure\_days}, 1)$ | Spend velocity. Detects sudden high-velocity spenders vs. slow accumulators. |
| `avg_basket_value` | Basket | Float | $\text{monetary\_total} / \text{frequency\_orders}$ | Average monetary size per transaction order. |
| `avg_basket_size` | Basket | Float | $\sum \text{Quantity} / \text{frequency\_orders}$ | Average units purchased per transaction order. |
| `item_diversity_ratio`| Ratio | Float | $\text{nunique}(\text{StockCode}) / \sum \text{Quantity}$ | Breadth of catalog exploration. Higher diversity signals cross-category engagement. |
| `purchase_interval_mean`| Time Dynamics| Float | $\text{mean}(\Delta t_{\text{orders}})$ | Average days between consecutive purchases. Establishes expected reorder cycle. |
| `purchase_interval_std` | Time Dynamics| Float | $\text{std}(\Delta t_{\text{orders}})$ | Order cadence volatility. High std indicates irregular, sporadic purchasing. |
| `weekend_purchase_ratio`| Time Dynamics| Float | $\sum \mathbb{I}(\text{Weekend}) / N_{\text{items}}$ | Proportion of purchases made on Saturday/Sunday. Separates B2B from consumer buyers. |
| `q4_purchase_ratio` | Time Dynamics| Float | $\sum \mathbb{I}(\text{Q4}) / N_{\text{items}}$ | Seasonal holiday concentration. High ratio flags holiday-only seasonal shoppers. |
| `is_churned` | Target | Binary | $\mathbb{I}(\text{recency\_days} \ge 90)$ | Ground truth churn target for classification. |

---

## 2. Subscription Contract Features (Telco)

| Feature Name | Category | Type | Description |
| :--- | :--- | :--- | :--- |
| `tenure` | Tenure | Integer | Number of months the customer has stayed with the company. |
| `monthly_charges` | Financial | Float | Monthly recurring subscription charge ($). |
| `total_charges` | Financial | Float | Cumulative billing charges ($). Imputed to $0.0$ for `tenure = 0`. |
| `contract` | Contract | Categorical | Billing commitment term: `Month-to-month`, `One year`, `Two year`. |
| `internet_service` | Product | Categorical | `DSL`, `Fiber optic`, `No`. |
| `payment_method` | Financial | Categorical | `Electronic check`, `Mailed check`, `Bank transfer`, `Credit card`. |
| `security_bundle_score` | Composite | Integer | Sum of active add-on protections (`OnlineSecurity` + `DeviceProtection` + `TechSupport`). |
| `streaming_bundle_score`| Composite | Integer | Sum of entertainment add-ons (`StreamingTV` + `StreamingMovies`). |
| `charges_tenure_ratio` | Ratio | Float | `total_charges / (tenure + 1)`. Identifies unexpected price increases or plan switches. |
| `churn_label` | Target | Binary | Ground truth churn: 1 if terminated, 0 if active. |

---

## 3. Data Leakage Prevention Guarantees

1. **Temporal Cutoff Enforcement**:
   - In transactional feature engineering, statistical bins (`r_bins_`, `f_bins_`, `m_bins_`) and reference dates are computed strictly on training data during `CustomerFeatureTransformer.fit()`.
   - Test and inference records use learned bins to map new observations without recalculating quantiles.
2. **Target Isolation**:
   - Churn labels are strictly excluded from all feature calculations, aggregations, and scaling routines.
3. **Reproducibility**:
   - The transformer is serialized with model weights, guaranteeing that production inferences use identical preprocessing logic as training.
