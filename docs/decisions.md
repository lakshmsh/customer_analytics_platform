# Architecture & Scope Decision Log

## Decision 1: Scope & Bonus Commitments
**Status:** Accepted & Locked  
**Context:** The Customer Analytics Platform requires both core ML functionality and advanced business-grade extensions.  
**Decision:** We formally commit to implementing the core pipeline along with **Bonus Points 1, 2, 4, and 5**:

1. **Bonus 1 - Advanced Recommender & Multi-Engine FastAPI Service**:
   - Deliver a hybrid collaborative + content-based recommendation engine.
   - Expose all 4 model capabilities (churn, segmentation, recommendation, forecasting) via production FastAPI REST endpoints with Pydantic validation.
2. **Bonus 2 - Deep Explainability with SHAP**:
   - Provide local waterfall/force plots and global beeswarm importance.
   - Generate interpretable reason codes per customer to explain churn risk to non-technical stakeholders.
3. **Bonus 4 - Actionable Business Retention & Financial ROI Simulator**:
   - Map machine learning risk scores to concrete marketing playbooks (promotional discounts, loyalty tiers, proactive check-ins).
   - Real-time what-if financial calculator estimating net revenue saved vs. campaign costs.
4. **Bonus 5 - Interactive Multi-Page Production Dashboard & Docker Deployment**:
   - Streamlit multi-page interface featuring interactive customer lookup, batch CSV uploads, and executive summaries.
   - Complete Docker containerization (`docker-compose.yml`) for seamless deployment.

---

## Decision 2: Dual Dataset Strategy
**Status:** Accepted  
**Context:** We need realistic data that covers both subscription/contractual churn dynamics and transactional basket/recommendation behaviors.  
**Decision:**
- **Online Retail II (E-Commerce)**: Primary transactional backbone for basket analysis, RFM segmentation, collaborative recommendations, and daily time series demand forecasting.
- **Telco Customer Churn**: Supplementary contractual churn benchmark for evaluating subscription retention dynamics and customer service interactions.

---

## Decision 3: Time-Aware Cross-Validation
**Status:** Accepted  
**Context:** Standard random k-fold cross validation causes temporal lookahead leakage in transactional churn and demand forecasting models.  
**Decision:**
- Enforce strict cutoff date splits (train on historical window, evaluate on subsequent holdout window) for churn and time-series forecasting.
