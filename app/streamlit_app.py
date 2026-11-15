"""
Customer Analytics Platform - Main Executive Overview.

This multi-page Streamlit application serves as the centralized decision-support
frontend for the end-to-end Machine Learning system:
1. Churn Risk Scoring & Real-Time Simulation (Phase 5)
2. Behavioral Customer Segmentation & RFM Personas (Phase 5)
3. Personalized Hybrid Recommendations & Dynamic Retention Offers (Bonus 1)
4. Forward Demand & Revenue Forecasting (Bonus 2)
5. Business Impact & SHAP Explainability Engine (Phase 6 / Bonus)
"""

import streamlit as st
from pathlib import Path
import pandas as pd

from app.components.ui_elements import (
    load_custom_css,
    render_header,
    render_metric_card
)
from app.components.api_client import AnalyticsClient

# Configure page settings
st.set_page_config(
    page_title="Customer Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_custom_css()

# ---------------------------------------------------------
# Sidebar System Health & Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ System Status")
    health = AnalyticsClient.get_health()

    if health["status"] == "healthy":
        st.success(f"● System Status: Healthy\nEngine: {health.get('source', 'Online')}")
    else:
        st.warning(f"● System Status: Degraded\nEngine: {health.get('source', 'Local')}")

    st.markdown("#### 🧠 Registered Models")
    for m in health.get("models_loaded", []):
        st.markdown(f"- ✅ **{m}**")

    st.markdown("---")
    st.markdown("### 🔗 Quick Resources")
    st.markdown(
        """
        - [📄 Model Card Report](https://github.com/lakshmsh/customer_analytics_platform/blob/main/reports/model_card.md)
        - [💼 Business Impact Summary](https://github.com/lakshmsh/customer_analytics_platform/blob/main/reports/business_impact_summary.md)
        - [🔌 FastAPI Swagger Docs (Local)](http://127.0.0.1:8000/docs)
        - [🐙 GitHub Repository](https://github.com/lakshmsh/customer_analytics_platform)
        """
    )
    st.markdown("---")
    st.caption("Customer Analytics Platform v1.0.0 • MLflow & FastAPI Architecture")

# ---------------------------------------------------------
# Main Page Content
# ---------------------------------------------------------
render_header(
    title="Customer Analytics & Revenue Intelligence Platform",
    subtitle="End-to-End Enterprise System for Churn Mitigation, Persona Segmentation, Recommender Systems, and Demand Forecasting",
    icon="🚀"
)

# Executive KPI Ribbon
col1, col2, col3, col4 = st.columns(4)
render_metric_card(col1, "Historical Churn Rate", "26.54%", "Telco Customer Base Benchmark")
render_metric_card(col2, "Persona Clusters", "4 Tiers", "Recency, Frequency, Monetary & Basket")
render_metric_card(col3, "30-Day Projected Demand", "$1,059,184", "Forward Revenue Baseline")
render_metric_card(col4, "Projected Retention ROI", "$43,839", "Estimated Net Savings / 30 Days")

st.markdown("---")

# Feature Showcase Grid
st.markdown("### 🧭 Platform Modules & Workflows")

card1, card2 = st.columns(2)
with card1:
    st.markdown(
        """
        <div class="metric-card" style="min-height: 200px;">
            <div style="font-size: 1.25rem; font-weight: 700; color: #38bdf8; margin-bottom: 0.5rem;">
                🚨 1. Churn Risk Predictor
            </div>
            <p style="color: #cbd5e1; font-size: 0.95rem;">
                Scores individual customer profiles with calibrated Platt Sigmoid probabilities.
                Includes interactive risk tier assignment (High, Medium, Low) and CSV bulk-scoring engine.
            </p>
            <p style="font-size: 0.85rem; color: #94a3b8;">
                👉 Open <b>1_Churn_Risk</b> from the sidebar to test custom customer scenarios.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="metric-card" style="min-height: 200px;">
            <div style="font-size: 1.25rem; font-weight: 700; color: #a855f7; margin-bottom: 0.5rem;">
                🎁 3. Smart Product Recommender
            </div>
            <p style="color: #cbd5e1; font-size: 0.95rem;">
                Combines collaborative filtering (customer co-purchases) and TF-IDF content similarity.
                Dynamically injects customized retention discounts when a customer is flagged at risk of churn.
            </p>
            <p style="font-size: 0.85rem; color: #94a3b8;">
                👉 Open <b>3_Smart_Recommendations</b> to generate personalized product recommendations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with card2:
    st.markdown(
        """
        <div class="metric-card" style="min-height: 200px;">
            <div style="font-size: 1.25rem; font-weight: 700; color: #10b981; margin-bottom: 0.5rem;">
                👥 2. Behavioral Segmentation
            </div>
            <p style="color: #cbd5e1; font-size: 0.95rem;">
                Robust K-Means clustering across 6 RFM + basket dimensions with automated persona tagging
                (High-Value Champions, Loyalists, At-Risk, Budget Shoppers) and tailored marketing playbooks.
            </p>
            <p style="font-size: 0.85rem; color: #94a3b8;">
                👉 Open <b>2_Customer_Segments</b> to explore persona benchmarks and radar plots.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="metric-card" style="min-height: 200px;">
            <div style="font-size: 1.25rem; font-weight: 700; color: #f59e0b; margin-bottom: 0.5rem;">
                💰 4 & 5. Demand Forecast & Business Impact (SHAP)
            </div>
            <p style="color: #cbd5e1; font-size: 0.95rem;">
                Forward revenue time-series forecasting combined with SHAP waterfall explanations,
                prescriptive business interventions, and optimal decision threshold ROI optimization.
            </p>
            <p style="font-size: 0.85rem; color: #94a3b8;">
                👉 Open <b>4_Demand_Revenue_Forecast</b> and <b>5_Business_Impact</b> for full financial simulations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Architecture & Production Model Highlights
st.markdown("### 🏆 Production Machine Learning Performance")
tab_perf, tab_eda = st.tabs(["📊 Model Performance Benchmarks", "📈 Exploratory Data Visualizations"])

with tab_perf:
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    render_metric_card(pcol1, "Churn ROC-AUC", "0.851", "Calibrated Platt Sigmoid (Logistic/GBM)")
    render_metric_card(pcol2, "Churn PR-AUC", "0.668", "Significantly outperforms 0.26 baseline")
    render_metric_card(pcol3, "Segmentation Silhouette", "0.384", "4 Distinct Non-Overlapping Personas")
    render_metric_card(pcol4, "Forecasting R²", "0.87", "Forward Demand Time-Series Regressor")

    st.info(
        "💡 **Key Business Finding**: Cost-benefit threshold optimization reveals that lowering the churn intervention "
        "threshold from **0.50 down to 0.35** captures an additional **18.4% of at-risk revenue**, maximizing net "
        "annualized customer lifetime value."
    )

with tab_eda:
    fig_dir = Path("reports/figures")
    f_cols = st.columns(3)
    figs = [
        ("01_churn_distribution.png", "Target Churn Distribution"),
        ("03_churn_by_tenure_contract.png", "Churn by Tenure & Contract"),
        ("08_shap_local_waterfall.png", "Local SHAP Waterfall Attribution")
    ]
    for idx, (fname, label) in enumerate(figs):
        fpath = fig_dir / fname
        with f_cols[idx]:
            if fpath.exists():
                st.image(str(fpath), caption=label, use_container_width=True)
            else:
                st.caption(f"Image {fname} not found on disk.")
