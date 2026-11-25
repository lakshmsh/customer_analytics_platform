"""
Forward Demand & Revenue Forecasting Dashboard (Bonus 2).

Provides 7-to-90-day time-series forecasting of incoming revenue and order volumes,
and simulates the financial lift achieved by proactive churn-mitigation campaigns.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

from app.components.ui_elements import (
    load_custom_css,
    render_header,
    render_metric_card,
    plot_revenue_forecast_chart,
)
from app.components.api_client import AnalyticsClient

st.set_page_config(page_title="Demand & Revenue Forecasting", page_icon="📈", layout="wide")
load_custom_css()

render_header(
    title="Demand & Revenue Forecasting Engine",
    subtitle="Forward time-series demand projections and dynamic churn revenue-at-risk simulation",
    icon="📈"
)

# Controls
ctrl_col1, ctrl_col2 = st.columns([1, 2])

with ctrl_col1:
    horizon = st.select_slider(
        "Forecast Horizon (Days ahead):",
        options=[7, 14, 30, 60, 90],
        value=30
    )

with ctrl_col2:
    st.caption("Forecasting models utilize lag features, rolling windows, and seasonal day-of-week indicators.")

# Fetch Forecast Data
with st.spinner("Calculating time-series forecast projections..."):
    forecast_data = AnalyticsClient.get_forecast(horizon_days=horizon)

base_rev = float(forecast_data.get("projected_baseline_revenue", 1059184.46))
at_risk_rev = float(forecast_data.get("unmitigated_churn_revenue_at_risk", 132398.06))
saved_rev = float(forecast_data.get("estimated_revenue_saved", 46339.32))
cost = float(forecast_data.get("estimated_campaign_cost", 2500.0))
net_roi = float(forecast_data.get("estimated_net_roi", 43839.32))

# Scale values proportionally if horizon is different from 30 days
scale_factor = horizon / 30.0
base_rev_scaled = base_rev * scale_factor
at_risk_scaled = at_risk_rev * scale_factor
saved_scaled = saved_rev * scale_factor
cost_scaled = cost * scale_factor
net_roi_scaled = saved_scaled - cost_scaled

# Financial Metrics Ribbon
m1, m2, m3, m4 = st.columns(4)
render_metric_card(m1, f"{horizon}-Day Projected Revenue", f"${base_rev_scaled:,.0f}", "Baseline Demand Projection")
render_metric_card(m2, "Revenue at Churn Risk", f"${at_risk_scaled:,.0f}", "Estimated Gross Risk (12.5% churn)")
render_metric_card(m3, "Protected Revenue", f"${saved_scaled:,.0f}", "Assuming 35% Retention Success")
render_metric_card(m4, "Net Campaign ROI", f"${net_roi_scaled:,.0f}", f"After ${cost_scaled:,.0f} outreach cost")

st.markdown("---")

# Main Forecast Chart
st.markdown(f"### 📊 Forward {horizon}-Day Revenue Projection Curve")
st.plotly_chart(
    plot_revenue_forecast_chart(horizon_days=horizon, baseline_revenue=base_rev_scaled, revenue_saved=saved_scaled),
    use_container_width=True
)

st.markdown("---")

# Interactive Scenario Simulator
st.markdown("### 🎛️ Churn Impact & Retention Scenario Simulator")
st.caption("Adjust business parameters to simulate different churn severity levels and retention budget outcomes:")

s_col1, s_col2, s_col3 = st.columns(3)
with s_col1:
    churn_rate_sim = st.slider("Simulated Customer Churn Rate (%)", min_value=5.0, max_value=40.0, value=12.5, step=0.5)
with s_col2:
    success_rate_sim = st.slider("Retention Campaign Success Rate (%)", min_value=10.0, max_value=60.0, value=35.0, step=5.0)
with s_col3:
    cost_per_cust = st.number_input("Intervention Cost per Customer ($)", min_value=5.0, max_value=100.0, value=25.0, step=5.0)

# Re-simulate
sim_risk = base_rev_scaled * (churn_rate_sim / 100.0)
sim_saved = sim_risk * (success_rate_sim / 100.0)
sim_cost = (sim_risk / 1200.0) * cost_per_cust  # Assumed CLV $1200
sim_net = sim_saved - sim_cost

sim_c1, sim_c2, sim_c3 = st.columns(3)
render_metric_card(sim_c1, "Simulated Revenue at Risk", f"${sim_risk:,.0f}", f"{churn_rate_sim}% of total baseline")
render_metric_card(sim_c2, "Recovered Revenue", f"${sim_saved:,.0f}", f"{success_rate_sim}% recovery efficiency")
render_metric_card(sim_c3, "Simulated Net Profit Lift", f"${sim_net:,.0f}", f"Net ROI: {((sim_net / max(sim_cost, 1)) * 100):.0f}%")

st.markdown("---")

# Historical Demand Trend Preview
st.markdown("#### Historical Transaction Demand Trend")
daily_path = Path("data/processed/daily_demand_revenue.parquet")
if daily_path.exists():
    df_daily = pd.read_parquet(daily_path)
    if "date" in df_daily.columns:
        df_daily["date"] = pd.to_datetime(df_daily["date"])
        df_daily = df_daily.sort_values("date")
    st.line_chart(df_daily.set_index("date")["revenue"] if "revenue" in df_daily.columns else df_daily.head(100))
else:
    trend_img = Path("reports/figures/05_daily_revenue_trend.png")
    if trend_img.exists():
        st.image(str(trend_img), caption="Historical Daily Demand & Revenue Trend", use_container_width=True)
