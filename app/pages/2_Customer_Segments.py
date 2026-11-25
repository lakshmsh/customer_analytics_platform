"""
Customer Segmentation & Behavioral Persona Explorer.

Segments customers into distinct personas using K-Means clustering across
6 RFM and basket dimensions. Displays interactive radar profiles and tailored
marketing strategy playbooks.
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
    plot_rfm_radar,
)
from app.components.api_client import AnalyticsClient

st.set_page_config(page_title="Customer Segmentation", page_icon="👥", layout="wide")
load_custom_css()

render_header(
    title="Customer Behavioral Segmentation Engine",
    subtitle="K-Means clustering across RFM and basket dimensions with tailored marketing playbooks",
    icon="👥"
)

# Persona Playbooks
PERSONA_PLAYBOOKS = {
    "Champions / High-Value": {
        "badge": '<span class="badge-low">🌟 TOP TIER CHAMPION</span>',
        "description": "High spenders with frequent orders and recent purchases. Highest lifetime value contribution.",
        "marketing_strategy": "VIP concierge support, exclusive sneak-peeks, early access to new lines, loyalty tier rewards.",
        "discount_sensitivity": "Low – focus on recognition, prestige, and premium service rather than price discounts."
    },
    "Loyal Customers": {
        "badge": '<span class="badge-low">💎 LOYAL REGULAR</span>',
        "description": "Consistent purchase cadence and dependable order history with steady basket sizes.",
        "marketing_strategy": "Upsell higher-value bundles, referral bonus incentives, cross-sell complementary categories.",
        "discount_sensitivity": "Medium – value-add gifts with purchase perform better than straight markdowns."
    },
    "At-Risk / Dormant": {
        "badge": '<span class="badge-high">⚠️ AT-RISK CHURNER</span>',
        "description": "Previously valuable customers whose recency has lapsed significantly. High danger of defection.",
        "marketing_strategy": "Aggressive win-back campaigns, 'We miss you' personalized email sequences, dynamic return discounts.",
        "discount_sensitivity": "High – requires clear financial or promotional incentive to reignite engagement."
    },
    "Budget / Occasional Shoppers": {
        "badge": '<span class="badge-medium">🛒 OCCASIONAL SHOPPER</span>',
        "description": "Infrequent purchases with low-to-medium order values. Sensitive to shipping fees and promotions.",
        "marketing_strategy": "Threshold-based free shipping promotions ('Add $10 for free delivery'), flash sales, clearance alerts.",
        "discount_sensitivity": "Very High – heavily driven by discounts, promotional coupons, and seasonal sales."
    },
    "Hibernating / Dormant": {
        "badge": '<span class="badge-high">💤 HIBERNATING</span>',
        "description": "Long recency with low order frequency. Minimal recent touchpoints.",
        "marketing_strategy": "Re-engagement survey to identify reasons for inactivity, low-cost automated email drip, clearance offers.",
        "discount_sensitivity": "High – test steep reactivation discounts on previous favorites."
    }
}
# Persona aliases for model mapping
PERSONA_PLAYBOOKS["Champions"] = PERSONA_PLAYBOOKS["Champions / High-Value"]
PERSONA_PLAYBOOKS["Potential Loyalists"] = PERSONA_PLAYBOOKS["Loyal Customers"]

tab_profile, tab_dataset = st.tabs(["🎯 Persona Simulator & Radar", "📊 Full Segment Distribution"])

with tab_profile:
    st.markdown("#### Input Customer RFM & Purchasing Behavior")

    # Presets
    col_pre, _ = st.columns([2, 1])
    preset = col_pre.selectbox(
        "⚡ Choose a Persona Archetype to Simulate:",
        [
            "Custom Input",
            "High-Value Champion (Recent, High Spend, Frequent)",
            "At-Risk Lapsed Buyer (High Spend in Past, Inactive Now)",
            "Budget Occasional Buyer (Small Baskets, High Recency)",
            "Consistent Loyalist (Moderate Spend, Reliable Cadence)"
        ]
    )

    defaults = {
        "recency": 14,
        "frequency": 8,
        "monetary": 2450.0,
        "tenure": 200,
        "basket_val": 306.25,
        "basket_sz": 18.0
    }

    if preset == "High-Value Champion (Recent, High Spend, Frequent)":
        defaults = {"recency": 5, "frequency": 24, "monetary": 6800.0, "tenure": 320, "basket_val": 283.33, "basket_sz": 25.0}
    elif preset == "At-Risk Lapsed Buyer (High Spend in Past, Inactive Now)":
        defaults = {"recency": 180, "frequency": 12, "monetary": 3900.0, "tenure": 250, "basket_val": 325.00, "basket_sz": 20.0}
    elif preset == "Budget Occasional Buyer (Small Baskets, High Recency)":
        defaults = {"recency": 90, "frequency": 2, "monetary": 140.0, "tenure": 90, "basket_val": 70.00, "basket_sz": 4.0}
    elif preset == "Consistent Loyalist (Moderate Spend, Reliable Cadence)":
        defaults = {"recency": 20, "frequency": 10, "monetary": 1800.0, "tenure": 180, "basket_val": 180.00, "basket_sz": 12.0}

    with st.form("segment_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            customer_id = st.text_input("Customer ID", value="CUST-13085")
            recency = st.slider("Recency (Days since last order)", min_value=1, max_value=365, value=int(defaults["recency"]))
            frequency = st.number_input("Frequency (Total Orders)", min_value=1, max_value=100, value=int(defaults["frequency"]))

        with c2:
            monetary = st.number_input("Monetary (Total Spend £)", min_value=10.0, max_value=25000.0, value=float(defaults["monetary"]), step=50.0)
            tenure_days = st.slider("Customer Tenure (Days)", min_value=1, max_value=400, value=int(defaults["tenure"]))

        with c3:
            basket_val = st.number_input(
                "Average Basket Value (£)",
                min_value=5.0,
                max_value=2000.0,
                value=float(defaults["basket_val"]),
                step=10.0
            )
            basket_sz = st.slider("Average Basket Size (Items/Order)", min_value=1.0, max_value=60.0, value=float(defaults["basket_sz"]))

        submit_seg = st.form_submit_button("👥 Assign Behavioral Persona", type="primary", use_container_width=True)

    if submit_seg:
        payload = {
            "customer_id": customer_id,
            "recency_days": int(recency),
            "frequency_orders": int(frequency),
            "monetary_total": float(monetary),
            "tenure_days": int(tenure_days),
            "avg_basket_value": float(basket_val),
            "avg_basket_size": float(basket_sz)
        }

        with st.spinner("Classifying customer cluster..."):
            seg_res = AnalyticsClient.get_segment(payload)

        persona = seg_res.get("persona_name", "Standard Shopper")
        cluster_id = seg_res.get("cluster_id", 0)

        st.markdown("---")
        st.markdown(f"### Assigned Segment: **Cluster #{cluster_id} — {persona}**")

        p_info = PERSONA_PLAYBOOKS.get(persona, PERSONA_PLAYBOOKS.get("Champions / High-Value"))

        s_col1, s_col2 = st.columns([1.1, 1])

        with s_col1:
            st.plotly_chart(
                plot_rfm_radar(payload, cluster_means={"benchmark": True}, persona_name=persona),
                use_container_width=True
            )

        with s_col2:
            st.markdown(f"#### 📋 Recommended Playbook for {persona}")
            st.markdown(f"{p_info['badge']}", unsafe_allow_html=True)
            st.markdown(f"**Customer Profile:** {p_info['description']}")
            st.markdown(
                f"""
                <div class="action-item" style="border-left-color: #10b981;">
                    <b style="color: #34d399;">Marketing Strategy:</b><br/>
                    <span style="font-size: 0.9rem; color: #cbd5e1;">{p_info['marketing_strategy']}</span>
                </div>
                <div class="action-item" style="border-left-color: #f59e0b;">
                    <b style="color: #fbbf24;">Promotion & Discount Sensitivity:</b><br/>
                    <span style="font-size: 0.9rem; color: #cbd5e1;">{p_info['discount_sensitivity']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

with tab_dataset:
    st.markdown("#### Processed Customer Base Segment Distribution")
    seg_data_path = Path("data/processed/customers_features.parquet")
    if seg_data_path.exists():
        df_seg = pd.read_parquet(seg_data_path)
        st.write(f"Displaying sample of {min(len(df_seg), 200)} customers from the processed feature store:")
        show_cols = [c for c in ["Customer ID", "customer_id", "recency_days", "frequency_orders", "monetary_total", "tenure_days", "avg_basket_value", "avg_basket_size"] if c in df_seg.columns]
        st.dataframe(df_seg[show_cols].head(200), use_container_width=True)
    else:
        st.info("Feature store file `customers_features.parquet` not found on disk.")
