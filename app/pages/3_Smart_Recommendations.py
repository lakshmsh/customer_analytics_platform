"""
Smart Product Recommendation System (Bonus 1).

Delivers personalized product suggestions by synthesizing user-item collaborative
filtering with item-content TF-IDF similarity. Dynamically attaches targeted
churn-mitigation discounts to retain high-risk customers.
"""

import streamlit as st
import pandas as pd

from app.components.ui_elements import (
    load_custom_css,
    render_header,
    render_metric_card,
)
from app.components.api_client import AnalyticsClient

st.set_page_config(page_title="Smart Recommendations", page_icon="🎁", layout="wide")
load_custom_css()

render_header(
    title="Hybrid Personalized Recommendation Engine",
    subtitle="Collaborative filtering and content-based recommendations with dynamic churn retention incentives",
    icon="🎁"
)

# Controls
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1, 1.2])

with ctrl_col1:
    customer_id = st.selectbox(
        "Select Target Customer ID:",
        ["13085", "12347", "17850", "15311", "14646", "Custom New Customer..."]
    )
    if customer_id == "Custom New Customer...":
        customer_id = st.text_input("Enter Customer ID", value="99999")

with ctrl_col2:
    n_items = st.slider("Number of Recommendations", min_value=1, max_value=10, value=4)

with ctrl_col3:
    st.markdown("<br/>", unsafe_allow_html=True)
    is_churn_risk = st.checkbox(
        "🚨 Customer is At-Risk of Churn",
        value=True,
        help="When checked, attaches dynamic retention incentives and price markdowns to prevent defection."
    )

st.markdown("---")

# Query Recommender
with st.spinner("Generating personalized recommendations..."):
    rec_result = AnalyticsClient.get_recommendations(
        customer_id=str(customer_id),
        n_items=int(n_items),
        is_churn_risk=is_churn_risk
    )

items = rec_result.get("recommendations", [])

# Summary Ribbon
m1, m2, m3 = st.columns(3)
render_metric_card(m1, "Target Customer", str(customer_id), "Account Identifier")
render_metric_card(m2, "Catalogue Scope", f"{len(items)} Items", "Top-K Highest Relevance")
render_metric_card(
    m3,
    "Retention Offer Status",
    "Active (15% OFF)" if is_churn_risk else "Standard Catalog",
    "Dynamic Margin Discount" if is_churn_risk else "Standard List Price"
)

st.markdown("### 🛍️ Recommended Products for Customer")

if not items:
    st.warning("No recommendations returned. Please verify customer ID.")
else:
    grid_cols = st.columns(2)
    for idx, item in enumerate(items):
        target_col = grid_cols[idx % 2]
        with target_col:
            stock = item.get("stock_code", "PROD")
            desc = item.get("description", "Product Title")
            base_p = float(item.get("base_price", 10.0))
            score = float(item.get("relevance_score", 0.85))
            strat = item.get("strategy", "Hybrid").title()
            incentive = item.get("retention_incentive")
            disc_p = item.get("discounted_price")

            price_html = f"<span style='font-size: 1.15rem; font-weight: 700; color: #f1f5f9;'>£{base_p:.2f}</span>"
            if disc_p is not None:
                price_html = f"""
                <span style='text-decoration: line-through; color: #94a3b8; font-size: 0.95rem;'>£{base_p:.2f}</span>
                <span style='font-size: 1.25rem; font-weight: 700; color: #10b981; margin-left: 8px;'>£{disc_p:.2f}</span>
                """

            incentive_html = ""
            if incentive:
                incentive_html = f"""
                <div style='margin-top: 8px;'>
                    <span class='discount-pill'>🎁 {incentive}</span>
                </div>
                """

            st.markdown(
                f"""
                <div class="product-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <span style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">
                                SKU: {stock} • Strategy: {strat}
                            </span>
                            <div style="font-size: 1.05rem; font-weight: 600; color: #f8fafc; margin-top: 3px;">
                                {desc}
                            </div>
                        </div>
                        <div style="background: rgba(59, 130, 246, 0.15); border: 1px solid #3b82f6; color: #60a5fa; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold;">
                            {score * 100:.0f}% Match
                        </div>
                    </div>
                    <div style="margin-top: 12px; display: flex; justify-content: space-between; align-items: center;">
                        <div>{price_html}</div>
                        <button style="background: #3b82f6; border: none; color: white; padding: 6px 14px; border-radius: 6px; font-size: 0.85rem; font-weight: 600; cursor: pointer;">
                            Add to Retention Bundle
                        </button>
                    </div>
                    {incentive_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("---")
st.info(
    "💡 **Recommendation Logic**: When a customer exhibits elevated churn risk ($P \\ge 0.35$), "
    "the recommendation engine filters for high-affinity products from categories they have historically "
    "favorited and applies an algorithmic retention discount (15% markdown). This significantly increases "
    "re-engagement conversion while protecting positive transaction margins."
)
