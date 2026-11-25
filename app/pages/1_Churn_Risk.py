"""
Churn Risk Prediction & Batch Scoring Dashboard.

Enables interactive scoring of single customer profiles with calibrated Platt Sigmoid
probabilities and provides high-throughput batch scoring from CSV uploads.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import io

from app.components.ui_elements import (
    load_custom_css,
    render_header,
    render_metric_card,
    plot_churn_gauge,
)
from app.components.api_client import AnalyticsClient

st.set_page_config(page_title="Churn Risk Predictor", page_icon="🚨", layout="wide")
load_custom_css()

render_header(
    title="Customer Churn Risk Scoring Engine",
    subtitle="Real-time calibrated churn probability, risk classification, and high-throughput batch scoring",
    icon="🚨"
)

tab_single, tab_batch = st.tabs(["👤 Single Customer Scoring", "📂 Batch CSV Scoring"])

# ---------------------------------------------------------
# Tab 1: Single Customer Scoring
# ---------------------------------------------------------
with tab_single:
    st.markdown("#### Configure Customer Subscription & Billing Profile")

    # Quick Preset Loader
    col_pre, col_btn = st.columns([3, 1])
    preset = col_pre.selectbox(
        "⚡ Quick Load Preset Profile:",
        ["Custom Input", "High-Risk Customer (Fiber + Month-to-Month)", "Loyal Customer (Two-Year + Auto-Pay)", "Mid-Risk Early Customer"],
    )

    defaults = {
        "tenure": 3,
        "monthly_charges": 89.50,
        "contract": "Month-to-month",
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check",
        "tech_support": "No",
        "online_security": "No",
        "paperless_billing": "Yes",
        "senior_citizen": 0
    }

    if preset == "High-Risk Customer (Fiber + Month-to-Month)":
        defaults = {
            "tenure": 2,
            "monthly_charges": 98.40,
            "contract": "Month-to-month",
            "internet_service": "Fiber optic",
            "payment_method": "Electronic check",
            "tech_support": "No",
            "online_security": "No",
            "paperless_billing": "Yes",
            "senior_citizen": 0
        }
    elif preset == "Loyal Customer (Two-Year + Auto-Pay)":
        defaults = {
            "tenure": 65,
            "monthly_charges": 64.20,
            "contract": "Two year",
            "internet_service": "DSL",
            "payment_method": "Credit card (automatic)",
            "tech_support": "Yes",
            "online_security": "Yes",
            "paperless_billing": "No",
            "senior_citizen": 0
        }
    elif preset == "Mid-Risk Early Customer":
        defaults = {
            "tenure": 12,
            "monthly_charges": 55.00,
            "contract": "One year",
            "internet_service": "DSL",
            "payment_method": "Bank transfer (automatic)",
            "tech_support": "No",
            "online_security": "Yes",
            "paperless_billing": "Yes",
            "senior_citizen": 1
        }

    with st.form("churn_single_form"):
        f_col1, f_col2, f_col3 = st.columns(3)

        with f_col1:
            customer_id = st.text_input("Customer ID", value="CUST-2026-LIVE")
            tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=defaults["tenure"])
            monthly_charges = st.number_input(
                "Monthly Charges ($)", min_value=15.0, max_value=250.0, value=float(defaults["monthly_charges"]), step=1.0
            )
            total_charges = st.number_input(
                "Total Lifetime Charges ($)",
                min_value=0.0,
                max_value=12000.0,
                value=float(round(tenure * monthly_charges, 2)),
                help="Defaults to estimated (Tenure * Monthly Charges) if left untouched."
            )

        with f_col2:
            contract = st.selectbox(
                "Contract Term",
                ["Month-to-month", "One year", "Two year"],
                index=["Month-to-month", "One year", "Two year"].index(defaults["contract"])
            )
            internet_service = st.selectbox(
                "Internet Service",
                ["Fiber optic", "DSL", "No"],
                index=["Fiber optic", "DSL", "No"].index(defaults["internet_service"])
            )
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(defaults["payment_method"])
            )
            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"],
                index=0 if defaults["paperless_billing"] == "Yes" else 1
            )

        with f_col3:
            tech_support = st.selectbox(
                "Dedicated Tech Support",
                ["No", "Yes", "No internet service"],
                index=0 if defaults["tech_support"] == "No" else 1
            )
            online_security = st.selectbox(
                "Online Cyber Security",
                ["No", "Yes", "No internet service"],
                index=0 if defaults["online_security"] == "No" else 1
            )
            online_backup = st.selectbox("Cloud Backup", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV Service", ["No", "Yes", "No internet service"])
            senior_citizen = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes (Senior)" if x == 1 else "No",
                index=defaults["senior_citizen"]
            )

        submit_btn = st.form_submit_button("⚡ Predict Churn Probability", type="primary", use_container_width=True)

    if submit_btn:
        payload = {
            "customer_id": customer_id,
            "tenure": int(tenure),
            "monthly_charges": float(monthly_charges),
            "total_charges": float(total_charges),
            "contract": contract,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "senior_citizen": int(senior_citizen),
            "partner": "No",
            "dependents": "No",
            "phone_service": "Yes",
            "multiple_lines": "No",
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": "No",
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": "No",
            "paperless_billing": paperless_billing
        }

        with st.spinner("Scoring customer profile..."):
            result = AnalyticsClient.predict_churn(payload)

        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")

        res_col1, res_col2 = st.columns([1, 1.4])

        with res_col1:
            st.plotly_chart(plot_churn_gauge(result["churn_probability"]), use_container_width=True)

            tier = result["risk_tier"]
            if tier == "High Risk":
                badge_html = '<span class="badge-high">🚨 HIGH RISK (P ≥ 65%)</span>'
            elif tier == "Medium Risk":
                badge_html = '<span class="badge-medium">⚠️ MEDIUM RISK (35% ≤ P < 65%)</span>'
            else:
                badge_html = '<span class="badge-low">✅ LOW RISK (P < 35%)</span>'

            st.markdown(
                f"""
                <div style="text-align: center; margin-top: -15px;">
                    {badge_html}
                    <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 8px;">
                        Model: Calibrated Sigmoid Classifier • Source: {result.get('source', 'Local')}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_col2:
            st.markdown("#### 🔍 Primary Risk Drivers")
            drivers = result.get("top_risk_drivers", [])
            if drivers:
                for d in drivers[:3]:
                    st.markdown(
                        f"""
                        <div class="action-item" style="border-left-color: #ef4444;">
                            <b style="color: #f87171;">{d['readable_name']}</b><br/>
                            <span style="font-size: 0.85rem; color: #cbd5e1;">
                                Increases churn log-odds by <b>+{d['shap_impact']:.3f}</b>
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No strong positive risk drivers detected; customer exhibits high stability.")

            # Prescribed Intervention Preview
            actions = result.get("prescribed_actions", [])
            if actions:
                st.markdown("#### 💡 Prescribed Retention Action")
                first_act = actions[0]
                st.markdown(
                    f"""
                    <div class="action-item" style="border-left-color: #3b82f6;">
                        <b style="color: #60a5fa;">{first_act['action']}</b> (Est. Cost: ${first_act['cost']:.0f})<br/>
                        <span style="font-size: 0.85rem; color: #cbd5e1;">{first_act['detail']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------------
# Tab 2: Batch Scoring
# ---------------------------------------------------------
with tab_batch:
    st.markdown("#### High-Throughput Batch Scoring")
    st.markdown("Upload a customer records CSV or load an instant sample of 100 Telco customers.")

    b_col1, b_col2 = st.columns([2, 1])

    uploaded_file = b_col1.file_uploader("Upload CSV file", type=["csv"])
    load_sample = b_col2.button("📥 Load Sample Batch (100 Records)", use_container_width=True)

    df_batch = None

    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.success(f"Uploaded {len(df_batch)} records successfully.")
        except Exception as e:
            st.error(f"Error parsing CSV: {e}")
    elif load_sample:
        try:
            sample_path = "data/raw/telco_customer_churn.csv"
            df_raw = pd.read_csv(sample_path)
            df_batch = df_raw.head(100).copy()
            df_batch.rename(columns={
                "customerID": "customer_id",
                "MonthlyCharges": "monthly_charges",
                "TotalCharges": "total_charges",
                "Contract": "contract",
                "InternetService": "internet_service",
                "PaymentMethod": "payment_method",
                "SeniorCitizen": "senior_citizen",
                "TechSupport": "tech_support",
                "OnlineSecurity": "online_security",
                "PaperlessBilling": "paperless_billing",
            }, inplace=True)
            st.info(f"Loaded sample batch of {len(df_batch)} customer records.")
        except Exception as e:
            st.warning(f"Could not load local raw sample: {e}")

    if df_batch is not None:
        if st.button("🚀 Score Entire Batch", type="primary"):
            records = df_batch.to_dict(orient="records")
            # Map column names if needed
            cleaned_records = []
            for r in records:
                cid = str(r.get("customer_id", r.get("customerID", "CUST-X")))
                ten = int(r.get("tenure", 1))
                try:
                    tot = float(r.get("total_charges", r.get("TotalCharges", ten * 50.0)))
                except Exception:
                    tot = float(ten * 50.0)

                cleaned_records.append({
                    "customer_id": cid,
                    "tenure": ten,
                    "monthly_charges": float(r.get("monthly_charges", r.get("MonthlyCharges", 50.0))),
                    "total_charges": tot,
                    "contract": str(r.get("contract", r.get("Contract", "Month-to-month"))),
                    "internet_service": str(r.get("internet_service", r.get("InternetService", "Fiber optic"))),
                    "payment_method": str(r.get("payment_method", r.get("PaymentMethod", "Electronic check"))),
                    "senior_citizen": int(r.get("senior_citizen", r.get("SeniorCitizen", 0))),
                    "partner": str(r.get("partner", "No")),
                    "dependents": str(r.get("dependents", "No")),
                    "phone_service": str(r.get("phone_service", "Yes")),
                    "multiple_lines": str(r.get("multiple_lines", "No")),
                    "online_security": str(r.get("online_security", "No")),
                    "online_backup": str(r.get("online_backup", "No")),
                    "device_protection": str(r.get("device_protection", "No")),
                    "tech_support": str(r.get("tech_support", "No")),
                    "streaming_tv": str(r.get("streaming_tv", "No")),
                    "streaming_movies": str(r.get("streaming_movies", "No")),
                    "paperless_billing": str(r.get("paperless_billing", "Yes")),
                })

            with st.spinner("Scoring batch..."):
                batch_res = AnalyticsClient.predict_batch_churn(cleaned_records)

            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            render_metric_card(m1, "Total Customers", str(batch_res["total_customers"]), "Batch Size")
            render_metric_card(m2, "High Risk", str(batch_res["high_risk_count"]), "P(Churn) ≥ 65%")
            render_metric_card(m3, "Medium Risk", str(batch_res["medium_risk_count"]), "35% ≤ P < 65%")
            render_metric_card(m4, "Low Risk", str(batch_res["low_risk_count"]), "P(Churn) < 35%")

            out_rows = []
            for pred in batch_res["predictions"]:
                out_rows.append({
                    "Customer ID": pred["customer_id"],
                    "Churn Probability": f"{pred['churn_probability'] * 100:.1f}%",
                    "Risk Tier": pred["risk_tier"],
                    "Top Risk Driver": pred["top_risk_drivers"][0]["readable_name"] if pred["top_risk_drivers"] else "None",
                    "Recommended Action": pred["prescribed_actions"][0]["action"] if pred["prescribed_actions"] else "Monitor Only"
                })

            df_out = pd.DataFrame(out_rows)
            st.dataframe(df_out, use_container_width=True)

            # Download CSV
            csv_buf = io.StringIO()
            df_out.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Download Scored Batch Results (CSV)",
                data=csv_buf.getvalue(),
                file_name="churn_batch_scores.csv",
                mime="text/csv",
            )
