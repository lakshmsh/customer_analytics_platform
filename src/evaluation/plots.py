"""Visualization utilities for Exploratory Data Analysis and Model Evaluation."""

from pathlib import Path
from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set cohesive aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.autolayout"] = True

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_churn_distribution(
    telco_df: pd.DataFrame,
    customer_features: pd.DataFrame,
    save_path: Optional[str | Path] = FIGURES_DIR / "01_churn_distribution.png"
) -> plt.Figure:
    """Plot target class distributions for both subscription and transactional churn."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Telco Churn
    telco_counts = telco_df["churn_label"].value_counts(normalize=True) * 100
    sns.barplot(
        x=["Retained (0)", "Churned (1)"],
        y=telco_counts.values,
        palette=["#2b5c8f", "#d9534f"],
        ax=axes[0]
    )
    axes[0].set_title(
        f"Telco Contractual Churn (Imbalance: {telco_counts[1]:.1f}% Churn vs {telco_counts[0]:.1f}% Retained)",
        fontsize=12, fontweight="bold"
    )
    axes[0].set_ylabel("Percentage of Customers (%)")
    for i, v in enumerate(telco_counts.values):
        axes[0].text(i, v + 1, f"{v:.1f}% ({telco_df['churn_label'].value_counts()[i]:,})", ha="center", fontweight="bold")
    axes[0].set_ylim(0, 100)

    # Retail Churn
    retail_counts = customer_features["is_churned"].value_counts(normalize=True) * 100
    sns.barplot(
        x=["Active (0)", "Churned / Inactive (1)"],
        y=retail_counts.values,
        palette=["#2b5c8f", "#e67e22"],
        ax=axes[1]
    )
    axes[1].set_title(
        f"Retail E-Commerce Churn (>=90d Inactive: {retail_counts[1]:.1f}% vs {retail_counts[0]:.1f}% Active)",
        fontsize=12, fontweight="bold"
    )
    axes[1].set_ylabel("Percentage of Customers (%)")
    for i, v in enumerate(retail_counts.values):
        axes[1].text(i, v + 1, f"{v:.1f}% ({customer_features['is_churned'].value_counts()[i]:,})", ha="center", fontweight="bold")
    axes[1].set_ylim(0, 100)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_rfm_distributions(
    customer_features: pd.DataFrame,
    save_path: Optional[str | Path] = FIGURES_DIR / "02_rfm_distributions.png"
) -> plt.Figure:
    """Plot log-scaled distributions of Recency, Frequency, and Monetary metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Recency
    sns.histplot(customer_features["recency_days"], kde=True, color="#3498db", bins=30, ax=axes[0])
    axes[0].axvline(90, color="red", linestyle="--", linewidth=2, label="Churn Cutoff (90d)")
    axes[0].set_title("Recency Distribution (Days since Last Order)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Recency (Days)")
    axes[0].legend()

    # Frequency (Log)
    sns.histplot(customer_features["frequency_orders"], kde=True, color="#2ecc71", bins=30, log_scale=True, ax=axes[1])
    axes[1].set_title("Frequency Distribution (Orders placed, Log Scale)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Frequency (Orders)")

    # Monetary (Log)
    sns.histplot(customer_features["monetary_total"].clip(lower=1), kde=True, color="#9b59b6", bins=30, log_scale=True, ax=axes[2])
    axes[2].set_title("Monetary Spend (£, Log Scale)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Total Spend (£)")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_churn_by_contract_and_tenure(
    telco_df: pd.DataFrame,
    save_path: Optional[str | Path] = FIGURES_DIR / "03_churn_by_tenure_contract.png"
) -> plt.Figure:
    """Plot bivariate relationships of tenure, contract type, and churn rate."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Contract Churn Rate
    contract_churn = telco_df.groupby("contract")["churn_label"].agg(["mean", "count"]).reset_index()
    contract_churn["mean"] *= 100
    sns.barplot(data=contract_churn, x="contract", y="mean", palette="Reds_r", ax=axes[0])
    axes[0].set_title("Churn Rate by Contract Commitment Type", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Churn Rate (%)")
    axes[0].set_xlabel("Contract Type")
    for i, r in contract_churn.iterrows():
        axes[0].text(i, r["mean"] + 1, f"{r['mean']:.1f}% (n={r['count']:,})", ha="center", fontweight="bold")
    axes[0].set_ylim(0, 60)

    # Tenure Distribution by Churn
    sns.kdeplot(data=telco_df, x="tenure", hue="churn", common_norm=False, fill=True, palette={"No": "#2b5c8f", "Yes": "#d9534f"}, ax=axes[1])
    axes[1].set_title("Tenure Kernel Density by Churn Status", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Customer Tenure (Months)")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_customer_pareto(
    customer_features: pd.DataFrame,
    save_path: Optional[str | Path] = FIGURES_DIR / "04_customer_value_tiers.png"
) -> plt.Figure:
    """Plot cumulative revenue share (Pareto 80/20 principle) and value tier distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Pareto Cumulative Curve
    sorted_df = customer_features.sort_values("monetary_total", ascending=False).reset_index(drop=True)
    sorted_df["cum_spend"] = sorted_df["monetary_total"].cumsum()
    sorted_df["cum_spend_pct"] = (sorted_df["cum_spend"] / sorted_df["monetary_total"].sum()) * 100
    sorted_df["cum_cust_pct"] = ((sorted_df.index + 1) / len(sorted_df)) * 100

    axes[0].plot(sorted_df["cum_cust_pct"], sorted_df["cum_spend_pct"], color="#e74c3c", linewidth=2.5, label="Cumulative Revenue")
    axes[0].plot([0, 100], [0, 100], color="grey", linestyle="--", label="Equality Benchmark")
    top20_spend = sorted_df[sorted_df["cum_cust_pct"] <= 20]["cum_spend_pct"].max()
    axes[0].axvline(20, color="black", linestyle=":", label=f"Top 20% Customers = {top20_spend:.1f}% Revenue")
    axes[0].axhline(top20_spend, color="black", linestyle=":")
    axes[0].set_title(f"Pareto Principle: Top 20% generate {top20_spend:.1f}% of Total Revenue", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("% of Total Customers (Ranked by Spend)")
    axes[0].set_ylabel("% of Total Cumulative Revenue")
    axes[0].legend()

    # Value Tier Spend Share
    tier_agg = customer_features.groupby("value_tier")["monetary_total"].sum().reset_index()
    tier_order = ["VIP", "High Value", "Medium Value", "Low Value"]
    tier_agg["value_tier"] = pd.Categorical(tier_agg["value_tier"], categories=tier_order, ordered=True)
    tier_agg = tier_agg.sort_values("value_tier")
    tier_agg["spend_pct"] = (tier_agg["monetary_total"] / tier_agg["monetary_total"].sum()) * 100

    sns.barplot(data=tier_agg, x="value_tier", y="spend_pct", palette="Blues_r", ax=axes[1])
    axes[1].set_title("Total Revenue Contribution by Customer Tier", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("% of Revenue")
    axes[1].set_xlabel("Customer Value Tier")
    for i, r in enumerate(tier_agg.itertuples()):
        axes[1].text(i, r.spend_pct + 1, f"{r.spend_pct:.1f}% (£{r.monetary_total/1e6:.2f}M)", ha="center", fontweight="bold")
    axes[1].set_ylim(0, 80)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_daily_revenue_trend(
    daily_ts: pd.DataFrame,
    save_path: Optional[str | Path] = FIGURES_DIR / "05_daily_revenue_trend.png"
) -> plt.Figure:
    """Plot time-series revenue trends with 7-day and 30-day moving averages."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    df = daily_ts.copy()
    df["rolling_7"] = df["daily_revenue"].rolling(7, min_periods=1).mean()
    df["rolling_30"] = df["daily_revenue"].rolling(30, min_periods=1).mean()

    # Revenue
    axes[0].plot(df["date"], df["daily_revenue"], color="#bdc3c7", alpha=0.6, label="Daily Net Revenue")
    axes[0].plot(df["date"], df["rolling_7"], color="#3498db", linewidth=1.8, label="7-Day Moving Avg")
    axes[0].plot(df["date"], df["rolling_30"], color="#2c3e50", linewidth=2.5, label="30-Day Trend")
    axes[0].set_title("Daily Transaction Revenue Trend (2009 - 2011)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Revenue (£)")
    axes[0].legend(loc="upper left")

    # Order Volume
    axes[1].plot(df["date"], df["daily_orders"], color="#16a085", linewidth=1.5, label="Daily Orders Placed")
    axes[1].set_title("Daily Order Volume", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Order Count")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="upper left")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: List[str],
    title: str = "Feature Correlation Matrix",
    save_path: Optional[str | Path] = FIGURES_DIR / "06_correlation_matrix.png"
) -> plt.Figure:
    """Plot correlation matrix heatmap for selected numeric features."""
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title(title, fontsize=13, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
