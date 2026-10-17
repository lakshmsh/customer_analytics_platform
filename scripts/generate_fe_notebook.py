"""Generate notebooks/02_feature_engineering.ipynb demonstrating modular feature generation and leakage prevention."""

import json
from pathlib import Path

notebook_path = Path("notebooks/02_feature_engineering.ipynb")
notebook_path.parent.mkdir(parents=True, exist_ok=True)

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# ⚙️ Feature Engineering & Leakage Prevention Pipeline\n",
            "\n",
            "## Objectives\n",
            "- Extract rich behavioral signals from raw transactions (RFM, tenure, basket dynamics, spend velocity, purchase periodicity).\n",
            "- Guarantee **zero data leakage**: Compute reference dates and statistical quantiles strictly on the training partition.\n",
            "- Transform subscription features with composite domain indicators (`security_bundle_score`, `charges_tenure_ratio`).\n",
            "- Validate transformer consistency across train and test sets using `CustomerFeatureTransformer`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, sys\n",
            "from pathlib import Path\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
            "os.chdir(project_root)\n",
            "sys.path.insert(0, str(project_root))\n",
            "\n",
            "from src.features.build_features import CustomerFeatureTransformer\n",
            "from src.features.encoding import add_telco_engineered_features, encode_telco_features\n",
            "print('Imports initialized successfully!')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 2,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Load Cleaned Transactions\n",
            "tx_df = pd.read_parquet('data/processed/retail_cleaned_transactions.parquet')\n",
            "print(f'Total cleaned transactions: {len(tx_df):,}')\n",
            "\n",
            "# 2. Demonstrate Leakage Prevention via Temporal Split\n",
            "# Split transactions into 80% train and 20% holdout test based on date\n",
            "tx_df['invoice_date'] = pd.to_datetime(tx_df['invoice_date'])\n",
            "split_date = tx_df['invoice_date'].quantile(0.80)\n",
            "print(f'Temporal train/test cutoff date: {split_date.strftime(\"%Y-%m-%d\")}')\n",
            "\n",
            "train_tx = tx_df[tx_df['invoice_date'] <= split_date].copy()\n",
            "test_tx = tx_df[tx_df['invoice_date'] > split_date].copy()\n",
            "print(f'Train transactions: {len(train_tx):,} | Test transactions: {len(test_tx):,}')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 3,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Fit Transformer on Train Only, Transform Both\n",
            "transformer = CustomerFeatureTransformer(inactivity_days_threshold=90)\n",
            "transformer.fit(train_tx)\n",
            "\n",
            "train_features = transformer.transform(train_tx)\n",
            "test_features = transformer.transform(test_tx)\n",
            "\n",
            "print(f'Train customer features shape: {train_features.shape}')\n",
            "print(f'Test customer features shape: {test_features.shape}')\n",
            "display(train_features.head())"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 4,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 4. Telco Composite & Ratio Engineering\n",
            "raw_telco = pd.read_parquet('data/interim/telco_cleaned.parquet')\n",
            "telco_enhanced = add_telco_engineered_features(raw_telco)\n",
            "print('Enhanced Telco Features:')\n",
            "display(telco_enhanced[['customer_id', 'security_bundle_score', 'streaming_bundle_score', 'charges_tenure_ratio', 'has_family']].head())"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Summary of Leakage Prevention & Feature Guarantees\n",
            "- **Statistical Isolation**: Reference dates, quantiles, and category distributions are learned strictly from the historical training split.\n",
            "- **Scikit-Learn Standard API**: `CustomerFeatureTransformer` implements `fit()` and `transform()`, allowing direct integration into production scoring pipelines and FastAPI endpoints.\n",
            "- **Rich Signal**: Produces 18+ high-signal behavioral and temporal variables capturing RFM, purchase intervals, tenure, basket size, and weekend/holiday affinities."
        ]
    }
]

notebook_data = {
    "cells": cells,
    "metadata": {
        "language_info": {"name": "python", "version": "3.13"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook_data, f, indent=2)

print(f"Created {notebook_path}")
