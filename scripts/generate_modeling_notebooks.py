"""Generate notebooks 03, 04, 05, 06 for Phase 5."""

import json
from pathlib import Path
import nbformat

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def save_notebook(filename: str, cells: list):
    nb_data = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.13"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    path = NOTEBOOKS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=2)

    nb = nbformat.read(path, as_version=4)
    nbformat.validator.normalize(nb)
    nbformat.write(nb, path)
    print(f"Created and normalized: {path}")


# ----------------------------------------------------
# 03_churn_modeling.ipynb
# ----------------------------------------------------
cells_03 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🎯 Churn Prediction Modeling Engine\n",
            "\n",
            "## Model Suite:\n",
            "- **Logistic Regression** (Interpretable Linear Baseline with Balanced Weights)\n",
            "- **XGBoost** (Gradient Boosted Trees with Scale-Pos-Weight)\n",
            "- **LightGBM** (Histogram-based Boosting with Optuna Tuning)\n",
            "- **CatBoost** (Ordered Boosting with Auto-Balanced Class Weights)\n",
            "\n",
            "## Key Techniques:\n",
            "- Stratified Train / Validation / Test split (70 / 15 / 15).\n",
            "- Evaluation metrics: ROC-AUC, PR-AUC, Brier Score, Precision, Recall.\n",
            "- Platt Scaling (Sigmoid Probability Calibration).\n",
            "- MLflow Experiment Tracking."
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
            "import json\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
            "os.chdir(project_root)\n",
            "sys.path.insert(0, str(project_root))\n",
            "\n",
            "from src.models.train_churn import train_and_evaluate_churn_models\n",
            "from src.models.predict import predict_churn\n",
            "print('Environment ready.')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 2,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Inspect Benchmark Results\n",
            "with open('models/churn_metadata.json', 'r') as f:\n",
            "    meta = json.load(f)\n",
            "\n",
            "print(f'Selected Best Model: {meta[\"best_model_name\"]}')\n",
            "benchmarks = meta['all_benchmarks']\n",
            "\n",
            "summary_rows = []\n",
            "for model_name, evals in benchmarks.items():\n",
            "    summary_rows.append({\n",
            "        'Model': model_name,\n",
            "        'Val ROC-AUC': evals['validation']['roc_auc'],\n",
            "        'Val PR-AUC': evals['validation']['pr_auc'],\n",
            "        'Test ROC-AUC': evals['test']['roc_auc'],\n",
            "        'Test PR-AUC': evals['test']['pr_auc'],\n",
            "        'Test Brier Score': evals['test']['brier_score'],\n",
            "        'Test F1': evals['test']['f1']\n",
            "    })\n",
            "bench_df = pd.DataFrame(summary_rows).sort_values('Test PR-AUC', ascending=False)\n",
            "display(bench_df)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 3,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Test Real-Time Single-Customer Prediction\n",
            "sample_input = {\n",
            "    'tenure': 2,\n",
            "    'monthly_charges': 89.50,\n",
            "    'total_charges': 179.0,\n",
            "    'contract_Month-to-month': 1,\n",
            "    'internet_service_Fiber optic': 1,\n",
            "    'payment_method_Electronic check': 1\n",
            "}\n",
            "res = predict_churn(sample_input)\n",
            "print('Prediction Output for High-Risk Test Account:')\n",
            "print(json.dumps(res, indent=2))"
        ]
    }
]
save_notebook("03_churn_modeling.ipynb", cells_03)


# ----------------------------------------------------
# 04_segmentation.ipynb
# ----------------------------------------------------
cells_04 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 👥 Customer Segmentation & Persona Profiling\n",
            "\n",
            "## Objectives:\n",
            "- Apply unsupervised clustering (K-Means and Gaussian Mixture Models) on scaled RFM and basket dynamics.\n",
            "- Identify optimal cluster count ($k=4$) via Silhouette analysis and Davies-Bouldin index.\n",
            "- Map empirical clusters to business personas (*Champions*, *Potential Loyalists*, *At-Risk High Rollers*, *Hibernating*)."
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
            "import json\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
            "os.chdir(project_root)\n",
            "sys.path.insert(0, str(project_root))\n",
            "\n",
            "with open('models/segmentation_metadata.json', 'r') as f:\n",
            "    meta = json.load(f)\n",
            "\n",
            "print('Clustering Quality:', meta['metrics'])\n",
            "profiles_df = pd.DataFrame(meta['cluster_profiles'])\n",
            "display(profiles_df)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 2,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Visualize Persona Cluster Sizes and Churn Rates\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "sns.barplot(data=profiles_df, x='persona_name', y='size', palette='Blues_r', ax=axes[0])\n",
            "axes[0].set_title('Customer Count by Persona', fontweight='bold')\n",
            "axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=20)\n",
            "\n",
            "sns.barplot(data=profiles_df, x='persona_name', y='churn_rate', palette='Reds_r', ax=axes[1])\n",
            "axes[1].set_title('Churn Rate by Persona (%)', fontweight='bold')\n",
            "axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=20)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }
]
save_notebook("04_segmentation.ipynb", cells_04)


# ----------------------------------------------------
# 05_recommendation.ipynb
# ----------------------------------------------------
cells_05 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🎁 Advanced Hybrid Recommender System (Bonus 1)\n",
            "\n",
            "## Architecture:\n",
            "- **Item-Item Collaborative Filtering**: Computes cosine similarity over user-item log-interaction matrix.\n",
            "- **Content-Based Filtering**: TF-IDF vectorization on product descriptions to recommend thematic relatives.\n",
            "- **Hybrid Score Blending**: $\\text{Score} = \\alpha \\cdot \\text{CF} + (1-\\alpha) \\cdot \\text{Content}$.\n",
            "- **Churn Retention Incentives**: Generates personalized dynamic discounts (e.g. 15% VIP discount) for high-churn accounts."
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
            "import json\n",
            "import pandas as pd\n",
            "\n",
            "project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
            "os.chdir(project_root)\n",
            "sys.path.insert(0, str(project_root))\n",
            "\n",
            "from src.models.predict import get_recommendations\n",
            "\n",
            "# Test standard recommendation\n",
            "standard_recs = get_recommendations('13085', n_items=4, is_churn_risk=False)\n",
            "print('Standard Customer Recommendations:')\n",
            "display(pd.DataFrame(standard_recs))\n",
            "\n",
            "# Test churn-targeted retention recommendation with discount incentive\n",
            "retention_recs = get_recommendations('13085', n_items=4, is_churn_risk=True)\n",
            "print('\\nChurn Retention Tailored Offers:')\n",
            "display(pd.DataFrame(retention_recs))"
        ]
    }
]
save_notebook("05_recommendation.ipynb", cells_05)


# ----------------------------------------------------
# 06_forecasting.ipynb
# ----------------------------------------------------
cells_06 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 📈 Demand & Revenue Impact Forecasting (Bonus 2)\n",
            "\n",
            "## Methodology:\n",
            "- Multi-horizon time-series regression with LightGBM.\n",
            "- Features: Multi-scale lags (1, 2, 7, 14, 21, 28 days), rolling averages (7, 14, 30 days), and calendar cycles.\n",
            "- Financial Scenario Modeling: Quantifies baseline revenue vs unmitigated churn revenue at risk vs retention recovery."
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
            "import json\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
            "os.chdir(project_root)\n",
            "sys.path.insert(0, str(project_root))\n",
            "\n",
            "with open('models/forecast_metadata.json', 'r') as f:\n",
            "    forecast_meta = json.load(f)\n",
            "\n",
            "print('Forecasting Quality Metrics:')\n",
            "print('  Daily Revenue MAPE:', f\"{forecast_meta['revenue_metrics']['mape']*100:.2f}%\")\n",
            "print('  Daily Orders MAPE:', f\"{forecast_meta['orders_metrics']['mape']*100:.2f}%\")\n",
            "\n",
            "print('\\n30-Day Financial Impact Simulation:')\n",
            "impact = forecast_meta['impact_scenario']\n",
            "for k, v in impact.items():\n",
            "    print(f'  {k}: {v}')"
        ]
    }
]
save_notebook("06_forecasting.ipynb", cells_06)
print("All modeling notebooks successfully generated!")
