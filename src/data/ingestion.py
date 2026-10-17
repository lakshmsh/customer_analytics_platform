"""Data ingestion module for Customer Analytics Platform.

Loads raw files from data/raw/ directory.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
from src.utils.logger import get_logger
from src.utils.helpers import timer

logger = get_logger("data.ingestion")

RAW_DATA_DIR = Path("data/raw")


@timer
def load_raw_telco(filepath: Optional[str | Path] = None) -> pd.DataFrame:
    """Load raw Telco Customer Churn dataset from CSV.

    Args:
        filepath: Custom path to telco csv. If None, uses data/raw/telco_customer_churn.csv.

    Returns:
        pd.DataFrame: Raw Telco dataframe.
    """
    path = Path(filepath) if filepath else RAW_DATA_DIR / "telco_customer_churn.csv"
    if not path.exists():
        raise FileNotFoundError(f"Telco dataset not found at {path.resolve()}. Please run download_data.py first.")

    logger.info(f"Loading raw Telco dataset from {path}...")
    df = pd.read_csv(path)
    logger.info(f"Loaded Telco dataset: {df.shape[0]:,} rows, {df.shape[1]} columns.")
    return df


@timer
def load_raw_online_retail(filepath: Optional[str | Path] = None, sheet_name: Optional[str | int] = None) -> pd.DataFrame:
    """Load raw Online Retail II dataset from Excel file.
    Combines both sheets (Year 2009-2010 and Year 2010-2011) if sheet_name is None.

    Args:
        filepath: Custom path to online retail Excel file.
        sheet_name: Specific sheet name or index to load. If None, concatenates all sheets.

    Returns:
        pd.DataFrame: Combined raw Online Retail II dataframe.
    """
    path = Path(filepath) if filepath else RAW_DATA_DIR / "online_retail_II.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"Online Retail II dataset not found at {path.resolve()}. Please run download_data.py first.")

    logger.info(f"Inspecting sheets in Online Retail II from {path}...")
    xl = pd.ExcelFile(path)
    logger.info(f"Found sheets: {xl.sheet_names}")

    if sheet_name is not None:
        logger.info(f"Loading single sheet '{sheet_name}'...")
        df = pd.read_excel(path, sheet_name=sheet_name)
        logger.info(f"Loaded sheet '{sheet_name}': {df.shape[0]:,} rows.")
        return df

    # Load and combine all sheets
    dfs = []
    for sheet in xl.sheet_names:
        logger.info(f"Loading sheet '{sheet}'...")
        df_sheet = pd.read_excel(path, sheet_name=sheet)
        logger.info(f"Loaded sheet '{sheet}': {df_sheet.shape[0]:,} rows.")
        dfs.append(df_sheet)

    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined Online Retail II dataset: {combined_df.shape[0]:,} rows, {combined_df.shape[1]} columns.")
    return combined_df
