"""Download raw datasets for Customer Analytics Platform.

Datasets:
1. Telco Customer Churn (IBM / Kaggle) -> data/raw/telco_customer_churn.csv
2. Online Retail II (UCI Machine Learning Repository) -> data/raw/online_retail_II.xlsx
"""

import os
import sys
import zipfile
import logging
from pathlib import Path
import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("download_data")

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

URLS = {
    "telco": [
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    ],
    "online_retail_ii_zip": [
        "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
    ]
}


def download_file(urls: list[str], destination_path: Path, description: str) -> bool:
    """Download file from list of URLs with fallback support and progress bar."""
    if destination_path.exists() and destination_path.stat().st_size > 0:
        logger.info(f"[EXISTS] {description} already exists at {destination_path} ({destination_path.stat().st_size / 1024 / 1024:.2f} MB). Skipping download.")
        return True

    for url in urls:
        try:
            logger.info(f"Attempting download from: {url}")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024 * 64  # 64 KB

            with open(destination_path, "wb") as f, tqdm(
                desc=description,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        size = f.write(chunk)
                        bar.update(size)

            logger.info(f"[SUCCESS] Downloaded {description} -> {destination_path} ({destination_path.stat().st_size / 1024 / 1024:.2f} MB)")
            return True
        except Exception as e:
            logger.warning(f"Failed to download from {url}: {e}")
            if destination_path.exists():
                destination_path.unlink()

    logger.error(f"[FAILED] Could not download {description} from any provided URLs.")
    return False


def main():
    logger.info("Starting raw dataset gathering...")

    # 1. Download Telco Churn
    telco_path = RAW_DATA_DIR / "telco_customer_churn.csv"
    success_telco = download_file(URLS["telco"], telco_path, "Telco Customer Churn")

    # 2. Download Online Retail II
    retail_xlsx_path = RAW_DATA_DIR / "online_retail_II.xlsx"
    retail_zip_path = RAW_DATA_DIR / "online_retail_ii.zip"

    if not retail_xlsx_path.exists():
        success_zip = download_file(URLS["online_retail_ii_zip"], retail_zip_path, "Online Retail II (ZIP)")
        if success_zip and retail_zip_path.exists():
            try:
                logger.info(f"Extracting {retail_zip_path}...")
                with zipfile.ZipFile(retail_zip_path, "r") as z:
                    for member in z.namelist():
                        if member.lower().endswith((".xlsx", ".csv")):
                            z.extract(member, RAW_DATA_DIR)
                            extracted = RAW_DATA_DIR / member
                            if extracted.suffix.lower() == ".xlsx" and extracted != retail_xlsx_path:
                                extracted.rename(retail_xlsx_path)
                            logger.info(f"Extracted: {member}")
                # Remove the temporary zip to keep raw folder clean
                if retail_zip_path.exists():
                    retail_zip_path.unlink()
                logger.info(f"[SUCCESS] Online Retail II dataset ready at {retail_xlsx_path}")
            except Exception as e:
                logger.error(f"Error extracting Online Retail II zip: {e}")
    else:
        logger.info(f"[EXISTS] {retail_xlsx_path} already present.")

    # Verification
    files_ok = True
    for p in [telco_path, retail_xlsx_path]:
        if p.exists() and p.stat().st_size > 1000:
            logger.info(f"Verified: {p.name} ({p.stat().st_size / 1024 / 1024:.2f} MB)")
        else:
            logger.error(f"Missing or corrupted file: {p}")
            files_ok = False

    if files_ok:
        logger.info("All raw datasets collected and verified successfully.")
        sys.exit(0)
    else:
        logger.error("Dataset gathering completed with errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
