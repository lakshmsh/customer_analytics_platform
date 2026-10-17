"""CLI script to run end-to-end data processing and ETL."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.make_dataset import run_etl_pipeline

if __name__ == "__main__":
    try:
        summary = run_etl_pipeline()
        print("\n--- ETL Execution Summary ---")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] ETL failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
