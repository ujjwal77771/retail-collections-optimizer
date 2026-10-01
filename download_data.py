"""
=============================================================================
Collections Prioritization Project
Stage 0: Data Download Helper
=============================================================================
INSTRUCTIONS:
  1. Go to https://www.kaggle.com/c/home-credit-default-risk/data
  2. Accept competition rules and download these 6 files into data/raw/:
       - application_train.csv
       - bureau.csv
       - bureau_balance.csv
       - installments_payments.csv
       - credit_card_balance.csv
       - previous_application.csv
  3. OR use kaggle CLI (if configured):
       kaggle competitions download -c home-credit-default-risk -p data/raw/ --unzip

This script verifies the files are present and prints row counts.
=============================================================================
"""

import os
import sys

REQUIRED_FILES = {
    "application_train.csv":     "307,511 rows  | Master spine: one row per applicant",
    "bureau.csv":                "1,716,428 rows | Bureau credit history",
    "bureau_balance.csv":        "27,299,925 rows| Monthly bureau status",
    "installments_payments.csv": "13,605,401 rows| EMI schedule vs actual payment",
    "credit_card_balance.csv":   "3,840,312 rows | Monthly card balance & utilization",
    "previous_application.csv":  "1,670,214 rows | Previous loan applications",
}

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")


def check_files():
    print("\n" + "="*65)
    print("  HOME CREDIT DATA — FILE CHECK")
    print("="*65)
    all_ok = True
    for fname, desc in REQUIRED_FILES.items():
        fpath = os.path.join(RAW_DIR, fname)
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / 1e6
            print(f"  ✅  {fname:<35} {size_mb:>7.1f} MB")
        else:
            print(f"  ❌  {fname:<35} MISSING — {desc}")
            all_ok = False
    print("="*65)
    if all_ok:
        print("\n  All files present. Run stage1_feature_build.py next.\n")
    else:
        print("\n  Download missing files from Kaggle first.")
        print("  See instructions at the top of this script.\n")
    return all_ok


if __name__ == "__main__":
    ok = check_files()
    sys.exit(0 if ok else 1)
