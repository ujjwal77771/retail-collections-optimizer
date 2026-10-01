"""
=============================================================================
AXIS BANK BIU — Collections Prioritization Project
Stage 1: Feature Build Runner
=============================================================================
This script:
  1. Opens a DuckDB database (persisted to disk)
  2. Runs feature_engineering.sql to build ins/cc/bur/feature_store tables
  3. Runs 4 validation queries and prints results
  4. Exports feature_store to CSV for Stage 2 (scorecard)
  5. Produces a feature correlation heatmap and missing-value chart

Run from project root:
    python stage1_sql/run_stage1.py
=============================================================================
"""

import os
import sys
import time

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQL_FILE = os.path.join(ROOT, "stage1_sql", "feature_engineering.sql")
DB_FILE  = os.path.join(ROOT, "outputs",    "hcdr.duckdb")
OUT_CSV  = os.path.join(ROOT, "outputs",    "feature_store.csv")
OUT_DIR  = os.path.join(ROOT, "outputs")

os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def banner(msg: str):
    print("\n" + "="*65)
    print(f"  {msg}")
    print("="*65)


def run_sql_file(con: duckdb.DuckDBPyConnection, sql_path: str):
    """
    Execute a .sql file that may contain multiple statements.
    Splits on semicolons, skips validation SELECTs (they run separately).
    Returns the list of validation SELECT statements.
    """
    with open(sql_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Remove block comments (/* ... */) and inline comments (--)
    import re
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.DOTALL)
    lines = [l for l in raw.splitlines() if not l.strip().startswith("--")]
    clean = "\n".join(lines)

    statements = [s.strip() for s in clean.split(";") if s.strip()]
    validation_selects = []

    for stmt in statements:
        upper = stmt.upper().lstrip()
        if upper.startswith("SELECT"):
            validation_selects.append(stmt)
        else:
            con.execute(stmt)

    return validation_selects


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    banner("STAGE 1 — SQL Feature Engineering (DuckDB)")

    # 1. Check raw data exists
    raw_dir = os.path.join(ROOT, "data", "raw")
    required = [
        "application_train.csv",
        "bureau.csv",
        "bureau_balance.csv",
        "installments_payments.csv",
        "credit_card_balance.csv",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(raw_dir, f))]
    if missing:
        print(f"\n❌  Missing data files: {missing}")
        print("   Run download_data.py first. Aborting.")
        sys.exit(1)

    # 2. Connect to DuckDB (creates hcdr.duckdb if not exists)
    print(f"\n  Connecting to DuckDB: {DB_FILE}")
    con = duckdb.connect(DB_FILE)
    # Allow DuckDB to use multiple threads for speed
    con.execute("SET threads TO 4;")
    con.execute("SET memory_limit = '4GB';")

    # 3. Run SQL feature engineering
    banner("Building feature tables...")
    t0 = time.time()
    validation_queries = run_sql_file(con, SQL_FILE)
    elapsed = time.time() - t0
    print(f"\n  ✅  Feature tables built in {elapsed:.1f}s")

    # 4. Run validation queries
    banner("Validation Queries")
    val_labels = [
        "6a — Row count & default rate",
        "6b — Feature completeness (% non-null)",
        "6c — Default rate by n_products_delinquent",
        "6d — Avg feature values: defaulters vs. non-defaulters",
    ]
    for label, query in zip(val_labels, validation_queries):
        print(f"\n  ▶  {label}")
        result = con.execute(query).df()
        print(result.to_string(index=False))

    # 5. Load feature store into pandas
    banner("Loading feature_store into pandas...")
    df = con.execute("SELECT * FROM feature_store").df()
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Default rate: {df['TARGET'].mean():.2%}")

    # 6. Export to CSV for Stage 2
    df.to_csv(OUT_CSV, index=False)
    print(f"\n  ✅  Exported to: {OUT_CSV}")

    # 7. Missing value analysis
    banner("Missing Value Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ("SK_ID_CURR", "TARGET")]

    miss_pct = df[numeric_cols].isnull().mean().sort_values(ascending=False)
    print("\n  Top 15 columns by missing %:")
    print(miss_pct.head(15).apply(lambda x: f"{x:.1%}").to_string())

    # Plot missing values
    fig, ax = plt.subplots(figsize=(10, 6))
    miss_pct.head(20).plot(kind="barh", ax=ax, color="#d62728")
    ax.set_xlabel("Missing %")
    ax.set_title("Stage 1 — Feature Store: Top 20 Columns by Missing %", fontsize=13)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.invert_yaxis()
    plt.tight_layout()
    miss_path = os.path.join(OUT_DIR, "stage1_missing_values.png")
    plt.savefig(miss_path, dpi=150)
    plt.close()
    print(f"\n  📊 Missing-value chart saved: {miss_path}")

    # 8. Feature correlation heatmap (numeric features only, sampled)
    banner("Feature Correlation Heatmap")
    key_features = [
        "ins_days_late_mean", "ins_underpay_ratio_mean", "ins_pct_late",
        "ins_late_momentum", "ins_days_late_l3m",
        "cc_util_mean", "cc_util_l3m", "cc_util_trend",
        "cc_dpd_max", "cc_min_pay_ratio_mean",
        "bur_delinq_rate", "bur_delinq_rate_12m",
        "bur_debt_credit_ratio", "bur_worst_status_ever",
        "stress_composite", "n_products_delinquent",
        "TARGET"
    ]
    key_features = [c for c in key_features if c in df.columns]
    corr = df[key_features].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn_r",
        vmin=-1, vmax=1, linewidths=0.5, ax=ax,
        annot_kws={"size": 7}
    )
    ax.set_title("Stage 1 — Feature Correlation Matrix\n(red = positive correlation with default)",
                 fontsize=12)
    plt.tight_layout()
    corr_path = os.path.join(OUT_DIR, "stage1_correlation_heatmap.png")
    plt.savefig(corr_path, dpi=150)
    plt.close()
    print(f"  📊 Correlation heatmap saved: {corr_path}")

    # 9. Target correlation — which features predict default best?
    banner("Feature-Target Correlation (Predictive Power Proxy)")
    target_corr = df[key_features].corr()["TARGET"].drop("TARGET").abs().sort_values(ascending=False)
    print("\n  Top 10 features by |correlation with TARGET|:")
    print(target_corr.head(10).apply(lambda x: f"{x:.4f}").to_string())

    # 10. Distribution plot: defaulters vs. non-defaulters on top features
    banner("Generating Separation Plots...")
    top3 = target_corr.head(3).index.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, feat in zip(axes, top3):
        sub = df[[feat, "TARGET"]].dropna()
        sub[sub["TARGET"] == 0][feat].clip(
            sub[feat].quantile(0.01), sub[feat].quantile(0.99)
        ).hist(bins=50, ax=ax, alpha=0.6, color="#2ca02c", label="Good (0)", density=True)
        sub[sub["TARGET"] == 1][feat].clip(
            sub[feat].quantile(0.01), sub[feat].quantile(0.99)
        ).hist(bins=50, ax=ax, alpha=0.6, color="#d62728", label="Default (1)", density=True)
        ax.set_title(feat, fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("Stage 1 — Top 3 Features: Defaulters vs. Non-Defaulters", fontsize=12)
    plt.tight_layout()
    sep_path = os.path.join(OUT_DIR, "stage1_separation_plots.png")
    plt.savefig(sep_path, dpi=150)
    plt.close()
    print(f"  📊 Separation plots saved: {sep_path}")

    banner("Stage 1 Complete ✅")
    print(f"""
  Summary:
    • feature_store: {df.shape[0]:,} customers × {df.shape[1]} columns
    • Default rate  : {df['TARGET'].mean():.2%}
    • Saved to      : {OUT_CSV}

  Next: Run stage2_scorecard/run_stage2.py  (WOE/IV + Logistic Regression)
    """)

    con.close()


if __name__ == "__main__":
    main()
