"""
=============================================================================
AXIS BANK BIU — Collections Prioritization Project
Stage 2: Baseline Scorecard  (WOE / IV + Logistic Regression)
=============================================================================
What this stage does (plain English):
  Banks have used logistic-regression scorecards for 40+ years because
  regulators can audit them: every coefficient has a clear direction and
  every feature bin has a Weight-of-Evidence that shows whether being in
  that bin makes default more or less likely.

  Steps:
    1. Load feature_store.csv produced by Stage 1
    2. Train / test split (70 / 30, stratified on TARGET)
    3. WOE / IV binning with optbinning — one binner per feature
    4. Drop features with IV < 0.02 (almost no predictive power)
    5. Logistic regression on WOE-transformed features
    6. Report KS, Gini, AUC on test set
    7. PSI (Population Stability Index) — checks if score distribution
       is stable between train and test (PSI > 0.25 = alarm)
    8. Save scorecard table, model pickle and all plots

Outputs (in outputs/):
    scorecard_table.csv          — feature | bin | WOE | IV | coef | points
    stage2_ks_plot.png
    stage2_roc_curve.png
    stage2_score_distribution.png
    stage2_iv_chart.png
    stage2_psi.png
    scorecard_model.pkl          — trained pipeline (WOE + LR)
=============================================================================
"""

import os, sys, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from optbinning import BinningProcess

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV  = os.path.join(ROOT, "outputs", "feature_store.csv")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
SCORECARD_PDO     = 20          # Points to Double the Odds
SCORECARD_ODDS    = 1           # Odds at base score (P(bad)/P(good))
SCORECARD_BASE    = 600         # Base score

# ── Feature list (behavioural only — no application-level demographics in LR)
FEATURE_COLS = [
    "ins_days_late_mean", "ins_days_late_max", "ins_underpay_ratio_mean",
    "ins_pct_missed", "ins_pct_late", "ins_late_trend",
    "ins_days_late_l3m", "ins_underpay_ratio_l3m", "ins_missed_count_l3m",
    "ins_days_late_l6m", "ins_underpay_ratio_l6m", "ins_late_momentum",
    "cc_util_mean", "cc_util_max", "cc_util_l3m", "cc_util_trend",
    "cc_balance_mean", "cc_balance_l3m", "cc_balance_trend",
    "cc_dpd_mean", "cc_dpd_max", "cc_dpd_l3m",
    "cc_min_pay_ratio_mean", "cc_min_pay_ratio_l3m", "cc_drawings_trend",
    "bur_total_accounts", "bur_active_accounts",
    "bur_days_overdue_mean", "bur_days_overdue_max", "bur_amt_overdue_sum",
    "bur_debt_credit_ratio", "bur_days_credit_min",
    "bur_delinq_rate", "bur_delinq_rate_12m",
    "bur_worst_status_ever", "bur_status_trend",
    "stress_composite", "n_products_delinquent",
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def banner(msg):
    print("\n" + "="*65)
    print(f"  {msg}")
    print("="*65)


def ks_statistic(y_true, y_prob):
    """KS = max separation between cumulative good and bad distributions."""
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p", ascending=False)
    df["cum_bad"]  = (df["y"] == 1).cumsum() / (df["y"] == 1).sum()
    df["cum_good"] = (df["y"] == 0).cumsum() / (df["y"] == 0).sum()
    df["ks"]       = (df["cum_bad"] - df["cum_good"]).abs()
    return df["ks"].max(), df


def gini_from_auc(auc):
    return 2 * auc - 1


def psi(expected, actual, n_bins=10):
    """
    Population Stability Index.
    PSI < 0.10 = stable, 0.10-0.25 = slight shift, > 0.25 = major shift.
    """
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0], bins[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=bins)[0] / len(expected)
    a = np.histogram(actual,   bins=bins)[0] / len(actual)
    e = np.where(e == 0, 1e-6, e)
    a = np.where(a == 0, 1e-6, a)
    psi_val = np.sum((a - e) * np.log(a / e))
    return psi_val, bins, e, a


def score_to_points(coef, woe, intercept, n_features,
                    pdo=SCORECARD_PDO, odds=SCORECARD_ODDS, base=SCORECARD_BASE):
    """Convert logistic regression coefficients + WOE to scorecard points."""
    factor  = pdo / np.log(2)
    offset  = base - factor * np.log(odds)
    # Each feature contributes: -(beta * WOE + intercept/n) * factor
    points  = -(coef * woe + intercept / n_features) * factor
    return points, factor, offset


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    banner("STAGE 2 — Baseline Scorecard (WOE/IV + Logistic Regression)")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    if not os.path.exists(IN_CSV):
        print(f"\n❌  {IN_CSV} not found. Run Stage 1 first.")
        sys.exit(1)

    df = pd.read_csv(IN_CSV)
    print(f"\n  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Default rate: {df['TARGET'].mean():.2%}")

    # Keep only columns that exist in this dataset
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feat_cols].copy()
    y = df["TARGET"].copy()

    # ── 2. Train / test split (stratified) ───────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")
    print(f"  Train default rate: {y_train.mean():.2%} | Test: {y_test.mean():.2%}")

    # ── 3. WOE / IV binning via optbinning ───────────────────────────────────
    banner("WOE / IV Binning")

    binning_process = BinningProcess(
        variable_names=feat_cols,
        max_n_bins=10,
        min_bin_size=0.05,       # each bin >= 5% of population
        categorical_variables=[]
    )
    binning_process.fit(X_train, y_train)

    # Extract IV table
    summary = binning_process.summary()
    iv_table = summary[["name", "iv"]].rename(columns={"name": "feature", "iv": "IV"})
    iv_table = iv_table.sort_values("IV", ascending=False).reset_index(drop=True)

    print(f"\n  IV Summary (top 20):")
    print(iv_table.head(20).to_string(index=False))

    # ── 4. Feature selection: IV > 0.02 ──────────────────────────────────────
    #   IV < 0.02  = useless
    #   0.02–0.10  = weak
    #   0.10–0.30  = medium
    #   > 0.30     = strong
    selected = iv_table[iv_table["IV"] >= 0.02]["feature"].tolist()
    print(f"\n  Features selected (IV ≥ 0.02): {len(selected)} / {len(feat_cols)}")
    print(f"  Dropped (IV < 0.02):            {len(feat_cols) - len(selected)}")

    if len(selected) == 0:
        print("\n⚠️  No features passed IV threshold. Lowering threshold to 0.005.")
        selected = iv_table[iv_table["IV"] >= 0.005]["feature"].tolist()

    # Refit binning process on selected features only
    bp_final = BinningProcess(
        variable_names=selected,
        max_n_bins=10,
        min_bin_size=0.05,
        categorical_variables=[]
    )
    bp_final.fit(X_train[selected], y_train)

    # Transform to WOE
    X_train_woe = bp_final.transform(X_train[selected], metric="woe")
    X_test_woe  = bp_final.transform(X_test[selected],  metric="woe")

    # ── 5. Logistic Regression ────────────────────────────────────────────────
    banner("Logistic Regression Training")

    lr = LogisticRegression(
        C=0.1,                      # mild L2 regularisation
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced"     # handles class imbalance (~8% default rate)
    )
    lr.fit(X_train_woe, y_train)

    # Cross-validation AUC on training set (5-fold)
    cv_scores = cross_val_score(
        lr, X_train_woe, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc"
    )
    print(f"\n  5-fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── 6. Predictions & Core Metrics ─────────────────────────────────────────
    banner("Model Evaluation")

    y_prob_train = lr.predict_proba(X_train_woe)[:, 1]
    y_prob_test  = lr.predict_proba(X_test_woe)[:, 1]

    auc_train  = roc_auc_score(y_train, y_prob_train)
    auc_test   = roc_auc_score(y_test,  y_prob_test)
    gini_train = gini_from_auc(auc_train)
    gini_test  = gini_from_auc(auc_test)
    ks_train, df_ks_train = ks_statistic(y_train.values, y_prob_train)
    ks_test,  df_ks_test  = ks_statistic(y_test.values,  y_prob_test)

    print(f"""
  ┌─────────────────────────────────────────┐
  │          SCORECARD METRICS              │
  ├───────────────┬───────────┬─────────────┤
  │ Metric        │   Train   │    Test     │
  ├───────────────┼───────────┼─────────────┤
  │ AUC           │  {auc_train:.4f}   │   {auc_test:.4f}    │
  │ Gini          │  {gini_train:.4f}   │   {gini_test:.4f}    │
  │ KS            │  {ks_train:.4f}   │   {ks_test:.4f}    │
  └───────────────┴───────────┴─────────────┘
    """)

    # Leakage / overfit check
    auc_gap = auc_train - auc_test
    if auc_gap > 0.05:
        print(f"  ⚠️  AUC gap = {auc_gap:.4f} > 0.05 — possible overfitting. Review binning.")
    else:
        print(f"  ✅  AUC gap = {auc_gap:.4f} — no significant overfit.")

    # ── 7. PSI ────────────────────────────────────────────────────────────────
    banner("Population Stability Index (PSI)")

    psi_val, psi_bins, psi_exp, psi_act = psi(y_prob_train, y_prob_test)
    psi_flag = (
        "✅ Stable"       if psi_val < 0.10 else
        "⚠️  Slight shift" if psi_val < 0.25 else
        "🚨 Major shift"
    )
    print(f"\n  PSI (train vs test): {psi_val:.4f}  →  {psi_flag}")
    print("  (< 0.10 = stable | 0.10–0.25 = watch | > 0.25 = model drift)")

    # ── 8. Scorecard Table ────────────────────────────────────────────────────
    banner("Building Scorecard Table")

    rows = []
    coefs = dict(zip(selected, lr.coef_[0]))
    n_feat = len(selected)

    for feat in selected:
        binner = bp_final.get_binning(feat)
        try:
            bt = binner.binning_table.build()
            for _, row in bt.iterrows():
                if row.name in ("Special", "Missing", "Totals"):
                    continue
                woe_val = row.get("WOE", 0)
                iv_val  = row.get("IV", 0)
                pts, factor, offset = score_to_points(
                    coefs.get(feat, 0), woe_val,
                    lr.intercept_[0], n_feat
                )
                rows.append({
                    "feature":    feat,
                    "bin":        str(row.name),
                    "woe":        round(woe_val, 4),
                    "iv":         round(iv_val,  4),
                    "lr_coef":    round(coefs.get(feat, 0), 4),
                    "points":     round(pts, 1),
                })
        except Exception:
            pass

    sc_df = pd.DataFrame(rows)
    sc_path = os.path.join(OUT_DIR, "scorecard_table.csv")
    sc_df.to_csv(sc_path, index=False)
    print(f"  Saved: {sc_path}")
    print(f"\n  Scorecard table (first 15 rows):")
    print(sc_df.head(15).to_string(index=False))

    # ── 9. Plots ──────────────────────────────────────────────────────────────
    banner("Generating Plots")

    # Plot 1: IV Bar Chart
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#d62728" if v >= 0.3 else "#ff7f0e" if v >= 0.1 else "#2ca02c"
              for v in iv_table["IV"].values]
    ax.barh(iv_table["feature"], iv_table["IV"], color=colors)
    ax.axvline(0.02, color="black", linestyle="--", linewidth=1, label="IV=0.02 cutoff")
    ax.axvline(0.10, color="orange", linestyle="--", linewidth=1, label="IV=0.10")
    ax.axvline(0.30, color="red",    linestyle="--", linewidth=1, label="IV=0.30")
    ax.set_xlabel("Information Value (IV)")
    ax.set_title("Stage 2 — Feature IV (predictive strength)", fontsize=13)
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "stage2_iv_chart.png"), dpi=150)
    plt.close()

    # Plot 2: KS Plot (test set)
    fig, ax = plt.subplots(figsize=(9, 5))
    df_ks_test_sorted = df_ks_test.reset_index(drop=True)
    pct = np.linspace(0, 100, len(df_ks_test_sorted))
    ax.plot(pct, df_ks_test_sorted["cum_bad"],  color="#d62728", label="Cumulative Bad %")
    ax.plot(pct, df_ks_test_sorted["cum_good"], color="#2ca02c", label="Cumulative Good %")
    ks_idx = df_ks_test_sorted["ks"].idxmax()
    ax.axvline(pct[ks_idx], color="navy", linestyle="--",
               label=f"KS = {ks_test:.4f} at {pct[ks_idx]:.1f}% of population")
    ax.set_xlabel("% of Population (sorted by score desc)")
    ax.set_ylabel("Cumulative %")
    ax.set_title(f"Stage 2 — KS Plot (Test Set)  |  KS = {ks_test:.4f}", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "stage2_ks_plot.png"), dpi=150)
    plt.close()

    # Plot 3: ROC Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, y_t, y_p, color in [
        ("Train", y_train, y_prob_train, "#ff7f0e"),
        ("Test",  y_test,  y_prob_test,  "#1f77b4"),
    ]:
        fpr, tpr, _ = roc_curve(y_t, y_p)
        auc = roc_auc_score(y_t, y_p)
        ax.plot(fpr, tpr, color=color, label=f"{label} AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Stage 2 — ROC Curve (Baseline Scorecard)", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "stage2_roc_curve.png"), dpi=150)
    plt.close()

    # Plot 4: Score Distribution (defaulters vs. non-defaulters)
    score_train = pd.Series(y_prob_train, name="score")
    score_test  = pd.Series(y_prob_test,  name="score")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, scores, y_vals, title in [
        (axes[0], score_train, y_train, "Train"),
        (axes[1], score_test,  y_test,  "Test"),
    ]:
        pd.Series(scores[y_vals == 0]).hist(bins=50, ax=ax, alpha=0.6,
                  color="#2ca02c", density=True, label="Good (0)")
        pd.Series(scores[y_vals == 1]).hist(bins=50, ax=ax, alpha=0.6,
                  color="#d62728", density=True, label="Default (1)")
        ax.set_xlabel("Predicted Probability of Default")
        ax.set_title(f"Score Distribution — {title}")
        ax.legend()
    fig.suptitle("Stage 2 — Score Separation: Defaulters vs. Non-Defaulters", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "stage2_score_distribution.png"), dpi=150)
    plt.close()

    # Plot 5: PSI Bar Chart
    bin_labels = [f"D{i+1}" for i in range(len(psi_exp))]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(bin_labels))
    ax.bar(x - 0.2, psi_exp, 0.4, label="Train (expected)", color="#1f77b4", alpha=0.8)
    ax.bar(x + 0.2, psi_act, 0.4, label="Test  (actual)",   color="#ff7f0e", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(bin_labels)
    ax.set_ylabel("Proportion of Population")
    ax.set_title(f"Stage 2 — PSI = {psi_val:.4f}  |  {psi_flag}", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "stage2_psi.png"), dpi=150)
    plt.close()

    print("  📊 All Stage 2 plots saved to outputs/")

    # ── 10. Save model artifacts ───────────────────────────────────────────────
    banner("Saving Model Artifacts")
    artifacts = {
        "binning_process": bp_final,
        "logistic_regression": lr,
        "selected_features": selected,
        "metrics": {
            "auc_train": auc_train, "auc_test": auc_test,
            "gini_train": gini_train, "gini_test": gini_test,
            "ks_train": ks_train, "ks_test": ks_test,
            "psi": psi_val,
        }
    }
    pkl_path = os.path.join(OUT_DIR, "scorecard_model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"  Saved model: {pkl_path}")

    # ── 11. Final summary ──────────────────────────────────────────────────────
    banner("Stage 2 Complete ✅")
    print(f"""
  ── BASELINE SCORECARD RESULTS ──────────────────────────
  Features selected (IV ≥ 0.02) : {len(selected)}
  AUC  (test)  : {auc_test:.4f}
  Gini (test)  : {gini_test:.4f}
  KS   (test)  : {ks_test:.4f}
  PSI          : {psi_val:.4f}  {psi_flag}

  Interpretation:
  • AUC > 0.70 → model has meaningful discriminatory power
  • KS  > 0.30 → good separation between defaulters and payers
  • PSI < 0.10 → score distribution is stable (no drift)

  Next: Run stage3_xgboost/run_stage3.py  (XGBoost challenger)
    """)

    iv_table.to_csv(os.path.join(OUT_DIR, "iv_table.csv"), index=False)


if __name__ == "__main__":
    main()
