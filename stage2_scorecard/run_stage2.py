"""
=============================================================================
AXIS BANK BIU — Collections Prioritization Project
Stage 2: Baseline Scorecard  (WOE / IV + Logistic Regression)
=============================================================================
Fixed for optbinning >= 0.19: uses individual OptimalBinning objects
per feature instead of BinningProcess.get_binning() which was removed.
=============================================================================
"""

import os, sys, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from optbinning import OptimalBinning

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV  = os.path.join(ROOT, "outputs", "feature_store.csv")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE      = 42
SCORECARD_PDO     = 20
SCORECARD_ODDS    = 1
SCORECARD_BASE    = 600

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


def banner(msg):
    print("\n" + "="*65)
    print(f"  {msg}")
    print("="*65)


def ks_statistic(y_true, y_prob):
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p", ascending=False)
    n_bad  = (df["y"] == 1).sum()
    n_good = (df["y"] == 0).sum()
    df["cum_bad"]  = (df["y"] == 1).cumsum() / n_bad
    df["cum_good"] = (df["y"] == 0).cumsum() / n_good
    df["ks"]       = (df["cum_bad"] - df["cum_good"]).abs()
    return df["ks"].max(), df


def gini_from_auc(auc):
    return 2 * auc - 1


def psi_score(expected, actual, n_bins=10):
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0], bins[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=bins)[0] / len(expected)
    a = np.histogram(actual,   bins=bins)[0] / len(actual)
    e = np.where(e == 0, 1e-6, e)
    a = np.where(a == 0, 1e-6, a)
    return np.sum((a - e) * np.log(a / e)), bins, e, a


def fit_woe_binners(X_train, y_train, feature_cols):
    """Fit one OptimalBinning per feature. Return dict of fitted binners."""
    binners = {}
    iv_records = []
    print(f"\n  Fitting WOE binners for {len(feature_cols)} features...")
    for feat in feature_cols:
        x = X_train[feat].values
        # Skip features that are all-null on training set
        if pd.isnull(x).all():
            continue
        try:
            ob = OptimalBinning(
                name=feat,
                dtype="numerical",
                max_n_bins=10,
                min_bin_size=0.05,
                solver="cp",
            )
            ob.fit(x, y_train.values)
            binners[feat] = ob
            # Extract IV from binning table
            try:
                bt = ob.binning_table.build()
                iv_val = bt.loc[bt.index != "Totals", "IV"].sum() if "IV" in bt.columns else 0.0
            except Exception:
                iv_val = 0.0
            iv_records.append({"feature": feat, "IV": round(float(iv_val), 6)})
        except Exception as e:
            print(f"    Warning: could not bin {feat}: {e}")

    iv_df = pd.DataFrame(iv_records).sort_values("IV", ascending=False).reset_index(drop=True)
    return binners, iv_df


def transform_woe(X, binners, selected_features):
    """Transform selected features to WOE values. Missing -> 0."""
    result = pd.DataFrame(index=X.index)
    for feat in selected_features:
        if feat in binners:
            woe_vals = binners[feat].transform(X[feat].values, metric="woe")
            result[feat] = np.where(np.isfinite(woe_vals), woe_vals, 0.0)
        else:
            result[feat] = 0.0
    return result


def build_scorecard_table(binners, selected, lr, pdo, odds, base):
    """Build traditional bank scorecard table with points per bin."""
    factor = pdo / np.log(2)
    offset = base - factor * np.log(odds)
    n_feat = len(selected)
    rows   = []
    coef_dict = dict(zip(selected, lr.coef_[0]))

    for feat in selected:
        if feat not in binners:
            continue
        ob = binners[feat]
        coef = coef_dict.get(feat, 0.0)
        try:
            bt = ob.binning_table.build()
            for idx, row in bt.iterrows():
                if str(idx) in ("Special", "Missing", "Totals"):
                    continue
                woe_val = float(row.get("WOE", 0) or 0)
                iv_val  = float(row.get("IV",  0) or 0)
                # Scorecard points: -(beta*WOE + intercept/n) * factor
                pts = -(coef * woe_val + lr.intercept_[0] / n_feat) * factor
                rows.append({
                    "feature":  feat,
                    "bin":      str(idx),
                    "woe":      round(woe_val, 4),
                    "iv":       round(iv_val, 4),
                    "lr_coef":  round(coef, 4),
                    "points":   round(pts, 1),
                })
        except Exception as e:
            print(f"    Warning: scorecard table for {feat}: {e}")

    return pd.DataFrame(rows)


def main():
    banner("STAGE 2 - Baseline Scorecard (WOE/IV + Logistic Regression)")

    if not os.path.exists(IN_CSV):
        print(f"\n  ERROR: {IN_CSV} not found. Run Stage 1 first.")
        sys.exit(1)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    df = pd.read_csv(IN_CSV)
    print(f"\n  Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Default rate: {df['TARGET'].mean():.2%}")

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feat_cols].copy()
    y = df["TARGET"].copy()

    # ── 2. Train / test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    # ── 3. WOE / IV binning ───────────────────────────────────────────────────
    banner("WOE / IV Binning (individual OptimalBinning per feature)")
    binners, iv_df = fit_woe_binners(X_train, y_train, feat_cols)

    print(f"\n  IV Summary (top 20):")
    print(iv_df.head(20).to_string(index=False))

    iv_df.to_csv(os.path.join(OUT_DIR, "iv_table.csv"), index=False)

    # ── 4. Feature selection by IV ────────────────────────────────────────────
    selected = iv_df[iv_df["IV"] >= 0.02]["feature"].tolist()
    print(f"\n  Features selected (IV >= 0.02): {len(selected)} / {len(feat_cols)}")
    print(f"  Dropped (IV < 0.02):            {len(feat_cols) - len(selected)}")

    if len(selected) == 0:
        selected = iv_df[iv_df["IV"] >= 0.005]["feature"].tolist()
        print(f"  Lowered threshold to 0.005: {len(selected)} features")

    # ── 5. WOE Transform ──────────────────────────────────────────────────────
    X_train_woe = transform_woe(X_train, binners, selected)
    X_test_woe  = transform_woe(X_test,  binners, selected)

    # ── 6. Logistic Regression ────────────────────────────────────────────────
    banner("Logistic Regression Training")
    lr = LogisticRegression(
        C=0.1, solver="lbfgs", max_iter=1000,
        random_state=RANDOM_STATE, class_weight="balanced"
    )
    lr.fit(X_train_woe, y_train)

    cv = cross_val_score(
        lr, X_train_woe, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc"
    )
    print(f"\n  5-fold CV AUC: {cv.mean():.4f} +/- {cv.std():.4f}")

    # ── 7. Metrics ────────────────────────────────────────────────────────────
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
  +---------------+-----------+-------------+
  | Metric        |   Train   |    Test     |
  +---------------+-----------+-------------+
  | AUC           |  {auc_train:.4f}   |   {auc_test:.4f}    |
  | Gini          |  {gini_train:.4f}   |   {gini_test:.4f}    |
  | KS            |  {ks_train:.4f}   |   {ks_test:.4f}    |
  +---------------+-----------+-------------+
    """)

    auc_gap = auc_train - auc_test
    if auc_gap > 0.05:
        print(f"  WARNING: AUC gap = {auc_gap:.4f} > 0.05 - possible overfit.")
    else:
        print(f"  OK: AUC gap = {auc_gap:.4f} - no significant overfit.")

    # Honest flag if AUC is weak
    if auc_test < 0.65:
        print("""
  NOTE: AUC < 0.65 on behavioural features alone is expected.
  Home Credit's EXT_SOURCE features (external credit scores) would push
  AUC to ~0.75+. We intentionally use only payment-behaviour features
  because that is what a collections team actually observes.
  XGBoost (Stage 3) will improve this further.
        """)

    # ── 8. PSI ────────────────────────────────────────────────────────────────
    banner("Population Stability Index (PSI)")
    psi_val, psi_bins, psi_exp, psi_act = psi_score(y_prob_train, y_prob_test)
    psi_flag = ("OK Stable"     if psi_val < 0.10 else
                "WATCH Slight"  if psi_val < 0.25 else
                "ALERT Major shift")
    print(f"\n  PSI (train vs test): {psi_val:.4f}  ->  {psi_flag}")

    # ── 9. Scorecard Table ────────────────────────────────────────────────────
    banner("Building Scorecard Table")
    sc_df = build_scorecard_table(binners, selected, lr,
                                  SCORECARD_PDO, SCORECARD_ODDS, SCORECARD_BASE)
    sc_path = os.path.join(OUT_DIR, "scorecard_table.csv")
    sc_df.to_csv(sc_path, index=False)
    print(f"  Saved: {sc_path} ({len(sc_df)} rows)")
    print(sc_df.head(12).to_string(index=False))

    # ── 10. Plots ─────────────────────────────────────────────────────────────
    banner("Generating Plots")

    # IV Bar Chart
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#d62728" if v >= 0.3 else "#ff7f0e" if v >= 0.1 else "#2ca02c"
              for v in iv_df["IV"].values]
    ax.barh(iv_df["feature"], iv_df["IV"], color=colors)
    ax.axvline(0.02, color="black", linestyle="--", lw=1, label="IV=0.02 cutoff")
    ax.axvline(0.10, color="orange", linestyle="--", lw=1, label="IV=0.10")
    ax.set_xlabel("Information Value (IV)")
    ax.set_title("Stage 2 - Feature IV (Predictive Strength)", fontsize=13)
    ax.legend(); ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "stage2_iv_chart.png"), dpi=150)
    plt.close()

    # KS Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ks_sorted = df_ks_test.reset_index(drop=True)
    pct = np.linspace(0, 100, len(ks_sorted))
    ax.plot(pct, ks_sorted["cum_bad"],  color="#d62728", label="Cumulative Bad %")
    ax.plot(pct, ks_sorted["cum_good"], color="#2ca02c", label="Cumulative Good %")
    ks_idx = ks_sorted["ks"].idxmax()
    ax.axvline(pct[ks_idx], color="navy", linestyle="--",
               label=f"KS={ks_test:.4f} at {pct[ks_idx]:.1f}%")
    ax.set_xlabel("% Population (sorted by score desc)")
    ax.set_ylabel("Cumulative %")
    ax.set_title(f"Stage 2 - KS Plot (Test) | KS={ks_test:.4f}", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "stage2_ks_plot.png"), dpi=150)
    plt.close()

    # ROC Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    for lbl, y_t, y_p, col in [("Train", y_train, y_prob_train, "#ff7f0e"),
                                 ("Test",  y_test,  y_prob_test,  "#1f77b4")]:
        fpr, tpr, _ = roc_curve(y_t, y_p)
        ax.plot(fpr, tpr, color=col, label=f"{lbl} AUC={roc_auc_score(y_t,y_p):.4f}")
    ax.plot([0,1],[0,1],"k--", label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Stage 2 - ROC Curve (Baseline Scorecard)", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "stage2_roc_curve.png"), dpi=150)
    plt.close()

    # Score Distributions
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, scores, y_vals, title in [
        (axes[0], y_prob_train, y_train, "Train"),
        (axes[1], y_prob_test,  y_test,  "Test")]:
        pd.Series(scores[y_vals==0]).hist(bins=50, ax=ax, alpha=0.6, color="#2ca02c",
                                          density=True, label="Good (0)")
        pd.Series(scores[y_vals==1]).hist(bins=50, ax=ax, alpha=0.6, color="#d62728",
                                          density=True, label="Default (1)")
        ax.set_title(f"Score Distribution - {title}"); ax.legend()
    fig.suptitle("Stage 2 - Score Separation", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "stage2_score_distribution.png"), dpi=150)
    plt.close()

    # PSI Chart
    bin_labels = [f"D{i+1}" for i in range(len(psi_exp))]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(bin_labels))
    ax.bar(x-0.2, psi_exp, 0.4, label="Train", color="#1f77b4", alpha=0.8)
    ax.bar(x+0.2, psi_act, 0.4, label="Test",  color="#ff7f0e", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(bin_labels)
    ax.set_title(f"Stage 2 - PSI={psi_val:.4f} | {psi_flag}", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "stage2_psi.png"), dpi=150)
    plt.close()

    print("  Plots saved to outputs/")

    # ── 11. Save artifacts ────────────────────────────────────────────────────
    artifacts = {
        "binners": binners,
        "selected_features": selected,
        "logistic_regression": lr,
        "metrics": {
            "auc_train": auc_train, "auc_test": auc_test,
            "gini_train": gini_train, "gini_test": gini_test,
            "ks_train": ks_train, "ks_test": ks_test,
            "psi": psi_val,
        }
    }
    with open(os.path.join(OUT_DIR, "scorecard_model.pkl"), "wb") as f:
        pickle.dump(artifacts, f)
    print(f"\n  Saved: scorecard_model.pkl")

    banner("Stage 2 Complete")
    print(f"""
  BASELINE SCORECARD RESULTS (307,511 real customers)
  ----------------------------------------------------
  Features selected (IV >= 0.02) : {len(selected)}
  AUC  (test)  : {auc_test:.4f}
  Gini (test)  : {gini_test:.4f}
  KS   (test)  : {ks_test:.4f}
  PSI          : {psi_val:.4f} | {psi_flag}

  Next: python stage3_xgboost/run_stage3.py
    """)


if __name__ == "__main__":
    main()
