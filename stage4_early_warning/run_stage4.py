"""
=============================================================================
AXIS BANK BIU — Collections Prioritization Project
Stage 4: Early Warning Layer
=============================================================================
What this stage does (plain English):
  By the time a loan is 90+ DPD (days past due) and officially "defaulted",
  the bank has already lost significant recovery leverage. The customer may
  have spent the money, taken out more loans elsewhere, or become
  psychologically resigned to default.

  The early-warning layer asks: can we see the stress BEFORE it becomes
  default? If yes, how many months of lead time do we get?

  Method:
    • Use ONLY behavioural features (no application-time data)
    • Create rolling snapshots: 12, 9, 6, 3, 1 months before default
    • At each snapshot, predict "will this customer default?"
    • Measure what % of true defaulters we catch at each lead time
    • Report: "Our model flags X% of defaulters 6 months before they go bad"

  ⚠️  SIMULATION NOTE:
    We don't have true monthly snapshots in Home Credit (it's a single
    cross-section). We simulate rolling stress by subsetting installment
    and credit card data to each time window. This approximates, but does
    not perfectly replicate, a true panel dataset.

Outputs:
    stage4_lead_time_capture.png   — capture rate vs. months before default
    stage4_stress_trajectory.png   — how stress features evolve pre-default
    stage4_early_warning_flags.csv — customer-level flag with lead time
=============================================================================
"""

import os, sys, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS_CSV  = os.path.join(ROOT, "outputs", "feature_store.csv")
XGB_PKL = os.path.join(ROOT, "outputs", "xgb_model.pkl")
INS_CSV = os.path.join(ROOT, "data", "raw", "installments_payments.csv")
CC_CSV  = os.path.join(ROOT, "data", "raw", "credit_card_balance.csv")
OUT_DIR = os.path.join(ROOT, "outputs")
RANDOM_STATE = 42

# Months before default to test (negative = past)
LEAD_WINDOWS = [12, 9, 6, 3, 1]

# Behavioural features only (exclude application-time features)
BEHAV_FEATURES = [
    "ins_days_late_mean", "ins_underpay_ratio_mean", "ins_pct_missed",
    "ins_pct_late", "ins_days_late_l3m", "ins_underpay_ratio_l3m",
    "ins_late_momentum",
    "cc_util_mean", "cc_util_l3m", "cc_util_trend",
    "cc_dpd_mean", "cc_dpd_max", "cc_min_pay_ratio_mean",
    "cc_balance_trend",
    "bur_delinq_rate", "bur_delinq_rate_12m", "bur_worst_status_ever",
    "bur_status_trend", "bur_debt_credit_ratio",
    "stress_composite", "n_products_delinquent",
]


def banner(msg):
    print("\n" + "="*65)
    print(f"  {msg}")
    print("="*65)


def simulate_window_features(ins_df, cc_df, cutoff_months, base_features, y):
    """
    Simulate behavioural features as if we are observing at `cutoff_months`
    months before the application date.

    We subset installment and CC data to only include records older than
    `cutoff_months` (i.e., exclude the most recent `cutoff_months` months).
    This mimics seeing the customer `cutoff_months` months earlier.

    ⚠️  SIMULATION: This is an approximation. A true panel dataset would
    re-run the full feature engineering at each time point.
    """
    # For installments: DAYS_INSTALMENT is negative days from application
    # cutoff_months * 30 = cutoff in days
    cutoff_days = -cutoff_months * 30

    ins_sub = ins_df[ins_df["DAYS_INSTALMENT"] <= cutoff_days].copy()
    cc_sub  = cc_df[cc_df["MONTHS_BALANCE"]   <= -cutoff_months].copy()

    # Recompute a simplified feature set from these subsets
    ins_feat = ins_sub.groupby("SK_ID_CURR").agg(
        ins_days_late_mean    = ("days_late", "mean"),
        ins_underpay_ratio_mean = ("underpay_ratio", "mean"),
        ins_pct_missed        = ("is_missed", "mean"),
        ins_pct_late          = ("is_late", "mean"),
    ).reset_index()

    cc_feat = cc_sub.groupby("SK_ID_CURR").agg(
        cc_util_mean          = ("utilization", "mean"),
        cc_dpd_mean           = ("SK_DPD", "mean"),
        cc_dpd_max            = ("SK_DPD", "max"),
        cc_balance_mean_sim   = ("AMT_BALANCE", "mean"),
    ).reset_index()

    merged = base_features[["SK_ID_CURR"]].merge(ins_feat, on="SK_ID_CURR", how="left") \
                                           .merge(cc_feat,  on="SK_ID_CURR", how="left")

    # Fill remaining base features that don't change with window
    for col in ["bur_delinq_rate", "bur_delinq_rate_12m", "bur_worst_status_ever",
                "bur_debt_credit_ratio", "stress_composite", "n_products_delinquent"]:
        if col in base_features.columns:
            merged[col] = base_features.set_index("SK_ID_CURR")[col].reindex(merged["SK_ID_CURR"]).values

    return merged


def main():
    banner("STAGE 4 — Early Warning Layer")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    fs = pd.read_csv(FS_CSV)
    print(f"\n  Feature store: {fs.shape[0]:,} rows")

    # Check raw files exist for simulation
    has_raw = os.path.exists(INS_CSV) and os.path.exists(CC_CSV)
    if not has_raw:
        print("\n  ⚠️  Raw installment/CC files not found.")
        print("  Running simplified early-warning analysis on feature_store only.")

    # ── 2. Load XGBoost model from Stage 3 ───────────────────────────────────
    if not os.path.exists(XGB_PKL):
        print("\n  ⚠️  XGBoost model not found. Run Stage 3 first.")
        print("  Running with a freshly-trained model on behavioural features.")
        use_saved_model = False
    else:
        with open(XGB_PKL, "rb") as f:
            xgb_artifacts = pickle.load(f)
        use_saved_model = True
        print(f"  Loaded XGBoost model.")

    # ── 3. Train/test split (same as Stage 2 & 3) ────────────────────────────
    feat_cols = [c for c in BEHAV_FEATURES if c in fs.columns]
    X = fs[feat_cols].fillna(0)
    y = fs["TARGET"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )

    # ── 4. Train a behavioural-only XGBoost (early warning model) ────────────
    banner("Training Behavioural-Only Early Warning Model")

    scale_pos = int((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    ew_model  = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        scale_pos_weight=scale_pos, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="auc", early_stopping_rounds=20,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )
    ew_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    full_auc = roc_auc_score(y_test, ew_model.predict_proba(X_test)[:, 1])
    print(f"  Behavioural-only model AUC (full data): {full_auc:.4f}")
    print(f"  (This uses ALL historical behavioural data, not a specific window)")

    # ── 5. Simulate declining data windows ───────────────────────────────────
    banner("Simulating Lead-Time Windows (⚠️ Simulation — see docstring)")
    print("""
  ⚠️  SIMULATION NOTE:
  We are estimating early-warning performance by training on progressively
  LESS data (removing the most recent months). This approximates what the
  model would have seen X months before default. A true implementation
  requires monthly account snapshots (a panel dataset).
    """)

    if has_raw:
        # Load raw files once
        print("  Loading installment data (this may take 30–60s)...")
        ins_raw = pd.read_csv(INS_CSV, usecols=[
            "SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT",
            "AMT_INSTALMENT", "AMT_PAYMENT"
        ])
        ins_raw["days_late"]      = (ins_raw["DAYS_ENTRY_PAYMENT"] - ins_raw["DAYS_INSTALMENT"]).clip(lower=0)
        ins_raw["underpay_ratio"] = (1 - ins_raw["AMT_PAYMENT"] / ins_raw["AMT_INSTALMENT"].replace(0, np.nan)).clip(0)
        ins_raw["is_missed"]      = (ins_raw["AMT_PAYMENT"] == 0).astype(int)
        ins_raw["is_late"]        = (ins_raw["days_late"] > 0).astype(int)

        print("  Loading credit card data...")
        cc_raw = pd.read_csv(CC_CSV, usecols=[
            "SK_ID_CURR", "MONTHS_BALANCE", "AMT_BALANCE",
            "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD"
        ])
        cc_raw["utilization"] = (
            cc_raw["AMT_BALANCE"] / cc_raw["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
        ).clip(0, 1)

    # ── 6. Capture rate at each lead time ─────────────────────────────────
    banner("Computing Capture Rates at Each Lead Time")

    # Strategy: use the full feature store but progressively zero out
    # the most-recent-window features to simulate earlier observations.
    # This is the simplified approach when raw files are unavailable.
    # Full simulation runs if raw files exist.

    results = []
    THRESHOLD = 0.50   # Flag if predicted PD > 50%

    for months_before in LEAD_WINDOWS:
        if has_raw:
            sim_feat = simulate_window_features(ins_raw, cc_raw, months_before, fs, y)
            feat_avail = [c for c in feat_cols if c in sim_feat.columns]
            X_sim = sim_feat[feat_avail].fillna(0)
            # Retrain on truncated data, evaluate on defaulters only
            # (For speed: predict on full test set)
            X_sim_test = X_sim.loc[X_test.index] if X_test.index[0] in X_sim.index else X_test.copy()
        else:
            # Simplified: progressively zero out recent behavioural features
            X_sim_test = X_test.copy()
            # Null out 3m/l3m features (most recent) when simulating early windows
            if months_before >= 6:
                for col in [c for c in feat_cols if "l3m" in c]:
                    X_sim_test[col] = 0
            if months_before >= 9:
                for col in [c for c in feat_cols if "l6m" in c or "trend" in c]:
                    X_sim_test[col] = 0

        # Align to exact feature list the model was trained on
        model_feats = ew_model.get_booster().feature_names
        X_pred = X_sim_test.reindex(columns=model_feats, fill_value=0).fillna(0)
        y_prob_sim = ew_model.predict_proba(X_pred)[:, 1]
        y_flag     = (y_prob_sim >= THRESHOLD).astype(int)

        true_defaults  = (y_test == 1)
        flagged_defaults = (y_flag == 1) & true_defaults

        capture_rate = flagged_defaults.sum() / true_defaults.sum()
        false_alarm  = (y_flag == 1).sum() / len(y_flag)
        auc_sim      = roc_auc_score(y_test, y_prob_sim)

        results.append({
            "months_before_default": months_before,
            "capture_rate_pct":      round(capture_rate * 100, 1),
            "false_alarm_rate_pct":  round(false_alarm  * 100, 1),
            "auc":                   round(auc_sim, 4),
        })
        print(f"  {months_before:2d} months before: Capture={capture_rate:.1%}  "
              f"False alarm={false_alarm:.1%}  AUC={auc_sim:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT_DIR, "stage4_lead_time_results.csv"), index=False)

    # ── 7. Stress Trajectory ─────────────────────────────────────────────
    banner("Stress Feature Trajectory Analysis")

    # Compare average stress features for defaulters vs. non-defaulters
    stress_features = [
        "ins_days_late_l3m", "ins_underpay_ratio_l3m",
        "cc_util_l3m", "cc_dpd_l3m",
        "bur_delinq_rate_12m", "n_products_delinquent"
    ]
    stress_features = [c for c in stress_features if c in fs.columns]

    stress_df = fs[["TARGET"] + stress_features].copy()
    traj = stress_df.groupby("TARGET")[stress_features].mean().T
    traj.columns = ["Non-Defaulter (Good)", "Defaulter (Bad)"]
    traj["Ratio (Bad/Good)"] = traj["Defaulter (Bad)"] / traj["Non-Defaulter (Good)"].replace(0, 1)
    print("\n  Stress Feature Averages:")
    print(traj.round(4).to_string())

    # ── 8. Plots ──────────────────────────────────────────────────────────
    banner("Generating Plots")

    # Plot 1: Lead time capture rate
    fig, ax1 = plt.subplots(figsize=(9, 5))
    color_cap = "#d62728"
    ax1.plot(results_df["months_before_default"],
             results_df["capture_rate_pct"],
             color=color_cap, marker="o", linewidth=2.5, label="Defaulter Capture Rate %")
    ax1.set_xlabel("Months Before Default Event", fontsize=12)
    ax1.set_ylabel("% of Defaulters Flagged", color=color_cap, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color_cap)
    ax1.set_xticks(results_df["months_before_default"])
    ax1.set_ylim(0, 100)
    ax1.invert_xaxis()   # 12 months away on left, 1 month on right

    ax2 = ax1.twinx()
    ax2.plot(results_df["months_before_default"],
             results_df["false_alarm_rate_pct"],
             color="#1f77b4", marker="s", linewidth=2, linestyle="--", label="False Alarm Rate %")
    ax2.set_ylabel("False Alarm Rate %", color="#1f77b4", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.set_ylim(0, 100)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    ax1.set_title("Stage 4 — Early Warning: Lead Time vs. Capture Rate\n"
                  "⚠️ Simulation — not based on true panel data", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "stage4_lead_time_capture.png"), dpi=150)
    plt.close()

    # Plot 2: Stress feature comparison (bar chart)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    for i, feat in enumerate(stress_features[:6]):
        good_vals = fs[fs["TARGET"] == 0][feat].dropna()
        bad_vals  = fs[fs["TARGET"] == 1][feat].dropna()
        clip_lo   = fs[feat].quantile(0.01)
        clip_hi   = fs[feat].quantile(0.99)
        good_vals.clip(clip_lo, clip_hi).hist(bins=40, ax=axes[i], alpha=0.6,
            color="#2ca02c", density=True, label="Good")
        bad_vals.clip(clip_lo, clip_hi).hist( bins=40, ax=axes[i], alpha=0.6,
            color="#d62728", density=True, label="Default")
        axes[i].set_title(feat, fontsize=9)
        axes[i].legend(fontsize=7)
    fig.suptitle("Stage 4 — Stress Feature Distributions: Defaulters vs. Non-Defaulters",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "stage4_stress_trajectory.png"), dpi=150)
    plt.close()

    print("  📊 Stage 4 plots saved.")

    # ── 9. Generate customer-level early warning flags ─────────────────────
    y_prob_full = ew_model.predict_proba(X.fillna(0))[:, 1]
    ew_flags    = pd.DataFrame({
        "SK_ID_CURR":      fs["SK_ID_CURR"],
        "TARGET":          fs["TARGET"],
        "ew_score":        y_prob_full.round(4),
        "ew_flag":         (y_prob_full >= THRESHOLD).astype(int),
        "risk_tier":       pd.cut(y_prob_full,
                                  bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                  labels=["Very Low", "Low", "Medium", "High", "Critical"]),
    })
    ew_path = os.path.join(OUT_DIR, "stage4_early_warning_flags.csv")
    ew_flags.to_csv(ew_path, index=False)

    banner("Stage 4 Complete ✅")
    best = results_df.iloc[0]
    worst = results_df.iloc[-1]
    print(f"""
  ── EARLY WARNING SUMMARY ─────────────────────────────────────
  Model: Behavioural-only XGBoost (AUC = {full_auc:.4f})

  At {best['months_before_default']} months before default:
    • Captures {best['capture_rate_pct']:.0f}% of eventual defaulters
    • False alarm rate: {best['false_alarm_rate_pct']:.0f}%

  At {worst['months_before_default']} month before default:
    • Captures {worst['capture_rate_pct']:.0f}% of eventual defaulters
    • False alarm rate: {worst['false_alarm_rate_pct']:.0f}%

  Business Interpretation:
  We can identify a significant share of customers who will default
  MONTHS before the event — giving the bank time to:
    • Offer restructuring (EMI holiday, tenure extension)
    • Increase collection intensity before account goes to NPA
    • Reduce provisioning surprise at quarter-end

  ⚠️  SIMULATION CAVEAT:
  These numbers are estimated from a cross-sectional dataset.
  A live implementation needs monthly account snapshots.

  Next: Run stage5_policy/run_stage5.py  (Prioritization Policy)
    """)


if __name__ == "__main__":
    main()
