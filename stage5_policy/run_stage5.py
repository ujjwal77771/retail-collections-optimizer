"""
=============================================================================
Collections Prioritization Project
Stage 5: Prioritization Policy  (PD × EAD × LGD × Contact Effectiveness)
=============================================================================
What this stage does (plain English):
  Not all high-PD customers are equally worth calling. A customer with
  80% PD but a ₹10,000 loan balance is less valuable to call than a
  customer with 50% PD and a ₹5,00,000 balance.

  We rank customers by EXPECTED LOSS AVOIDED:
      Priority Score = PD × EAD × LGD × contact_effectiveness

  Where:
    PD  = probability of default (from XGBoost)
    EAD = exposure at default = outstanding loan balance (from application)
    LGD = loss given default = assumed 45% (RBI Basel norm for unsecured retail)
    contact_effectiveness = assumed 30% (if we call, 30% of expected loss
                            is recovered)

  ⚠️  SIMULATION DISCLAIMER:
    contact_effectiveness (0.30) is a SIMULATION ASSUMPTION.
    There is no real call-outcome data in this dataset.
    Do NOT present this as measured uplift.

  Three strategies are compared:
    1. "Call Everyone"    — random order, fill capacity daily
    2. "Highest PD First" — sort by PD score, call top-N
    3. "Our Policy"       — sort by PD × EAD × LGD (Expected Loss Avoided)

Outputs:
    stage5_strategy_comparison.png
    stage5_recovery_curve.png
    stage5_priority_list.csv        — ranked customer list for agents
    stage5_policy_metrics.csv       — summary table for the Excel tool
=============================================================================
"""

import os, sys, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS_CSV  = os.path.join(ROOT, "outputs", "feature_store.csv")
XGB_PKL = os.path.join(ROOT, "outputs", "xgb_model.pkl")
EW_CSV  = os.path.join(ROOT, "outputs", "stage4_early_warning_flags.csv")
OUT_DIR = os.path.join(ROOT, "outputs")

# ── Policy Parameters (easily changed — these feed the Excel tool) ────────────
DAILY_CAPACITY      = 200        # agents × calls/day
LGD_ASSUMED         = 0.45       # 45% — RBI Basel II standard for unsecured retail
CONTACT_EFF         = 0.30       # ⚠️ SIMULATION: 30% of EL recovered if contacted
COST_PER_CALL       = 150        # ₹150 per outbound call (agent cost + infra)
HOURS_PER_AGENT_DAY = 7          # productive hours/agent/day
AVG_CALL_DURATION   = 0.25       # hours per call (15 min)

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


def simulate_recovery(ranked_df, capacity, cost_per_call, contact_eff, strategy_name):
    """
    Simulate collections outcomes for a given ranking strategy.
    Calls top `capacity` customers per day.
    Returns total expected recovery and net profit.

    ⚠️ SIMULATION: All monetary outcomes are estimates based on modelled PD and
    assumed contact_eff. No real call-outcome data is used.
    """
    called  = ranked_df.head(capacity).copy()
    total_el     = called["expected_loss"].sum()
    recovery     = total_el * contact_eff          # ⚠️ simulation assumption
    call_cost    = capacity * cost_per_call
    net_profit   = recovery - call_cost
    agent_hours  = capacity * AVG_CALL_DURATION
    recovery_per_hour = recovery / max(agent_hours, 1)

    return {
        "strategy":           strategy_name,
        "customers_called":   capacity,
        "total_EL_portfolio": ranked_df["expected_loss"].sum().round(2),
        "EL_called":          total_el.round(2),
        "recovery_simulated": recovery.round(2),    # ⚠️ SIMULATION
        "call_cost":          call_cost,
        "net_profit":         net_profit.round(2),  # ⚠️ SIMULATION
        "recovery_per_hour":  recovery_per_hour.round(2),
        "pct_EL_covered":     (total_el / ranked_df["expected_loss"].sum() * 100).round(1),
    }


def main():
    banner("STAGE 5 — Prioritization Policy (PD × EAD × LGD)")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    fs = pd.read_csv(FS_CSV)
    print(f"\n  Feature store: {fs.shape[0]:,} customers")

    # ── 2. Load XGBoost model for PD ─────────────────────────────────────────
    if not os.path.exists(XGB_PKL):
        print("\n  ⚠️  XGBoost model not found. Run Stage 3 first.")
        sys.exit(1)

    with open(XGB_PKL, "rb") as f:
        xgb_art = pickle.load(f)
    xgb_model   = xgb_art["xgb_model"]
    feat_names   = xgb_art["feature_names"]

    # ── 3. Generate PD scores ─────────────────────────────────────────────────
    banner("Generating PD Scores")

    feat_avail = [c for c in feat_names if c in fs.columns]
    X = fs[feat_avail].fillna(0)
    pd_scores = xgb_model.predict_proba(X)[:, 1]
    print(f"  Mean PD: {pd_scores.mean():.4f} | Max: {pd_scores.max():.4f}")

    # ── 4. Build priority table ───────────────────────────────────────────────
    banner("Building Priority Table")

    priority = pd.DataFrame({
        "SK_ID_CURR":  fs["SK_ID_CURR"],
        "TARGET":      fs["TARGET"],
        "PD":          pd_scores.round(4),
        # EAD: use loan amount as proxy (real EAD = outstanding principal)
        "EAD":         fs["app_loan_amount"].fillna(fs["app_loan_amount"].median()),
        "LGD":         LGD_ASSUMED,
    })
    priority["expected_loss"]   = priority["PD"] * priority["EAD"] * priority["LGD"]
    priority["contact_eff"]     = CONTACT_EFF  # ⚠️ simulation label
    priority["recovery_if_called"] = priority["expected_loss"] * CONTACT_EFF

    # Add early warning flag if available
    if os.path.exists(EW_CSV):
        ew = pd.read_csv(EW_CSV)[["SK_ID_CURR", "ew_flag", "risk_tier"]]
        priority = priority.merge(ew, on="SK_ID_CURR", how="left")

    # ── 5. Three strategies ───────────────────────────────────────────────────
    banner("Defining Three Strategies")

    # Strategy 1: Random (call everyone = no model)
    np.random.seed(42)
    s1 = priority.sample(frac=1).reset_index(drop=True)

    # Strategy 2: Highest PD first (most common naive approach)
    s2 = priority.sort_values("PD", ascending=False).reset_index(drop=True)

    # Strategy 3: Our policy — Highest Expected Loss Avoided first
    s3 = priority.sort_values("expected_loss", ascending=False).reset_index(drop=True)

    # ── 6. Simulate recovery for each strategy at daily capacity ─────────────
    banner("Simulating Recovery Outcomes  ⚠️ SIMULATION — no real call data")

    results = []
    for name, ranked in [("1. Call Everyone (Random)", s1),
                          ("2. Highest PD First",       s2),
                          ("3. Expected Loss Policy",   s3)]:
        r = simulate_recovery(ranked, DAILY_CAPACITY, COST_PER_CALL, CONTACT_EFF, name)
        results.append(r)
        print(f"\n  {name}")
        print(f"    EL in called set:         ₹{r['EL_called']:>12,.0f}")
        print(f"    Recovery (simulated):     ₹{r['recovery_simulated']:>12,.0f}  ⚠️ simulation")
        print(f"    Call cost:                ₹{r['call_cost']:>12,.0f}")
        print(f"    Net profit (simulated):   ₹{r['net_profit']:>12,.0f}  ⚠️ simulation")
        print(f"    Recovery per agent-hour:  ₹{r['recovery_per_hour']:>12,.0f}  ⚠️ simulation")
        print(f"    % of portfolio EL covered: {r['pct_EL_covered']:.1f}%")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT_DIR, "stage5_policy_metrics.csv"), index=False)

    # ── 7. Recovery curve (cumulative recovery as % of customers called) ──────
    banner("Computing Recovery Curves")

    max_calls = min(5000, len(priority))
    call_range = range(100, max_calls, 100)

    curves = {}
    for name, ranked in [("Random",          s1),
                          ("Highest PD",      s2),
                          ("EL Policy (Ours)", s3)]:
        recoveries = []
        for n in call_range:
            rec = ranked.head(n)["expected_loss"].sum() * CONTACT_EFF
            recoveries.append(rec)
        curves[name] = recoveries

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    banner("Generating Plots")

    call_list = list(call_range)

    # Plot 1: Recovery Curve
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"Random": "#7f7f7f", "Highest PD": "#1f77b4", "EL Policy (Ours)": "#d62728"}
    for name, vals in curves.items():
        ax.plot(call_list, [v / 1e6 for v in vals],
                linewidth=2.5, color=colors[name], label=name)
    ax.axvline(DAILY_CAPACITY, color="black", linestyle="--",
               label=f"Daily capacity = {DAILY_CAPACITY}")
    ax.set_xlabel("Number of Customers Called")
    ax.set_ylabel("Cumulative Simulated Recovery (₹ millions)")
    ax.set_title("Stage 5 — Recovery Curve by Strategy\n⚠️ Simulated — contact_eff = 30% assumption",
                 fontsize=12)
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:.1f}M"))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "stage5_recovery_curve.png"), dpi=150)
    plt.close()

    # Plot 2: Strategy comparison bar chart
    metrics_plot = ["recovery_simulated", "call_cost", "net_profit"]
    labels_plot  = ["Recovery (sim.)", "Call Cost", "Net Profit (sim.)"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    strategy_labels = [r["strategy"].split(". ")[-1] for r in results]
    bar_colors = ["#7f7f7f", "#1f77b4", "#d62728"]
    for ax, metric, label in zip(axes, metrics_plot, labels_plot):
        vals = [r[metric] for r in results]
        bars = ax.bar(strategy_labels, [v / 1e6 for v in vals], color=bar_colors)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("₹ millions")
        ax.set_xticklabels(strategy_labels, rotation=20, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:.2f}M"))
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals) * 0.01,
                    f"₹{val/1e6:.2f}M", ha="center", fontsize=8)
    fig.suptitle(f"Stage 5 — Strategy Comparison ({DAILY_CAPACITY} calls/day)\n"
                 "⚠️ Simulated recovery — not real call outcome data", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "stage5_strategy_comparison.png"), dpi=150)
    plt.close()

    # Plot 3: PD vs EAD scatter (coloured by priority score)
    fig, ax = plt.subplots(figsize=(9, 6))
    sample = priority.sample(n=min(5000, len(priority)), random_state=42)
    sc = ax.scatter(
        sample["PD"], sample["EAD"] / 1e5,
        c=sample["expected_loss"], cmap="RdYlGn_r",
        alpha=0.5, s=10
    )
    plt.colorbar(sc, ax=ax, label="Expected Loss (₹)")
    ax.set_xlabel("PD (Predicted Default Probability)")
    ax.set_ylabel("EAD (₹ lakhs)")
    ax.set_title("Stage 5 — PD vs. EAD: Why PD-only ranking misses high-EAD customers",
                 fontsize=11)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "stage5_pd_ead_scatter.png"), dpi=150)
    plt.close()

    print("  📊 Stage 5 plots saved.")

    # ── 9. Export priority list ────────────────────────────────────────────────
    priority_out = s3[["SK_ID_CURR", "PD", "EAD", "LGD", "expected_loss",
                        "recovery_if_called", "TARGET"]].copy()
    priority_out["priority_rank"] = range(1, len(priority_out) + 1)
    priority_out["contact_eff_assumption"] = CONTACT_EFF  # clearly labelled
    priority_out.to_csv(os.path.join(OUT_DIR, "stage5_priority_list.csv"), index=False)
    print(f"  Saved priority list: {len(priority_out):,} customers ranked.")

    # ── 10. Summary ────────────────────────────────────────────────────────────
    r_random = results[0]
    r_el     = results[2]
    uplift   = r_el["net_profit"] - r_random["net_profit"]

    banner("Stage 5 Complete ✅")
    print(f"""
  ── POLICY COMPARISON at {DAILY_CAPACITY} calls/day ─────────────────────────
  Assumptions:
    LGD                   = {LGD_ASSUMED:.0%}
    Contact effectiveness = {CONTACT_EFF:.0%}  ⚠️ SIMULATION ASSUMPTION
    Cost per call         = ₹{COST_PER_CALL}

  ┌───────────────────────────┬───────────────┬───────────────┐
  │ Strategy                  │ Recovery(sim) │ Net Profit(s) │
  ├───────────────────────────┼───────────────┼───────────────┤
  │ Call Everyone (Random)    │ ₹{r_random['recovery_simulated']:>11,.0f} │ ₹{r_random['net_profit']:>11,.0f} │
  │ Expected Loss Policy      │ ₹{r_el['recovery_simulated']:>11,.0f} │ ₹{r_el['net_profit']:>11,.0f} │
  └───────────────────────────┴───────────────┴───────────────┘

  EL Policy vs. Random: +₹{uplift:,.0f} net profit  ⚠️ simulated

  Business insight:
  Sorting by PD × EAD × LGD ensures agents spend time on accounts where
  the MONEY AT RISK is highest — not just the most likely defaulters.
  A customer with 40% PD and ₹5L outstanding is worth 5× more to call
  than a 40% PD customer with ₹1L outstanding.

  Next: Run stage6_excel/build_excel.py  (Manager Scenario Tool)
    """)


if __name__ == "__main__":
    main()
