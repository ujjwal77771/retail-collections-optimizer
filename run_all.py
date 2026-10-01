"""
=============================================================================
Retail Collections Optimizer — Master Runner
=============================================================================
Runs all stages in sequence. Stop at any stage by Ctrl+C.

Usage:
    python run_all.py              # all stages
    python run_all.py --stage 1    # only stage 1
    python run_all.py --from 2     # from stage 2 onwards
=============================================================================
"""

import subprocess, sys, os, time, argparse

ROOT = os.path.dirname(os.path.abspath(__file__))

STAGES = [
    (1, "Stage 1 — SQL Feature Engineering",   "stage1_sql/run_stage1.py"),
    (2, "Stage 2 — WOE/IV Baseline Scorecard", "stage2_scorecard/run_stage2.py"),
    (3, "Stage 3 — XGBoost Challenger + SHAP", "stage3_xgboost/run_stage3.py"),
    (4, "Stage 4 — Early Warning Layer",        "stage4_early_warning/run_stage4.py"),
    (5, "Stage 5 — Prioritization Policy",      "stage5_policy/run_stage5.py"),
    (6, "Stage 6 — Excel Scenario Tool",        "stage6_excel/build_excel.py"),
]


def run_stage(num, label, script):
    print(f"\n{'='*65}")
    print(f"  RUNNING: {label}")
    print(f"{'='*65}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, script)],
        cwd=ROOT
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n❌  Stage {num} failed after {elapsed:.0f}s. Fix errors above then re-run.")
        sys.exit(result.returncode)
    print(f"\n✅  Stage {num} done in {elapsed:.0f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=None, help="Run only this stage")
    parser.add_argument("--from",  type=int, default=1,    dest="from_stage",
                        help="Run from this stage onwards")
    args = parser.parse_args()

    stages_to_run = STAGES
    if args.stage:
        stages_to_run = [s for s in STAGES if s[0] == args.stage]
    else:
        stages_to_run = [s for s in STAGES if s[0] >= args.from_stage]

    print("\n" + "="*65)
    print("  RETAIL COLLECTIONS OPTIMIZER — MASTER RUNNER")
    print("="*65)
    print(f"\n  Stages to run: {[s[0] for s in stages_to_run]}")
    print("\n  Pre-requisite: data/raw/ must contain Kaggle CSVs.")
    print("  Run: python download_data.py  to verify.\n")

    for num, label, script in stages_to_run:
        run_stage(num, label, script)

    print(f"\n{'='*65}")
    print("  ALL STAGES COMPLETE ✅")
    print("="*65)
    print(f"""
  Outputs in outputs/:
    feature_store.csv              — 31 features per customer
    iv_table.csv                   — WOE/IV for each feature
    scorecard_table.csv            — scorecard points
    scorecard_model.pkl            — logistic regression pipeline
    xgb_model.pkl                  — XGBoost model
    model_comparison.csv           — baseline vs challenger metrics
    stage4_early_warning_flags.csv — per-customer risk flag
    stage5_priority_list.csv       — customers ranked by EL policy
    stage5_policy_metrics.csv      — strategy comparison table
    collections_scenario_tool.xlsx — manager Excel tool
    stage2_*.png / stage3_*.png / stage4_*.png / stage5_*.png

  Stage 7 (Power BI + Executive Summary):
    See stage7_dashboard/POWER_BI_PLAN.md
    """)


if __name__ == "__main__":
    main()
