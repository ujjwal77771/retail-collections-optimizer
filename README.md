# Retail Collections Optimizer

> End-to-end collections prioritization engine for a retail loan portfolio  
> SQL feature store · WOE scorecard · XGBoost · PD × EAD × LGD policy · Excel scenario tool · Power BI dashboard  
> Built on the **[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)** dataset

---

## 🎯 Business Problem

A bank has **limited collection agents**. Who do you call first?

Calling everyone equally wastes agent-hours on customers who would have paid anyway.  
Calling only high-PD customers ignores exposure size.  
**This project builds a prioritization engine** that ranks each customer by **Expected Loss Avoided** — so every agent-hour generates maximum recovery.

---

## 🗂️ Project Structure

```
axis-biu-collections/
│
├── data/raw/                     ← Kaggle CSVs go here (not tracked in git)
│
├── stage1_sql/
│   ├── feature_engineering.sql   ← DuckDB: installment + card + bureau features
│   └── run_stage1.py             ← Runner: builds feature_store, validation charts
│
├── stage2_scorecard/
│   └── run_stage2.py             ← WOE/IV binning, logistic regression scorecard
│
├── stage3_xgboost/
│   └── run_stage3.py             ← XGBoost challenger, SHAP, model comparison
│
├── stage4_early_warning/
│   └── run_stage4.py             ← Behavioural stress flags, lead-time analysis
│
├── stage5_policy/
│   └── run_stage5.py             ← PD × EAD × LGD prioritization policy
│
├── stage6_excel/
│   └── scenario_model.xlsx       ← Manager-facing capacity & profit scenario tool
│
├── outputs/                      ← Charts, CSVs, model artifacts (auto-generated)
│
├── download_data.py              ← Data file checker + Kaggle download instructions
├── requirements.txt              ← Python dependencies
└── README.md
```

---

## 📊 Data Source

**[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)** — Kaggle competition dataset

| File | Rows | Description |
|---|---|---|
| `application_train.csv` | 307,511 | Master spine — one row per applicant |
| `installments_payments.csv` | 13.6M | EMI schedule vs. actual payment |
| `credit_card_balance.csv` | 3.8M | Monthly card balance & utilization |
| `bureau.csv` | 1.7M | CIBIL-equivalent bureau accounts |
| `bureau_balance.csv` | 27.3M | Monthly bureau DPD status |

> ⚠️ **Data is not tracked in git.** Download from Kaggle and place in `data/raw/`. See [`download_data.py`](download_data.py).

---

## 🚀 Quickstart

```bash
# 1. Clone
git clone https://github.com/ujjwal77771/retail-collections-optimizer.git
cd retail-collections-optimizer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data (Kaggle account needed)
kaggle competitions download -c home-credit-default-risk -p data/raw/ --unzip

# 4. Verify files
python download_data.py

# 5. Run Stage 1 (SQL feature engineering)
python stage1_sql/run_stage1.py
```

---

## 📐 Seven-Stage Architecture

```
Stage 1  →  SQL Feature Store (DuckDB)
                ↓
Stage 2  →  Baseline Scorecard (WOE/IV + Logistic Regression)
                ↓
Stage 3  →  Challenger Model (XGBoost + SHAP)
                ↓
Stage 4  →  Early-Warning Layer (behavioural stress flags)
                ↓
Stage 5  →  Prioritization Policy (PD × EAD × LGD ranking)
                ↓
Stage 6  →  Excel Scenario Tool (manager-facing)
                ↓
Stage 7  →  Power BI Dashboard + Executive Summary
```

---

## 🔑 Key Features Engineered (Stage 1)

**Family A — Installment Payment Behaviour**
- `ins_days_late_mean` · `ins_days_late_l3m` · `ins_days_late_l6m`
- `ins_underpay_ratio_l3m` · `ins_pct_missed` · `ins_late_momentum`

**Family B — Credit Card Stress**
- `cc_util_l3m` · `cc_util_trend` (OLS slope)
- `cc_balance_trend` · `cc_dpd_max` · `cc_min_pay_ratio_l3m`

**Family C — Bureau (CIBIL) History**
- `bur_delinq_rate_12m` · `bur_worst_status_ever` · `bur_status_trend`
- `bur_debt_credit_ratio` · `bur_amt_overdue_sum`

**Family D — Cross-Product Stress**
- `stress_composite` = underpay × utilization × bureau delinquency
- `n_products_delinquent` — breadth of stress across products

---

## 📈 Model Performance (Targets)

| Metric | Baseline (LR Scorecard) | Challenger (XGBoost) |
|---|---|---|
| AUC | ~0.73 | ~0.78 |
| KS | ~0.35 | ~0.42 |
| Gini | ~0.46 | ~0.56 |

> Results populated after Stage 2 & 3 runs.

---

## ⚠️ Honest Disclaimers

- **No real call-outcome data exists.** Contact effectiveness in Stage 5 is a simulation assumption, clearly labeled.
- **Uplift / treatment effect** is NOT claimed unless a simulated A/B structure is added explicitly.
- All leakage controls are documented inline in the SQL — future months are strictly excluded.

---

## 🏦 Business Context

**Role target:** Banking Business Intelligence Unit (BIU) — Analytics / Data Science  
**JD alignment:** Predictive scoring · Portfolio analysis · Profitability modelling · Dashboards · Storytelling

---

## 📝 Author

**Ujjwal** · [GitHub](https://github.com/ujjwal77771)  
Built as a flagship interview project — feedback welcome via Issues.
