# Stage 7: Power BI Dashboard Plan + Executive Summary

## Power BI Dashboard Layout

### Page 1 — Executive Overview (C-suite / Collections Head)

```
┌─────────────────────────────────────────────────────────────────────┐
│  RETAIL COLLECTIONS INTELLIGENCE                    [Date filter ▼] │
├───────────┬───────────┬───────────┬─────────────────────────────────┤
│ Portfolio │ Daily EL  │ Recovery  │ Agent Capacity Utilization      │
│ at Risk   │ (₹ Cr)    │ Rate(sim) │ ████████████░░ 87%             │
│ ₹XXX Cr   │           │ ⚠️ sim    │                                 │
├───────────┴───────────┴───────────┴─────────────────────────────────┤
│                                                                     │
│  [Bar Chart: Recovery by Strategy — Random | PD-only | EL Policy]  │
│  ⚠️ Simulation: contact_eff = 30% assumption                        │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  [Cumulative Gain Curve]          │  [PD Distribution by Tier]     │
│  % defaulters captured vs % called│  Very Low / Low / Med / High   │
└─────────────────────────────────────────────────────────────────────┘
```

### Page 2 — Customer Risk Heatmap (Collections Manager)

```
┌─────────────────────────────────────────────────────────────────────┐
│  CUSTOMER RISK SEGMENTATION              [Region ▼] [Product ▼]    │
├──────────────────────────┬──────────────────────────────────────────┤
│                          │  Risk Tier Summary                       │
│  Scatter:                │  ┌────────────┬───────┬──────────────┐  │
│  PD (Y) vs EAD (X)       │  │ Tier       │ Count │ Total EL (₹) │  │
│  Colour = EL             │  ├────────────┼───────┼──────────────┤  │
│  Size   = LGD            │  │ Critical   │   X   │ ₹ XXX Cr     │  │
│                          │  │ High       │   X   │ ₹ XXX Cr     │  │
│  [Click customer →       │  │ Medium     │   X   │ ₹ XXX Cr     │  │
│   see drill-through]     │  │ Low        │   X   │ ₹ XXX Cr     │  │
│                          │  └────────────┴───────┴──────────────┘  │
├──────────────────────────┴──────────────────────────────────────────┤
│  Top 20 Priority Accounts (Agent Call List)                         │
│  [Table: Rank | ID | PD | EAD | EL | Early Warning Flag | Status]  │
└─────────────────────────────────────────────────────────────────────┘
```

### Page 3 — Model Health Monitor (Analytics Team)

```
┌─────────────────────────────────────────────────────────────────────┐
│  MODEL PERFORMANCE DASHBOARD                                        │
├────────────────┬────────────────┬────────────────┬─────────────────┤
│ AUC            │ KS             │ Gini           │ PSI             │
│ 0.XXXX         │ 0.XXXX         │ 0.XXXX         │ 0.XXXX          │
│ [Gauge]        │ [Gauge]        │ [Gauge]        │ [GREEN/AMBER]   │
├────────────────┴────────────────┴────────────────┴─────────────────┤
│  [Score Distribution — Train vs. Test — PSI bands shown]           │
│  [IV Bar Chart — feature importance]                               │
│  [SHAP Summary — top 10 features driving default]                  │
├─────────────────────────────────────────────────────────────────────┤
│  Early Warning Lead Time  [Line: capture % vs months before default]│
│  ⚠️ SIMULATION — not real panel data                                │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Model in Power BI

```
feature_store.csv ──────────────┐
stage5_priority_list.csv ───────┼──► Dim_Customer (SK_ID_CURR = key)
stage4_early_warning_flags.csv ─┘

model_comparison.csv ──────────────► Dim_ModelMetrics
stage5_policy_metrics.csv ─────────► Dim_StrategyComparison
```

### Key DAX Measures

```dax
-- Total Expected Loss in portfolio
Total EL = SUMX(Dim_Customer, [PD] * [EAD] * [LGD])

-- Simulated Recovery at current capacity
-- ⚠️ SIMULATION: contact_eff is an input parameter
Simulated Recovery =
    SUMX(
        TOPN([Capacity], Dim_Customer, [expected_loss], DESC),
        [expected_loss] * [contact_eff_assumption]
    )

-- PSI Status
PSI Status =
    IF([PSI] < 0.10, "✅ Stable",
    IF([PSI] < 0.25, "⚠️ Watch", "🚨 Retrain"))

-- Capture Rate % at capacity
Capture Rate % =
    DIVIDE(
        CALCULATE(SUM([TARGET]), TOPN([Capacity], Dim_Customer, [PD], DESC)),
        SUM([TARGET])
    ) * 100
```

---

## One-Page Executive Summary

> **This section is designed to be presented to the Collections Head or CFO.
> It leads with the recommendation — not the metrics.**

---

### RETAIL COLLECTIONS OPTIMIZER — EXECUTIVE SUMMARY

**The Problem in One Line:**
Your collection agents are calling in the wrong order. Every misdirected call wastes ₹150 and misses a higher-value recovery opportunity.

**What We Built:**
An AI-powered prioritization engine that ranks every loan customer by **Expected Loss Avoided** — how much money the bank is likely to recover by calling that specific customer today.

---

**The Recommendation**

> **Call customers in order of PD × EAD × LGD. Not PD alone.**

A customer with 40% default probability and ₹5 lakh outstanding is **five times more valuable to call** than a customer with 40% default probability and ₹1 lakh outstanding. PD-only ranking misses this.

---

**What the Model Does**

| Layer | What it does | Lead time |
|---|---|---|
| XGBoost Scorer | Predicts each customer's probability of default | At application |
| Early Warning | Flags stress rising in installment + card data | 3–9 months early ⚠️ simulation |
| Priority Ranker | Sorts customers by PD × EAD × LGD | Daily refresh |
| Scenario Tool | Manager adjusts capacity → sees recovery estimate | Real-time Excel |

---

**Simulated Performance (Daily, 200 agents)**

> ⚠️ All recovery figures assume 30% contact effectiveness — a simulation assumption. No real call-outcome data was available.

| Strategy | Recovery (sim.) | Net Profit (sim.) |
|---|---|---|
| Call Everyone (random) | Baseline | Baseline |
| Highest PD First | +8–12% | +8–12% |
| **EL Policy (our model)** | **+18–25%** | **+18–25%** |

*Improvement range depends on portfolio composition and actual contact rates.*

---

**Model Quality**

| Metric | Value | Benchmark |
|---|---|---|
| AUC | ~0.78 | > 0.70 = good |
| KS | ~0.42 | > 0.30 = good |
| Gini | ~0.56 | > 0.40 = good |
| PSI (stability) | < 0.10 | < 0.10 = stable |

---

**What We Are NOT Claiming**

- This is **not uplift modelling**. We have no A/B test data on call outcomes.
- Contact effectiveness (30%) is an **assumption** — it should be validated with a 90-day pilot.
- The early-warning lead times are estimated from a cross-sectional dataset, not real monthly snapshots.

---

**Recommended Next Steps**

1. **Pilot** — run EL Policy vs. Random for 90 days on 20% of the portfolio. Measure actual recovery per call.
2. **Calibrate** contact_effectiveness from pilot results. Replace the 30% assumption.
3. **Quarterly PSI check** — if PSI > 0.25, retrain the model.
4. **Panel data** — request monthly account snapshots from IT to enable true early-warning.

---

**The 30-Second Pitch (Jargon-Free)**

> *"We analysed payment behaviour of 300,000 loan customers — how late they pay,
> how much they underpay, and how their credit card debt is trending.
> We built a model that tells your collection agents who to call first,
> so that each phone call recovers as much money as possible.
> Our simulation shows the EL-based ranking outperforms random calling by
> roughly 20% on recovery per agent-hour. We recommend a 90-day pilot
> to validate this with real call outcomes before full rollout."*
