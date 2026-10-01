-- =============================================================================
-- Collections Prioritization Project
-- Stage 1: SQL Feature Engineering (DuckDB)
-- =============================================================================
-- CONVENTIONS
--   • All window references use MONTHS_BALANCE (negative = past months from
--     application date; 0 = application month, excluded to prevent leakage)
--   • "last 3 months"  = MONTHS_BALANCE IN (-3, -2, -1)
--   • "last 6 months"  = MONTHS_BALANCE BETWEEN -6 AND -1
--   • "last 12 months" = MONTHS_BALANCE BETWEEN -12 AND -1
--   • OLS slope via DuckDB's REGR_SLOPE(y, x) = native, no Python needed
--   • All features aggregate to SK_ID_CURR (one row per customer)
-- =============================================================================


-- ---------------------------------------------------------------------------
-- STEP 1: Load raw CSVs as views (DuckDB reads directly from disk)
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW v_app AS
    SELECT * FROM read_csv_auto('data/raw/application_train.csv');

CREATE OR REPLACE VIEW v_ins AS
    SELECT * FROM read_csv_auto('data/raw/installments_payments.csv');

CREATE OR REPLACE VIEW v_cc AS
    SELECT * FROM read_csv_auto('data/raw/credit_card_balance.csv');

CREATE OR REPLACE VIEW v_bur AS
    SELECT * FROM read_csv_auto('data/raw/bureau.csv');

CREATE OR REPLACE VIEW v_burbal AS
    SELECT * FROM read_csv_auto('data/raw/bureau_balance.csv');


-- ---------------------------------------------------------------------------
-- STEP 2: INSTALLMENT FEATURES (Family A)
--
-- Business logic: We measure HOW LATE and HOW MUCH customers pay on their
-- term loans (home loans, personal loans, auto loans).
-- Positive DAYS_LATE = payment came after due date = bad.
-- Underpayment ratio > 0 = customer didn't pay the full EMI = stress.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE TABLE ins_features AS
WITH
-- Base: clean installment records, exclude future scheduled payments
ins_base AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        NUM_INSTALMENT_NUMBER,
        DAYS_INSTALMENT,
        DAYS_ENTRY_PAYMENT,
        AMT_INSTALMENT,
        AMT_PAYMENT,
        -- Days late: positive = late, negative = early, NULL = not yet paid
        (DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)          AS days_late,
        -- Underpay ratio: 0 = paid in full, 1 = paid nothing
        -- Cap at 0 below (early payers show negative; treat as 0)
        GREATEST(0, 1.0 - AMT_PAYMENT / NULLIF(AMT_INSTALMENT, 0))
                                                        AS underpay_ratio,
        -- Flag: completely missed payment
        CASE WHEN AMT_PAYMENT = 0 THEN 1 ELSE 0 END    AS is_missed
    FROM v_ins
    WHERE
        DAYS_INSTALMENT  <= 0   -- only past due dates (no future schedule)
        AND AMT_INSTALMENT IS NOT NULL
        AND AMT_INSTALMENT >= 0
),

-- Last-3-month window (DAYS_INSTALMENT >= -91 ≈ 3 months in days)
ins_l3m AS (
    SELECT
        SK_ID_CURR,
        AVG(days_late)       AS ins_days_late_l3m,
        AVG(underpay_ratio)  AS ins_underpay_ratio_l3m,
        SUM(is_missed)       AS ins_missed_count_l3m,
        COUNT(*)             AS ins_obs_l3m
    FROM ins_base
    WHERE DAYS_INSTALMENT >= -91
    GROUP BY SK_ID_CURR
),

-- Last-6-month window
ins_l6m AS (
    SELECT
        SK_ID_CURR,
        AVG(days_late)      AS ins_days_late_l6m,
        AVG(underpay_ratio) AS ins_underpay_ratio_l6m,
        COUNT(*)            AS ins_obs_l6m
    FROM ins_base
    WHERE DAYS_INSTALMENT >= -182
    GROUP BY SK_ID_CURR
),

-- All-time aggregates
ins_all AS (
    SELECT
        SK_ID_CURR,
        AVG(days_late)                              AS ins_days_late_mean,
        MAX(days_late)                              AS ins_days_late_max,
        AVG(underpay_ratio)                         AS ins_underpay_ratio_mean,
        AVG(is_missed)                              AS ins_pct_missed,
        -- % of payments that were late (days_late > 0)
        AVG(CASE WHEN days_late > 0 THEN 1.0 ELSE 0.0 END)
                                                    AS ins_pct_late,
        COUNT(*)                                    AS ins_total_obs,
        -- Slope of days_late over time: positive slope = getting worse
        REGR_SLOPE(days_late, DAYS_INSTALMENT)      AS ins_late_trend
    FROM ins_base
    GROUP BY SK_ID_CURR
)

SELECT
    a.SK_ID_CURR,
    -- All-time
    a.ins_days_late_mean,
    a.ins_days_late_max,
    a.ins_underpay_ratio_mean,
    a.ins_pct_missed,
    a.ins_pct_late,
    a.ins_total_obs,
    a.ins_late_trend,
    -- 3-month window
    l3.ins_days_late_l3m,
    l3.ins_underpay_ratio_l3m,
    l3.ins_missed_count_l3m,
    l3.ins_obs_l3m,
    -- 6-month window
    l6.ins_days_late_l6m,
    l6.ins_underpay_ratio_l6m,
    l6.ins_obs_l6m,
    -- Momentum: is recent behaviour worse than medium-term?
    -- Positive = deteriorating (key early-warning signal)
    COALESCE(l3.ins_days_late_l3m, 0) -
    COALESCE(l6.ins_days_late_l6m, 0) AS ins_late_momentum
FROM ins_all a
LEFT JOIN ins_l3m l3 USING (SK_ID_CURR)
LEFT JOIN ins_l6m l6 USING (SK_ID_CURR);


-- ---------------------------------------------------------------------------
-- STEP 3: CREDIT CARD FEATURES (Family B)
--
-- Business logic: Credit cards reveal real-time financial stress.
-- Rising utilization = customer is borrowing more and not repaying.
-- Paying only minimum = customer is revolving debt, not clearing it.
-- Cash advances on a credit card = last resort borrowing = red flag.
--
-- MONTHS_BALANCE: -1 = last month, -12 = 12 months ago, 0 = excluded
-- ---------------------------------------------------------------------------

CREATE OR REPLACE TABLE cc_features AS
WITH
-- Base: exclude current month (potential leakage)
cc_base AS (
    SELECT
        SK_ID_CURR,
        MONTHS_BALANCE,
        AMT_BALANCE,
        AMT_CREDIT_LIMIT_ACTUAL,
        AMT_PAYMENT_CURRENT,
        AMT_INST_MIN_REGULARITY,
        AMT_DRAWINGS_CURRENT,
        SK_DPD,
        -- Utilization: balance as % of limit
        CASE
            WHEN AMT_CREDIT_LIMIT_ACTUAL > 0
            THEN AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL
            ELSE NULL
        END AS utilization,
        -- Minimum payment ratio: < 1 = not paying minimum = stress
        CASE
            WHEN AMT_INST_MIN_REGULARITY > 0
            THEN AMT_PAYMENT_CURRENT / AMT_INST_MIN_REGULARITY
            ELSE NULL
        END AS min_pay_ratio
    FROM v_cc
    WHERE MONTHS_BALANCE < 0  -- strict past only
),

-- Last 3 months
cc_l3m AS (
    SELECT
        SK_ID_CURR,
        AVG(utilization)        AS cc_util_l3m,
        AVG(AMT_BALANCE)        AS cc_balance_l3m,
        AVG(SK_DPD)             AS cc_dpd_l3m,
        AVG(min_pay_ratio)      AS cc_min_pay_ratio_l3m,
        COUNT(*)                AS cc_obs_l3m
    FROM cc_base
    WHERE MONTHS_BALANCE >= -3
    GROUP BY SK_ID_CURR
),

-- All-time + trends (slope computed on all history)
cc_all AS (
    SELECT
        SK_ID_CURR,
        AVG(utilization)        AS cc_util_mean,
        MAX(utilization)        AS cc_util_max,
        AVG(AMT_BALANCE)        AS cc_balance_mean,
        MAX(SK_DPD)             AS cc_dpd_max,
        AVG(SK_DPD)             AS cc_dpd_mean,
        AVG(min_pay_ratio)      AS cc_min_pay_ratio_mean,
        -- OLS slope of utilization over months: positive = rising util = stress
        REGR_SLOPE(utilization, MONTHS_BALANCE)     AS cc_util_trend,
        -- OLS slope of balance over months
        REGR_SLOPE(AMT_BALANCE, MONTHS_BALANCE)     AS cc_balance_trend,
        -- OLS slope of cash drawings (escalating draws = distress signal)
        REGR_SLOPE(AMT_DRAWINGS_CURRENT, MONTHS_BALANCE) AS cc_drawings_trend,
        COUNT(*)                AS cc_total_obs
    FROM cc_base
    GROUP BY SK_ID_CURR
)

SELECT
    a.SK_ID_CURR,
    -- All-time
    a.cc_util_mean,
    a.cc_util_max,
    a.cc_balance_mean,
    a.cc_dpd_max,
    a.cc_dpd_mean,
    a.cc_min_pay_ratio_mean,
    a.cc_util_trend,
    a.cc_balance_trend,
    a.cc_drawings_trend,
    a.cc_total_obs,
    -- 3-month window
    l3.cc_util_l3m,
    l3.cc_balance_l3m,
    l3.cc_dpd_l3m,
    l3.cc_min_pay_ratio_l3m,
    l3.cc_obs_l3m
FROM cc_all a
LEFT JOIN cc_l3m l3 USING (SK_ID_CURR);


-- ---------------------------------------------------------------------------
-- STEP 4: BUREAU FEATURES (Family C)
--
-- Business logic: Bureau data = the customer's entire credit history across
-- ALL lenders, not just this bank. Think of it as the CIBIL report.
-- bureau.csv = one row per credit account (e.g., 3 loans = 3 rows)
-- bureau_balance.csv = monthly DPD status for each bureau account
--
-- STATUS codes in bureau_balance:
--   0=no DPD, 1=1-30 DPD, 2=31-60, 3=61-90, 4=91-120, 5=120+,
--   C=Closed, X=Unknown
-- ---------------------------------------------------------------------------

CREATE OR REPLACE TABLE bur_features AS
WITH
-- Convert STATUS to numeric DPD bucket
burbal_num AS (
    SELECT
        bb.SK_ID_BUREAU,
        b.SK_ID_CURR,
        bb.MONTHS_BALANCE,
        bb.STATUS,
        CASE bb.STATUS
            WHEN '0' THEN 0
            WHEN '1' THEN 1
            WHEN '2' THEN 2
            WHEN '3' THEN 3
            WHEN '4' THEN 4
            WHEN '5' THEN 5
            WHEN 'C' THEN 0   -- closed = current
            WHEN 'X' THEN NULL -- unknown
        END AS dpd_bucket
    FROM v_burbal bb
    INNER JOIN v_bur b USING (SK_ID_BUREAU)
    WHERE bb.MONTHS_BALANCE < 0  -- strict past only
),

-- Bureau balance aggregates per customer
burbal_agg AS (
    SELECT
        SK_ID_CURR,
        -- Rate of delinquent months (any bucket >= 1)
        AVG(CASE WHEN dpd_bucket >= 1 THEN 1.0 ELSE 0.0 END)
                                    AS bur_delinq_rate,
        -- Same, last 12 months
        AVG(CASE
                WHEN MONTHS_BALANCE >= -12 AND dpd_bucket >= 1 THEN 1.0
                WHEN MONTHS_BALANCE >= -12 AND dpd_bucket IS NOT NULL THEN 0.0
                ELSE NULL
            END)                    AS bur_delinq_rate_12m,
        MAX(dpd_bucket)             AS bur_worst_status_ever,
        -- Slope: is status getting worse over time?
        REGR_SLOPE(dpd_bucket, MONTHS_BALANCE) AS bur_status_trend
    FROM burbal_num
    GROUP BY SK_ID_CURR
),

-- Bureau account-level aggregates
bur_agg AS (
    SELECT
        SK_ID_CURR,
        COUNT(*)                        AS bur_total_accounts,
        SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END)
                                        AS bur_active_accounts,
        AVG(CREDIT_DAY_OVERDUE)         AS bur_days_overdue_mean,
        MAX(CREDIT_DAY_OVERDUE)         AS bur_days_overdue_max,
        SUM(AMT_CREDIT_SUM_OVERDUE)     AS bur_amt_overdue_sum,
        -- Bureau leverage ratio: total debt / total credit
        CASE
            WHEN SUM(AMT_CREDIT_SUM) > 0
            THEN SUM(AMT_CREDIT_SUM_DEBT) / SUM(AMT_CREDIT_SUM)
            ELSE NULL
        END                             AS bur_debt_credit_ratio,
        -- How recently has the customer been seeking credit (negative = past)
        MIN(DAYS_CREDIT)                AS bur_days_credit_min,
        AVG(DAYS_CREDIT)                AS bur_days_credit_mean
    FROM v_bur
    GROUP BY SK_ID_CURR
)

SELECT
    a.SK_ID_CURR,
    -- Account-level
    a.bur_total_accounts,
    a.bur_active_accounts,
    a.bur_days_overdue_mean,
    a.bur_days_overdue_max,
    a.bur_amt_overdue_sum,
    a.bur_debt_credit_ratio,
    a.bur_days_credit_min,
    a.bur_days_credit_mean,
    -- Monthly status
    b.bur_delinq_rate,
    b.bur_delinq_rate_12m,
    b.bur_worst_status_ever,
    b.bur_status_trend
FROM bur_agg a
LEFT JOIN burbal_agg b USING (SK_ID_CURR);


-- ---------------------------------------------------------------------------
-- STEP 5: FINAL FEATURE STORE (Join everything to application spine)
--
-- Business logic: One clean row per customer. Missing values appear where
-- a customer has no bureau history or no credit card. We keep NULLs here
-- and handle imputation in the modelling stage (Stage 2).
-- We also add 2 cross-product stress features here.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE TABLE feature_store AS
SELECT
    -- ── Identifiers & Label ──────────────────────────────────────────────
    app.SK_ID_CURR,
    app.TARGET,                     -- 1 = defaulted, 0 = repaid (our Y)

    -- ── Application-level demographics (for segmentation only, not models)
    app.AMT_CREDIT              AS app_loan_amount,
    app.AMT_ANNUITY             AS app_emi,
    app.AMT_INCOME_TOTAL        AS app_income,
    app.DAYS_BIRTH              AS app_age_days,
    app.DAYS_EMPLOYED           AS app_days_employed,
    app.CODE_GENDER,
    app.NAME_CONTRACT_TYPE,

    -- ── Family A: Installment Features ───────────────────────────────────
    ins.ins_days_late_mean,
    ins.ins_days_late_max,
    ins.ins_underpay_ratio_mean,
    ins.ins_pct_missed,
    ins.ins_pct_late,
    ins.ins_late_trend,
    ins.ins_days_late_l3m,
    ins.ins_underpay_ratio_l3m,
    ins.ins_missed_count_l3m,
    ins.ins_days_late_l6m,
    ins.ins_underpay_ratio_l6m,
    ins.ins_late_momentum,          -- KEY: positive = getting worse recently

    -- ── Family B: Credit Card Features ───────────────────────────────────
    cc.cc_util_mean,
    cc.cc_util_max,
    cc.cc_util_l3m,
    cc.cc_util_trend,               -- KEY: positive slope = rising utilization
    cc.cc_balance_mean,
    cc.cc_balance_l3m,
    cc.cc_balance_trend,
    cc.cc_dpd_mean,
    cc.cc_dpd_max,
    cc.cc_dpd_l3m,
    cc.cc_min_pay_ratio_mean,
    cc.cc_min_pay_ratio_l3m,
    cc.cc_drawings_trend,

    -- ── Family C: Bureau Features ─────────────────────────────────────────
    bur.bur_total_accounts,
    bur.bur_active_accounts,
    bur.bur_days_overdue_mean,
    bur.bur_days_overdue_max,
    bur.bur_amt_overdue_sum,
    bur.bur_debt_credit_ratio,
    bur.bur_days_credit_min,
    bur.bur_delinq_rate,
    bur.bur_delinq_rate_12m,
    bur.bur_worst_status_ever,
    bur.bur_status_trend,

    -- ── Family D: Cross-Product Stress Signals ────────────────────────────
    -- Composite stress: high when customer is simultaneously:
    --   underpaying EMI + maxing card + bureau delinquent
    -- ⚠️ SIMULATION NOTE: This is a heuristic index, not a validated score.
    COALESCE(ins.ins_underpay_ratio_l3m, 0) *
    COALESCE(cc.cc_util_l3m, 0) *
    COALESCE(bur.bur_delinq_rate_12m, 0)    AS stress_composite,

    -- Count how many product types show DPD > 0 in last 3 months
    (CASE WHEN COALESCE(ins.ins_days_late_l3m, 0) > 0 THEN 1 ELSE 0 END +
     CASE WHEN COALESCE(cc.cc_dpd_l3m, 0)          > 0 THEN 1 ELSE 0 END +
     CASE WHEN COALESCE(bur.bur_delinq_rate_12m, 0) > 0 THEN 1 ELSE 0 END)
                                            AS n_products_delinquent,

    -- ── Family E: Recency / Momentum ──────────────────────────────────────
    -- Momentum: positive = 3m worse than 6m (deteriorating)
    ins.ins_late_momentum,
    -- Bureau getting worse: positive slope on dpd_bucket = deteriorating
    bur.bur_status_trend            AS bur_momentum

FROM v_app app
LEFT JOIN ins_features ins USING (SK_ID_CURR)
LEFT JOIN cc_features  cc  USING (SK_ID_CURR)
LEFT JOIN bur_features bur USING (SK_ID_CURR)
WHERE app.TARGET IS NOT NULL;       -- keep labelled records only


-- ---------------------------------------------------------------------------
-- STEP 6: QUICK VALIDATION QUERIES
-- Run these to sanity-check the feature store before modelling
-- ---------------------------------------------------------------------------

-- 6a: Row count and default rate
SELECT
    COUNT(*)                                AS total_customers,
    SUM(TARGET)                             AS total_defaults,
    ROUND(100.0 * AVG(TARGET), 2)           AS default_rate_pct
FROM feature_store;

-- 6b: Feature completeness (% non-null per column)
SELECT
    COUNT(*)                                        AS n,
    ROUND(100.0 * COUNT(ins_days_late_mean)  / COUNT(*), 1) AS pct_has_installments,
    ROUND(100.0 * COUNT(cc_util_mean)        / COUNT(*), 1) AS pct_has_credit_card,
    ROUND(100.0 * COUNT(bur_total_accounts)  / COUNT(*), 1) AS pct_has_bureau
FROM feature_store;

-- 6c: Default rate by number of delinquent products (should increase monotonically)
SELECT
    n_products_delinquent,
    COUNT(*)                            AS customers,
    ROUND(100.0 * AVG(TARGET), 2)       AS default_rate_pct
FROM feature_store
GROUP BY n_products_delinquent
ORDER BY n_products_delinquent;

-- 6d: Average feature values for defaulters vs. non-defaulters
SELECT
    TARGET,
    ROUND(AVG(ins_days_late_mean),    2) AS avg_days_late,
    ROUND(AVG(ins_underpay_ratio_mean),3) AS avg_underpay,
    ROUND(AVG(cc_util_mean),           3) AS avg_cc_util,
    ROUND(AVG(bur_delinq_rate_12m),    3) AS avg_bur_delinq,
    ROUND(AVG(stress_composite),       4) AS avg_stress
FROM feature_store
GROUP BY TARGET
ORDER BY TARGET;
