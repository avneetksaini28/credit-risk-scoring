-- =============================================================================
-- 02_analysis_queries.sql
-- Business analysis of the Home Credit portfolio in SQLite.
-- Each query starts with "-- name:" so notebook 03 can run them one by one
-- (credit_risk.database.load_named_queries). Techniques used: joins, GROUP BY
-- aggregations, CASE bucketing, CTEs, and window functions (SUM/AVG OVER, RANK,
-- NTILE, ROW_NUMBER, LAG, running totals).
-- =============================================================================


-- name: table_row_counts
-- question: How big is each table? (data-volume sanity check)
SELECT 'application_train' AS table_name, COUNT(*) AS row_count FROM application_train
UNION ALL SELECT 'bureau', COUNT(*) FROM bureau
UNION ALL SELECT 'previous_application', COUNT(*) FROM previous_application
UNION ALL SELECT 'installments_payments', COUNT(*) FROM installments_payments
UNION ALL SELECT 'pos_cash_balance', COUNT(*) FROM pos_cash_balance
UNION ALL SELECT 'credit_card_balance', COUNT(*) FROM credit_card_balance;


-- name: portfolio_default_rate
-- question: What is the overall default rate and the exposure at risk?
SELECT
    COUNT(*)                                       AS applicants,
    SUM(TARGET)                                    AS defaults,
    ROUND(100.0 * AVG(TARGET), 2)                  AS default_rate_pct,
    ROUND(SUM(AMT_CREDIT) / 1e6, 1)                AS total_credit_mn,
    ROUND(SUM(CASE WHEN TARGET = 1 THEN AMT_CREDIT ELSE 0 END) / 1e6, 1) AS credit_of_defaulters_mn,
    ROUND(100.0 * SUM(CASE WHEN TARGET = 1 THEN AMT_CREDIT ELSE 0 END) / SUM(AMT_CREDIT), 2) AS exposure_default_pct
FROM application_train;


-- name: default_by_contract_type
-- question: Do cash loans and revolving loans carry different risk?
SELECT
    NAME_CONTRACT_TYPE,
    COUNT(*)                                                   AS applicants,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)          AS portfolio_share_pct,
    ROUND(100.0 * AVG(TARGET), 2)                              AS default_rate_pct,
    ROUND(AVG(AMT_CREDIT), 0)                                  AS avg_credit
FROM application_train
GROUP BY NAME_CONTRACT_TYPE
ORDER BY default_rate_pct DESC;


-- name: default_by_income_type
-- question: Which income types are riskiest, and how much of the book do they represent?
SELECT
    NAME_INCOME_TYPE,
    COUNT(*)                                                   AS applicants,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)          AS portfolio_share_pct,
    ROUND(100.0 * AVG(TARGET), 2)                              AS default_rate_pct,
    ROUND(100.0 * AVG(TARGET) - 100.0 * (SELECT AVG(TARGET) FROM application_train), 2) AS vs_portfolio_pp
FROM application_train
GROUP BY NAME_INCOME_TYPE
HAVING COUNT(*) >= 20
ORDER BY default_rate_pct DESC;


-- name: default_by_education_ranked
-- question: Rank education levels by default rate (window function RANK).
SELECT
    NAME_EDUCATION_TYPE,
    COUNT(*)                                                   AS applicants,
    ROUND(100.0 * AVG(TARGET), 2)                              AS default_rate_pct,
    RANK() OVER (ORDER BY AVG(TARGET) DESC)                    AS risk_rank
FROM application_train
GROUP BY NAME_EDUCATION_TYPE
ORDER BY risk_rank;


-- name: default_by_age_band
-- question: How does default risk change over the life cycle?
SELECT
    age_band,
    COUNT(*)                                                   AS applicants,
    ROUND(100.0 * AVG(TARGET), 2)                              AS default_rate_pct,
    ROUND(AVG(AMT_INCOME_TOTAL), 0)                            AS avg_income,
    ROUND(AVG(credit_to_income), 2)                            AS avg_credit_to_income
FROM v_applications
GROUP BY age_band
ORDER BY age_band;


-- name: default_by_income_quintile
-- question: Is income alone a good risk separator? (NTILE quintiles)
WITH ranked AS (
    SELECT TARGET, AMT_INCOME_TOTAL,
           NTILE(5) OVER (ORDER BY AMT_INCOME_TOTAL) AS income_quintile
    FROM application_train
)
SELECT
    income_quintile,
    MIN(AMT_INCOME_TOTAL)                                      AS min_income,
    MAX(AMT_INCOME_TOTAL)                                      AS max_income,
    COUNT(*)                                                   AS applicants,
    ROUND(100.0 * AVG(TARGET), 2)                              AS default_rate_pct
FROM ranked
GROUP BY income_quintile
ORDER BY income_quintile;


-- name: default_by_affordability
-- question: Does a higher repayment burden (annuity / income) mean more defaults?
SELECT
    CASE
        WHEN annuity_to_income < 0.10 THEN '1. <10%'
        WHEN annuity_to_income < 0.15 THEN '2. 10-15%'
        WHEN annuity_to_income < 0.20 THEN '3. 15-20%'
        WHEN annuity_to_income < 0.30 THEN '4. 20-30%'
        WHEN annuity_to_income IS NOT NULL THEN '5. 30%+'
        ELSE '6. unknown'
    END                                                        AS payment_to_income_band,
    COUNT(*)                                                   AS applicants,
    ROUND(100.0 * AVG(TARGET), 2)                              AS default_rate_pct
FROM v_applications
GROUP BY payment_to_income_band
ORDER BY payment_to_income_band;


-- name: ext_source_decile_gains
-- question: How well does a single external score rank risk? Cumulative share of defaults captured (running SUM window).
WITH scored AS (
    SELECT TARGET, EXT_SOURCE_2,
           NTILE(10) OVER (ORDER BY EXT_SOURCE_2) AS decile          -- decile 1 = lowest score = riskiest
    FROM application_train
    WHERE EXT_SOURCE_2 IS NOT NULL
),
by_decile AS (
    SELECT decile, COUNT(*) AS applicants, SUM(TARGET) AS defaults,
           ROUND(MIN(EXT_SOURCE_2), 3) AS min_score, ROUND(MAX(EXT_SOURCE_2), 3) AS max_score
    FROM scored GROUP BY decile
)
SELECT
    decile, min_score, max_score, applicants,
    ROUND(100.0 * defaults / applicants, 2)                                         AS default_rate_pct,
    ROUND(100.0 * SUM(defaults) OVER (ORDER BY decile) / SUM(defaults) OVER (), 1)  AS cum_defaults_captured_pct
FROM by_decile
ORDER BY decile;


-- name: bureau_active_loans_vs_default
-- question: Do clients with more active loans at other lenders default more? (CTE + LEFT JOIN)
WITH b AS (
    SELECT SK_ID_CURR,
           SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END) AS active_loans
    FROM bureau
    GROUP BY SK_ID_CURR
)
SELECT
    CASE
        WHEN b.SK_ID_CURR IS NULL THEN '0. no bureau record'
        WHEN b.active_loans = 0  THEN '1. none active'
        WHEN b.active_loans <= 2 THEN '2. 1-2 active'
        WHEN b.active_loans <= 4 THEN '3. 3-4 active'
        ELSE '4. 5+ active'
    END                                                        AS active_bureau_loans,
    COUNT(*)                                                   AS applicants,
    ROUND(100.0 * AVG(a.TARGET), 2)                            AS default_rate_pct
FROM application_train a
LEFT JOIN b ON a.SK_ID_CURR = b.SK_ID_CURR
GROUP BY active_bureau_loans
ORDER BY active_bureau_loans;


-- name: bureau_debt_utilisation_vs_default
-- question: Does leverage on existing credit (debt / credit on active lines) predict default?
SELECT
    CASE
        WHEN s.active_credit IS NULL OR s.active_credit = 0 THEN '0. no active credit'
        WHEN s.active_debt / s.active_credit < 0.25 THEN '1. <25%'
        WHEN s.active_debt / s.active_credit < 0.50 THEN '2. 25-50%'
        WHEN s.active_debt / s.active_credit < 0.75 THEN '3. 50-75%'
        ELSE '4. 75%+'
    END                                                        AS debt_utilisation_band,
    COUNT(*)                                                   AS applicants,
    ROUND(100.0 * AVG(a.TARGET), 2)                            AS default_rate_pct
FROM application_train a
LEFT JOIN v_bureau_summary s ON a.SK_ID_CURR = s.SK_ID_CURR
GROUP BY debt_utilisation_band
ORDER BY debt_utilisation_band;


-- name: previous_refusals_vs_default
-- question: Are clients previously refused by Home Credit riskier today?
SELECT
    CASE
        WHEN p.SK_ID_CURR IS NULL THEN '0. new client'
        WHEN p.prev_refused = 0   THEN '1. never refused'
        WHEN p.prev_refused = 1   THEN '2. refused once'
        WHEN p.prev_refused <= 3  THEN '3. refused 2-3x'
        ELSE '4. refused 4x+'
    END                                                        AS refusal_history,
    COUNT(*)                                                   AS applicants,
    ROUND(100.0 * AVG(a.TARGET), 2)                            AS default_rate_pct
FROM application_train a
LEFT JOIN v_previous_summary p ON a.SK_ID_CURR = p.SK_ID_CURR
GROUP BY refusal_history
ORDER BY refusal_history;


-- name: last_previous_application_status
-- question: What did the client's MOST RECENT previous application look like? (ROW_NUMBER to pick latest)
WITH ranked AS (
    SELECT SK_ID_CURR, NAME_CONTRACT_STATUS, DAYS_DECISION,
           ROW_NUMBER() OVER (PARTITION BY SK_ID_CURR ORDER BY DAYS_DECISION DESC) AS rn
    FROM previous_application
)
SELECT
    COALESCE(r.NAME_CONTRACT_STATUS, 'No previous application') AS last_previous_status,
    COUNT(*)                                                   AS applicants,
    ROUND(100.0 * AVG(a.TARGET), 2)                            AS default_rate_pct,
    ROUND(AVG(-r.DAYS_DECISION), 0)                            AS avg_days_since_decision
FROM application_train a
LEFT JOIN ranked r ON a.SK_ID_CURR = r.SK_ID_CURR AND r.rn = 1
GROUP BY last_previous_status
ORDER BY default_rate_pct DESC;


-- name: payment_delay_vs_default
-- question: Payment behaviour on past loans: does the share of late instalments predict default? (multi-step CTE)
WITH per_instalment AS (
    SELECT SK_ID_CURR,
           CASE WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1 ELSE 0 END AS is_late,
           MAX(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0)                    AS days_late
    FROM installments_payments
    WHERE DAYS_ENTRY_PAYMENT IS NOT NULL
),
per_client AS (
    SELECT SK_ID_CURR, AVG(is_late) AS late_share, MAX(days_late) AS worst_delay
    FROM per_instalment
    GROUP BY SK_ID_CURR
)
SELECT
    CASE
        WHEN c.SK_ID_CURR IS NULL THEN '0. no instalment history'
        WHEN c.late_share = 0     THEN '1. never late'
        WHEN c.late_share < 0.05  THEN '2. <5% late'
        WHEN c.late_share < 0.15  THEN '3. 5-15% late'
        WHEN c.late_share < 0.30  THEN '4. 15-30% late'
        ELSE '5. 30%+ late'
    END                                                        AS lateness_band,
    COUNT(*)                                                   AS applicants,
    ROUND(AVG(c.worst_delay), 1)                               AS avg_worst_delay_days,
    ROUND(100.0 * AVG(a.TARGET), 2)                            AS default_rate_pct
FROM application_train a
LEFT JOIN per_client c ON a.SK_ID_CURR = c.SK_ID_CURR
GROUP BY lateness_band
ORDER BY lateness_band;


-- name: credit_card_utilisation_vs_default
-- question: Latest credit-card utilisation before the application (ROW_NUMBER on monthly snapshots).
WITH latest AS (
    SELECT SK_ID_CURR, AMT_BALANCE, AMT_CREDIT_LIMIT_ACTUAL,
           ROW_NUMBER() OVER (PARTITION BY SK_ID_CURR ORDER BY MONTHS_BALANCE DESC) AS rn
    FROM credit_card_balance
),
util AS (
    SELECT SK_ID_CURR, 1.0 * AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0) AS utilisation
    FROM latest WHERE rn = 1
)
SELECT
    CASE
        WHEN u.SK_ID_CURR IS NULL  THEN '0. no card'
        WHEN u.utilisation IS NULL THEN '1. no limit'
        WHEN u.utilisation < 0.30  THEN '2. <30%'
        WHEN u.utilisation < 0.70  THEN '3. 30-70%'
        WHEN u.utilisation < 1.00  THEN '4. 70-100%'
        ELSE '5. over limit'
    END                                                        AS utilisation_band,
    COUNT(*)                                                   AS applicants,
    ROUND(100.0 * AVG(a.TARGET), 2)                            AS default_rate_pct
FROM application_train a
LEFT JOIN util u ON a.SK_ID_CURR = u.SK_ID_CURR
GROUP BY utilisation_band
ORDER BY utilisation_band;


-- name: worst_bureau_delinquency
-- question: Worst monthly delinquency status ever reported to the bureau (3-table join: application -> bureau -> bureau_balance).
WITH worst AS (
    SELECT b.SK_ID_CURR,
           MAX(CASE WHEN bb.STATUS IN ('1','2','3','4','5') THEN CAST(bb.STATUS AS INTEGER) ELSE 0 END) AS worst_status
    FROM bureau b
    JOIN bureau_balance bb ON b.SK_ID_BUREAU = bb.SK_ID_BUREAU
    GROUP BY b.SK_ID_CURR
)
SELECT
    CASE
        WHEN w.SK_ID_CURR IS NULL THEN 'no monthly history'
        WHEN w.worst_status = 0   THEN '0. never past due'
        WHEN w.worst_status = 1   THEN '1. 1-30 DPD'
        WHEN w.worst_status = 2   THEN '2. 31-60 DPD'
        ELSE '3. 60+ DPD / written off'
    END                                                        AS worst_bureau_status,
    COUNT(*)                                                   AS applicants,
    ROUND(100.0 * AVG(a.TARGET), 2)                            AS default_rate_pct
FROM application_train a
LEFT JOIN worst w ON a.SK_ID_CURR = w.SK_ID_CURR
GROUP BY worst_bureau_status
ORDER BY worst_bureau_status;


-- name: application_frequency_vs_default
-- question: Do clients who apply for credit in quick succession default more? (LAG between consecutive applications)
WITH gaps AS (
    SELECT SK_ID_CURR,
           DAYS_DECISION - LAG(DAYS_DECISION) OVER (PARTITION BY SK_ID_CURR ORDER BY DAYS_DECISION) AS days_between_apps
    FROM previous_application
),
per_client AS (
    SELECT SK_ID_CURR, AVG(days_between_apps) AS avg_gap_days
    FROM gaps WHERE days_between_apps IS NOT NULL
    GROUP BY SK_ID_CURR
)
SELECT
    CASE
        WHEN g.SK_ID_CURR IS NULL   THEN '0. fewer than 2 applications'
        WHEN g.avg_gap_days < 30    THEN '1. < 1 month apart'
        WHEN g.avg_gap_days < 180   THEN '2. 1-6 months apart'
        WHEN g.avg_gap_days < 365   THEN '3. 6-12 months apart'
        ELSE '4. > 1 year apart'
    END                                                        AS application_frequency,
    COUNT(*)                                                   AS applicants,
    ROUND(100.0 * AVG(a.TARGET), 2)                            AS default_rate_pct
FROM application_train a
LEFT JOIN per_client g ON a.SK_ID_CURR = g.SK_ID_CURR
GROUP BY application_frequency
ORDER BY application_frequency;


-- name: segment_matrix_region_education
-- question: Risk heat-map: region rating x education (conditional aggregation pivot).
SELECT
    REGION_RATING_CLIENT_W_CITY                                                             AS region_rating,
    ROUND(100.0 * AVG(CASE WHEN NAME_EDUCATION_TYPE = 'Higher education' THEN TARGET END), 2)              AS higher_edu_pct,
    ROUND(100.0 * AVG(CASE WHEN NAME_EDUCATION_TYPE = 'Incomplete higher' THEN TARGET END), 2)             AS incomplete_higher_pct,
    ROUND(100.0 * AVG(CASE WHEN NAME_EDUCATION_TYPE = 'Secondary / secondary special' THEN TARGET END), 2) AS secondary_pct,
    ROUND(100.0 * AVG(CASE WHEN NAME_EDUCATION_TYPE = 'Lower secondary' THEN TARGET END), 2)               AS lower_secondary_pct,
    COUNT(*)                                                                                AS applicants
FROM application_train
GROUP BY REGION_RATING_CLIENT_W_CITY
ORDER BY region_rating;


-- name: top_occupations_by_exposure
-- question: Which occupations hold the most credit exposure, and how risky are they? (running share of exposure)
WITH occ AS (
    SELECT COALESCE(OCCUPATION_TYPE, 'Unknown') AS occupation,
           COUNT(*) AS applicants,
           SUM(AMT_CREDIT) AS exposure,
           AVG(TARGET) AS default_rate
    FROM application_train
    GROUP BY occupation
)
SELECT
    occupation,
    applicants,
    ROUND(exposure / 1e6, 1)                                                     AS exposure_mn,
    ROUND(100.0 * default_rate, 2)                                               AS default_rate_pct,
    ROUND(100.0 * SUM(exposure) OVER (ORDER BY exposure DESC) / SUM(exposure) OVER (), 1) AS cum_exposure_pct,
    DENSE_RANK() OVER (ORDER BY default_rate DESC)                               AS risk_rank
FROM occ
ORDER BY exposure DESC
LIMIT 12;
