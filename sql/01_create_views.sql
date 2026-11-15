-- =============================================================================
-- 01_create_views.sql
-- Reusable analytical views on top of the raw tables loaded by credit_risk.database.build_database().
-- Run after the CSVs are loaded. Safe to re-run.
-- =============================================================================

DROP VIEW IF EXISTS v_applications;
CREATE VIEW v_applications AS
SELECT
    a.SK_ID_CURR,
    a.TARGET,
    a.NAME_CONTRACT_TYPE,
    a.CODE_GENDER,
    a.NAME_INCOME_TYPE,
    a.NAME_EDUCATION_TYPE,
    a.NAME_FAMILY_STATUS,
    a.NAME_HOUSING_TYPE,
    a.OCCUPATION_TYPE,
    a.REGION_RATING_CLIENT_W_CITY,
    a.AMT_INCOME_TOTAL,
    a.AMT_CREDIT,
    a.AMT_ANNUITY,
    a.AMT_GOODS_PRICE,
    a.EXT_SOURCE_1,
    a.EXT_SOURCE_2,
    a.EXT_SOURCE_3,
    -a.DAYS_BIRTH / 365.25                                   AS age_years,
    CASE
        WHEN -a.DAYS_BIRTH / 365.25 < 25 THEN '1. <25'
        WHEN -a.DAYS_BIRTH / 365.25 < 35 THEN '2. 25-34'
        WHEN -a.DAYS_BIRTH / 365.25 < 45 THEN '3. 35-44'
        WHEN -a.DAYS_BIRTH / 365.25 < 55 THEN '4. 45-54'
        WHEN -a.DAYS_BIRTH / 365.25 < 65 THEN '5. 55-64'
        ELSE '6. 65+'
    END                                                      AS age_band,
    CASE WHEN a.DAYS_EMPLOYED = 365243 THEN NULL
         ELSE -a.DAYS_EMPLOYED / 365.25 END                   AS years_employed,
    1.0 * a.AMT_CREDIT  / NULLIF(a.AMT_INCOME_TOTAL, 0)       AS credit_to_income,
    1.0 * a.AMT_ANNUITY / NULLIF(a.AMT_INCOME_TOTAL, 0)       AS annuity_to_income
FROM application_train a;


-- One row per client summarising their Credit Bureau file
DROP VIEW IF EXISTS v_bureau_summary;
CREATE VIEW v_bureau_summary AS
SELECT
    SK_ID_CURR,
    COUNT(*)                                                        AS bureau_loans,
    SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END)       AS active_loans,
    SUM(CASE WHEN CREDIT_DAY_OVERDUE > 0 THEN 1 ELSE 0 END)         AS loans_overdue_now,
    SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN COALESCE(AMT_CREDIT_SUM_DEBT, 0) ELSE 0 END) AS active_debt,
    SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN COALESCE(AMT_CREDIT_SUM, 0) ELSE 0 END)      AS active_credit,
    MIN(DAYS_CREDIT)                                                AS oldest_credit_days,
    MAX(COALESCE(AMT_CREDIT_MAX_OVERDUE, 0))                        AS max_overdue_amount
FROM bureau
GROUP BY SK_ID_CURR;


-- One row per client summarising previous Home Credit applications
DROP VIEW IF EXISTS v_previous_summary;
CREATE VIEW v_previous_summary AS
SELECT
    SK_ID_CURR,
    COUNT(*)                                                               AS prev_apps,
    SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 ELSE 0 END)     AS prev_approved,
    SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused'  THEN 1 ELSE 0 END)     AS prev_refused,
    AVG(AMT_CREDIT)                                                        AS prev_avg_credit
FROM previous_application
GROUP BY SK_ID_CURR;
