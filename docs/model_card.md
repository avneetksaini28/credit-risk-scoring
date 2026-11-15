# Model card: Home Credit default-risk model

## Model details
- **Purpose:** estimate the probability that a new consumer-loan applicant will have payment difficulties (Home Credit's label: more than X days late on at least one of the first Y instalments)
- **Type:** LightGBM gradient-boosted trees (984 trees, early stopping on validation AUC), with class weighting (scale_pos_weight 11.4) and Platt calibration on a held-out validation split
- **Inputs:** 200 features (188 numeric, 12 categorical) from the application, Credit Bureau history and previous Home Credit loans (`models/feature_lists.json`)
- **Output:** calibrated PD in [0, 1] → risk grade A–E → approve / refer / decline (`models/decision_policy.json`)
- **Challenger models evaluated:** XGBoost (statistically tied), logistic regression (−0.014 AUC, fully interpretable), random forest

## Intended use
- Decision **support** for underwriters in a consumer-lending context, as a portfolio demonstration only
- **Out of scope:** real lending decisions, other products (mortgages, SME lending), other markets, and populations unlike Home Credit's applicants

## Training data
- Home Credit Default Risk (Kaggle): 307,511 labelled applications plus ~58M rows of credit history across 7 related tables; default rate 8.07%
- Split: 70% train (215,257) / 15% validation (46,127) / 15% test (46,127), stratified. The data has no time stamps, so no out-of-time sample was possible.
- Known gaps: outcomes exist only for approved loans (no reject inference); no out-of-time sample

## Performance (test set, 46,127 applicants)
| Metric | Value |
|---|---|
| ROC-AUC (95% bootstrap CI) | 0.792 (0.785–0.800) |
| Gini | 0.585 |
| KS | 0.445 |
| PR-AUC | 0.294 (base rate 0.081) |
| Brier score | 0.0654 |
| Mean PD vs actual default rate | 8.07% vs 8.07% |
| Defaults captured in riskiest 10% / 20% | 39% / 58% |
| Train vs validation AUC | 0.869 vs 0.790 |

## Decision policy
- Assumptions: LGD 60%, net margin 15% of credit amount → analytic break-even PD 20%
- Decline if PD ≥ 21% (profit-optimal on validation); auto-approve if PD < 14.7%; refer to an underwriter in between
- Test set: approval rate 90.6%; bad rate of approved 5.57% (vs 8.07% overall); 33.6% of credit losses avoided; net profit +7.5% vs approving everyone
- Decision mix: Approve 83.2% (actual default 4.6%) · Refer 7.4% (16.7%) · Decline 9.4% (32.0%)
- Risk grades A / B / C / D / E: observed default rates 1.8% / 4.2% / 7.6% / 13.2% / 31.1%
- Sensitivity: the optimal cutoff ranges from 9% to 41% across LGD 40–90% and margin 8–30%

## Explainability
- SHAP TreeExplainer on the champion model
- Top global drivers: external credit scores (average, lowest, highest, combined), loan term (credit ÷ annuity), time in current employment, remaining instalments on existing loans, previously granted vs requested amount, education level
- About 42% of total importance comes from aggregated credit-history features
- Reason codes: the top four positive SHAP contributions for an applicant, mapped to plain-language labels

## Fairness
- **Excluded inputs:** CODE_GENDER and NAME_FAMILY_STATUS (kept only for outcome testing). Age (AGE_YEARS) is retained and monitored.
- **Lowest adverse impact ratio:** gender 0.88 · marital status 0.92 · age group 0.80 (under 25)
- **Calibration by group:** mean PD tracks the actual default rate within about 1 percentage point for every gender, age and marital-status group
- **Error rates:** creditworthy under-25s are declined more often than older applicants (false positive rate 17.5% vs 2.4% for 55–64), driven by a higher actual default rate (12.8%) and thinner credit histories
- **Proxy test:** gender can be predicted from the model's inputs with AUC 0.90 (driven by occupation, car age and employer type), so outcome testing is required even though gender is excluded
- **Less-discriminatory-alternative test:** removing age features changes test AUC only slightly (0.7923 → 0.7915) but lowers the under-25 AIR from 0.80 to 0.75, so the age-free model is not a less discriminatory alternative
- **Recommendation before any deployment:** formal fair-lending review of the under-25 segment (AIR at the 0.80 threshold), focusing on the underwriter refer band and alternative data for thin-file applicants rather than removing age

## Monitoring plan
| What | How often | Trigger |
|---|---|---|
| Score PSI vs development sample | Monthly | > 0.10 investigate, > 0.25 escalate |
| Feature PSI (top 20 features) | Monthly | > 0.25 |
| Approval rate, refer rate, override rate | Weekly | ±20% vs plan |
| Realised default rate by grade (as outcomes mature) | Quarterly | Outside PD confidence band |
| Fairness metrics (AIR, false positive rate gaps) | Quarterly | AIR < 0.80 |
| Full revalidation | Annually | Always |

## Limitations
- No time-based validation, because the data has no application dates
- LGD and margin are assumptions, not estimates
- External scores (EXT_SOURCE_*) are opaque third-party scores, which complicates customer-facing reason codes
- No reject inference
- Possible proxies for protected attributes among the remaining features
- The data is several years old, so a production model would be retrained on recent data
