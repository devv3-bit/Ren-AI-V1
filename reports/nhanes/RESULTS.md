# Ren AI v2: NHANES training, temporal validation and external validation

Branch `nhanes-v2`. All numbers below are produced by the scripts in `scripts/nhanes_0*.py`
and stored in `reports/nhanes/*.json|csv`; nothing in this document was computed by hand.
Seeds are fixed at 42. The held-out cycle was evaluated once, after every modelling decision
was frozen and written down (`run_log.md`, "Decision record").

## 1. Headline

| quantity | value | what it measures |
|---|---|---|
| Final analytic cohort | **39,622 adults** | NHANES 2005-Mar 2020 participants aged >= 18, not pregnant, with serum creatinine and urine ACR measured |
| Training set | 31,584 (CKD 16.8%) | cycles 2005-2016 |
| Held-out test set | 8,038 (CKD 18.3%) | cycle 2017-Mar 2020, never used for any decision |
| Final model | gradient boosting (Model B), 25 features | chosen by mean 5-fold CV ROC-AUC on train before the test set was touched |
| **Held-out ROC-AUC** | **0.856 [0.844, 0.868]** | probability that a random CKD participant scores higher than a random non-CKD participant |
| **Early-subgroup ROC-AUC** (eGFR >= 60) | **0.751 [0.733, 0.768]** | same, restricted to the 7,420 test participants whose CKD can only be albuminuria (KDIGO G1-G2 A2-A3) |
| **External ROC-AUC**, 399 UCI hospital patients | **0.641 [0.587, 0.701]** | UCI-compatible logistic model (14 shared features) trained on NHANES only, applied unchanged |
| Blood tests / engineered scores used | 17 / 2 | analytes from serum chemistry, CBC and HbA1c; eGFR (CKD-EPI 2021) and BUN:creatinine ratio |

95% CIs are percentile bootstraps (1,000 resamples, seed 42).

## 2. Data and cohort (Steps 1-4)

Seven public CDC NHANES cycles: 2005-06, 2007-08, 2009-10, 2011-12, 2013-14, 2015-16 and the
combined 2017-Mar 2020 pre-pandemic release (`P_` files). Eight component files per cycle
(DEMO, BIOPRO, ALB_CR, CBC, BPX/BPXO, DIQ, BMX, GHB), 56 files in total, fetched by
`scripts/download_nhanes.sh` and verified by `scripts/nhanes_01_verify_download.py`
(`raw_file_manifest.md`). Components are left-joined onto DEMO on `SEQN` (`merged.parquet`,
76,496 rows x 274 columns).

Harmonisation (`src/nhanes/harmonize.py`, `harmonization_report.md`):
2005-06 creatinine recalibrated with the CDC formula `-0.016 + 0.978 x LBXSCR` (confirmed
in the BIOPRO_D analytic notes); WBC x 1000 to the UCI scale; BP = mean of available readings
(oscillometric columns for 2017-2020); ACR = URDACT, or URXUMA/URXUCR x 100 for 2005-08 where
URDACT is not released; diabetes 1/2/3 -> 1/0/0, 7/9 -> missing. The repo's hard-range rules
(`src/config.py::VALIDATION_RULES` via `validate_labs`) flag 863 values in the full table
(851 diastolic readings < 30 mmHg, 10 WBC > 50,000, 1 sodium, 1 glucose) and 283 in the adult
cohort; they are reported, not removed, as in v1.

| step | criterion | excluded | remaining |
|---:|---|---:|---:|
| 0 | All participants in the 7 cycles (DEMO rows) | - | 76,496 |
| 1 | Age >= 18 years | 30,516 | 45,980 |
| 2 | Not pregnant at exam (RIDEXPRG = 1 excluded) | 769 | 45,211 |
| 3 | Serum creatinine AND urine ACR measured | 5,589 | **39,622** |

By cycle: D 4,569; E 5,405; F 5,832; G 5,049; H 5,480; I 5,249; P 8,038 (`cohort_flow.md`).

## 3. Label (Step 5)

KDIGO-style, single visit: `ckd = 1` if eGFR < 60 mL/min/1.73 m2 OR urine ACR >= 30 mg/g.
eGFR is CKD-EPI 2021 (race-free) from `src/features/derive.py::egfr_cr`, vectorised in
`src/nhanes/egfr.py` (unit-tested against the scalar version to 1e-12 on 5,000 random rows),
using each participant's recorded sex.

| group | n | CKD | prevalence | eGFR >= 60 subgroup | CKD in subgroup | prevalence |
|---|---:|---:|---:|---:|---:|---:|
| all | 39,622 | 6,768 | 17.1% | 36,677 | 3,823 | 10.4% |
| train 2005-2016 | 31,584 | 5,301 | 16.8% | 29,257 | 2,974 | 10.2% |
| test 2017-2020 | 8,038 | 1,467 | 18.3% | 7,420 | 849 | 11.4% |

Label composition (all): eGFR < 60 only 1,812; ACR >= 30 only 3,823; both 1,133. More than
half of the CKD cases are albuminuria-only, which no blood test measures directly.

## 4. Features (Step 6)

* **17 blood tests**: serum creatinine, BUN, glucose (random), sodium, potassium, chloride,
  bicarbonate, serum albumin, uric acid, calcium, phosphorus, hemoglobin, hematocrit (PCV),
  RBC count, WBC count, platelets, HbA1c. (Three standard panels: metabolic panel, CBC, HbA1c.)
* **6 vitals / demographics**: age, sex, systolic BP, diastolic BP, BMI, self-reported diabetes.
* **2 engineered kidney scores**: eGFR (CKD-EPI 2021) and BUN:creatinine ratio.
* Missing-value indicators are added inside the imputer, as in v1 (Model A); Model B handles
  missing values natively.
* Creatinine, BUN, glucose and the BUN:creatinine ratio are log1p-transformed (v1 policy).
* Urine albumin, urine creatinine and ACR define the label and are **never** features
  (enforced by `build_X`, tested).

eGFR is both a feature and half of the label definition, so the overall AUC is partly
definitional (any participant with eGFR < 60 is CKD by construction). The early-subgroup AUC
in Section 7 is the honest measure of what blood work adds beyond that.

## 5. Split (Step 7)

Temporal: train on 2005-2016 (31,584), test on 2017-Mar 2020 (8,038). The test cycle also
switched to an oscillometric BP protocol, so it is a modest distribution shift as well as a
time shift.

## 6. Model development (Step 8; train only)

5-fold stratified CV (seed 42) on the training cycles. Every configuration is in
`cv_results.csv` and `run_log.md`.

| model set | grid | chosen | mean CV AUC (sd) |
|---|---|---|---:|
| Model A: imputer + indicators + scaler + balanced L2 logistic | C in {0.001, 0.01, 0.1, 1, 10} | C = 1.0 | 0.827 (0.004) |
| Model B: HistGradientBoosting | lr {0.05, 0.1} x depth {3, 6} x iters {100, 300} | lr 0.05, depth 6, 100 iters | 0.864 (0.008) |
| Baseline: eGFR alone (logistic) | C grid | C = 0.001 (all equal) | 0.760 (0.007) |
| Baseline: age + sex + diabetes + SBP + DBP (logistic) | C grid | C = 10 | 0.773 (0.008) |
| Model A, 14 UCI-shared features (for Step 10) | C grid | C = 1.0 | 0.810 (0.009) |

Pre-registered rule: final model = highest mean CV AUC among Model A and Model B -> **Model B**.
Decision thresholds are Youden's J on train out-of-fold predictions (Model B 0.140, Model A
0.546), never chosen on the test set. Fitted estimators are frozen in
`data/external/nhanes/models/` and described in `frozen_config.json`.

## 7. Held-out evaluation, 2017-Mar 2020 (Step 9; run once)

n = 8,038; CKD 1,467 (18.3%); early subgroup 7,420 with 849 CKD (11.4%).

| model | train CV AUC | **test AUC [95% CI]** | **early-subgroup AUC [95% CI]** | sens | spec | PPV | NPV | Brier | survey-weighted AUC |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| **Model B (final)** | 0.864 | **0.856 [0.844, 0.868]** | **0.751 [0.733, 0.768]** | 0.744 | 0.803 | 0.457 | 0.933 | 0.083 | 0.848 |
| Model A (logistic, 25 features) | 0.827 | 0.818 [0.803, 0.831] | 0.701 [0.679, 0.720] | 0.738 | 0.749 | 0.396 | 0.927 | 0.199 | 0.799 |
| Model A, 14 shared features | 0.810 | 0.811 [0.796, 0.824] | 0.687 [0.667, 0.706] | 0.643 | 0.862 | 0.509 | 0.915 | 0.171 | 0.796 |
| Baseline: eGFR alone | 0.760 | 0.743 [0.726, 0.758] | 0.556 [0.535, 0.577] | 0.487 | 0.938 | 0.638 | 0.891 | 0.192 | 0.715 |
| Baseline: age + sex + diabetes + BP | 0.773 | 0.763 [0.749, 0.776] | 0.691 [0.671, 0.713] | 0.695 | 0.727 | 0.362 | 0.914 | 0.203 | 0.752 |

Sensitivity/specificity/PPV/NPV are at each model's train-CV Youden threshold. Plain English,
for the final model:

* **AUC 0.856**: given one random adult with CKD and one without, the model ranks the CKD
  adult higher 85.6% of the time. The CV estimate (0.864) carried over to the later cycle with
  almost no optimism, so the model was not over-fitted to its training years.
* **Early-subgroup AUC 0.751**: among people whose eGFR is still normal, blood work alone ranks
  the albuminuric (early CKD) person higher 75% of the time. eGFR alone is near chance here
  (0.556); the gain to 0.751 is what the other blood tests, BP, HbA1c and age contribute.
* **Sensitivity 0.744 / specificity 0.803** at the train-chosen threshold: the model flags
  29.7% of test participants (2,388), catches 1,091 of the 1,467 CKD cases and misses 376.
  In the early subgroup at the same threshold, sensitivity is 0.557 and specificity 0.803.
* **PPV 0.457 / NPV 0.933**: fewer than half of flagged adults actually have CKD; 93.3% of
  unflagged adults are truly CKD-free. At 18% prevalence the model is a rule-out/triage tool,
  not a diagnosis.
* **Brier 0.083** (versus 0.149 for always predicting the training prevalence) and the
  calibration curve (Figure 4) show Model B's probabilities are well calibrated. Model A's
  Brier (0.199) is poor because `class_weight="balanced"` inflates its probabilities, as in v1;
  its ranking (AUC) is unaffected.
* **Survey-weighted AUC 0.848** (MEC weights WTMECPRP, secondary): the result is not an
  artefact of NHANES oversampling.

Permutation importance of the final model on the test set (Figure 6): eGFR dominates
(-0.104 AUC when permuted), then systolic BP (-0.017), HbA1c (-0.012), age (-0.011), serum
creatinine (-0.007), diabetes (-0.007), BMI (-0.006).

## 8. External validation on the 399 UCI hospital patients (Step 10; pre-registered result)

Model: the UCI-compatible Model A (logistic, C = 1.0, 14 features: age, diastolic BP, glucose,
BUN, creatinine, eGFR, BUN:creatinine ratio, sodium, potassium, hemoglobin, PCV, WBC, RBC,
diabetes), trained and tuned on NHANES 2005-2016 only, applied unchanged. UCI harmonisation:
BUN = urea / 2.14; `bp` is diastolic; `pcv` = hematocrit; diabetes yes/no -> 1/0; hypertension
not mappable (NHANES BPQ not in scope) and dropped. UCI has no sex column, so eGFR is computed
as male (the repo's v1 fallback) with female as a sensitivity analysis.

| eGFR assumption | **AUC [95% CI]** | sens | spec | PPV | NPV | Brier |
|---|---|---:|---:|---:|---:|---:|
| male (primary) | **0.641 [0.587, 0.701]** | 0.636 | 0.866 | 0.888 | 0.586 | 0.282 |
| female (sensitivity) | 0.642 [0.587, 0.702] | 0.644 | 0.725 | 0.797 | 0.548 | 0.300 |

Threshold 0.588 = Youden's J on NHANES train OOF. n = 399 (250 CKD, 62.7%). Plain English: on
these hospital patients the NHANES-trained model ranks a CKD patient above a non-CKD patient
only 64% of the time; it catches 64% of CKD patients and clears 87% of non-CKD patients.
This is a weak transfer, and it is the number that must be quoted.

### 8a. Post-hoc diagnosis (exploratory; NOT pre-registered; do not cite as a result)

`scripts/nhanes_09_uci_diagnostics.py` applies the same frozen model to subsets of the UCI data
(nothing retrained; `external_diagnostics.json`):

| UCI subset | n (CKD %) | AUC [95% CI] | median p(CKD) for CKD patients |
|---|---:|---|---:|
| all rows (pre-registered) | 399 (62.7%) | 0.641 [0.587, 0.701] | 0.982 |
| all 14 features observed | 212 (41.0%) | 0.982 [0.955, 0.999] | 1.000 |
| potassium and sodium observed | 311 (53.7%) | 0.937 [0.900, 0.971] | 1.000 |
| potassium or sodium missing | 88 (94.3%) | 0.586 [0.129, 0.930] | 0.000 |

Mechanism: in NHANES training data potassium is missing for 0.02% of rows, so after
`StandardScaler` the "missing potassium" indicator is worth z = 72.5 when present, and its
coefficient (-0.756 per SD) becomes a shift of **-55 log-odds**. In UCI, 22% of patients (almost
all CKD) have no potassium result and are pushed to probability ~0. The clinical signal itself
transfers: on UCI, serum creatinine alone gives AUC 0.922, eGFR alone 0.910, hemoglobin alone
0.969. The failure is a brittle preprocessing choice inherited from v1 (standardising
rare-missingness indicators), not a failure of the blood tests. The fix (drop or leave unscaled
indicators with < 1% missingness, or regularise harder) is a v3 change and is deliberately NOT
applied here, because it was discovered after seeing the external result.

## 9. Figures (`reports/nhanes/figures/`)

1. `fig01_cohort_flow.png` cohort flow and temporal split
2. `fig02_roc_nhanes_test.png` ROC on the held-out cycle, all participants (A) and early subgroup (B)
3. `fig03_roc_uci_external.png` ROC on the 399 UCI patients (male / female eGFR)
4. `fig04_calibration_nhanes_test.png` calibration of Model B and Model A on the held-out cycle
5. `fig05_model_a_odds_ratios.png` Model A odds ratios, top 15 by |log-odds|
6. `fig06_permutation_importance.png` permutation importance of the final model, top 15

## 10. Limitations

* **Single-visit labels.** KDIGO requires abnormalities to persist >= 3 months; NHANES measures
  once, so some labelled CKD is transient (acute illness, exercise, fever raise ACR).
* **eGFR is in both the label and the features.** The overall AUC is partly definitional; the
  early-subgroup AUC (0.751) is the fair estimate of early detection without a urine test, and
  it is moderate.
* **NHANES survey design.** Primary metrics are unweighted; the survey-weighted AUC (0.848) is
  reported as a check, but no variance estimation uses the PSU/strata design.
* **UCI label mismatch.** UCI labels are clinical diagnoses of mostly advanced CKD in one Indian
  hospital (62.7% prevalence); NHANES labels are KDIGO thresholds in the general US population
  (17%). The external test therefore mixes country, setting, labelling method and missingness
  pattern, and the 14-feature logistic model, not the 0.856 gradient-boosting model, is what
  was transferred.
* **Missing sex in UCI.** eGFR was computed as male (primary) and female (sensitivity); the
  two give the same AUC to two decimals.
* **Missing-value indicators are brittle across datasets** (Section 8a). They work within
  NHANES (train and test share the same missingness mechanism) but not across sites.
* **Model A collinearity.** log creatinine, log BUN and log BUN:creatinine ratio are nearly
  collinear, so Model A's individual odds ratios for those three terms (Figure 5) are not
  interpretable on their own; only their sum is.
* **Hypertension** could not be used for the external test (not in the NHANES files downloaded).
* Implausible values flagged by the repo's hard ranges (283 in the cohort, mostly diastolic
  readings < 30 mmHg) were kept, matching v1 behaviour.

## 11. Reproduction

```bash
bash scripts/download_nhanes.sh
.venv/bin/python scripts/nhanes_01_verify_download.py
.venv/bin/python scripts/nhanes_02_load_merge.py
.venv/bin/python scripts/nhanes_03_harmonize.py
.venv/bin/python scripts/nhanes_04_cohort_label.py
.venv/bin/python scripts/nhanes_05_train_cv.py      # train only; freezes models
.venv/bin/python scripts/nhanes_06_evaluate_test.py # the single test pass
.venv/bin/python scripts/nhanes_07_external_uci.py
.venv/bin/python scripts/nhanes_08_figures.py
.venv/bin/python scripts/nhanes_09_uci_diagnostics.py  # post-hoc, exploratory
.venv/bin/python -m pytest -q
```

Outputs: `baseline_uci.txt`, `raw_file_manifest.md`, `harmonization_report.md`,
`cohort_flow.md`, `cohort_summary.json`, `cv_results.csv`, `frozen_config.json`,
`model_A_coefficients.csv`, `test_metrics.json`, `test_predictions.csv`,
`permutation_importance.csv`, `external_metrics.json`, `uci_predictions.csv`,
`external_diagnostics.json`, `run_log.md`, `CLAIMS.md`, `figures/`.
