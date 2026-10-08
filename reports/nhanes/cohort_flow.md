# NHANES cohort flow

Produced by `scripts/nhanes_04_cohort_label.py`. Filters are applied in this order.

| step | criterion | excluded | remaining |
|---:|---|---:|---:|
| 0 | All participants in the 7 NHANES cycles (DEMO rows) | 0 | **76,496** |
| 1 | Age >= 18 years | 30,516 | **45,980** |
| 2 | Not pregnant at exam (RIDEXPRG = 1 excluded) | 769 | **45,211** |
| 3 | Non-missing serum creatinine AND non-missing urine ACR | 5,589 | **39,622** |

**Final analytic cohort: N = 39,622 adults.**

## By cycle

| cycle | years | split | n |
|---|---|---|---:|
| D | 2005-2006 | train | 4,569 |
| E | 2007-2008 | train | 5,405 |
| F | 2009-2010 | train | 5,832 |
| G | 2011-2012 | train | 5,049 |
| H | 2013-2014 | train | 5,480 |
| I | 2015-2016 | train | 5,249 |
| P | 2017-Mar 2020 (pre-pandemic) | test | 8,038 |

Train (2005-2016): **31,584**; Test (2017-Mar 2020): **8,038**.

## Label (Step 5)

KDIGO-style, single visit: `ckd = 1` if eGFR (CKD-EPI 2021, race-free, recorded sex) < 60 mL/min/1.73 m2 OR urine ACR >= 30 mg/g. NHANES has no 3-month confirmation, so chronicity cannot be verified.

| group | n | CKD | prevalence | early subgroup (eGFR >= 60) n | CKD in early subgroup | prevalence |
|---|---:|---:|---:|---:|---:|---:|
| all | 39,622 | 6,768 | 17.1% | 36,677 | 3,823 | 10.4% |
| train | 31,584 | 5,301 | 16.8% | 29,257 | 2,974 | 10.2% |
| test | 8,038 | 1,467 | 18.3% | 7,420 | 849 | 11.4% |

Label composition (all): eGFR < 60 only 1,812; ACR >= 30 only 3,823; both 1,133.

## Plausibility on the adult cohort

| column | n checked | hard range | violations |
|---|---:|---|---:|
| age | 39,622 | (0, 120) | 0 |
| blood_pressure | 37,818 | (30, 300) | 271 |
| serum_creatinine | 39,622 | (0, 30) | 0 |
| sodium | 39,618 | (100, 200) | 1 |
| potassium | 39,609 | (1, 10) | 0 |
| hemoglobin | 39,539 | (1, 25) | 0 |
| blood_glucose_random | 39,620 | (20, 1000) | 1 |
| packed_cell_volume | 39,539 | (0, 80) | 0 |
| white_blood_cell_count | 39,539 | (1000, 50000) | 10 |
| red_blood_cell_count | 39,539 | (0, 10) | 0 |
