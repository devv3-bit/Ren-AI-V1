# NHANES harmonisation report

Produced by `scripts/nhanes_03_harmonize.py` from `data/external/nhanes/merged.parquet`.

Rows: **76,496** participants (every DEMO row of the 7 cycles); columns: 35.

## Rules applied

- Direct renames per `src/nhanes/constants.py::DIRECT_MAP` (canonical names match `src/config.py`).
- `sex`: RIAGENDR 1 -> male, 2 -> female; `sex_male` numeric 1/0.
- `pregnant`: RIDEXPRG 1 -> 1, 2 -> 0, 3 (cannot ascertain) -> missing.
- `diabetes_mellitus`: DIQ010 1 -> 1, 2 -> 0, 3 (borderline) -> 0, 7/9 (refused/don't know) -> missing.
- `serum_creatinine`: 2005-06 (cycle D) only, CDC Deming recalibration `-0.016 + 0.978 x LBXSCR` (confirmed in the BIOPRO_D analytic notes: "Standard creatinine (mg/dL) = -0.016 + 0.978 X (NHANES 05-06 uncalibrated serum creatinine, mg/dL)"). Raw value kept as `serum_creatinine_raw`.
- `white_blood_cell_count`: LBXWBCSI x 1000 (NHANES 1000 cells/uL -> cells/uL, the UCI scale).
- `sbp` / `blood_pressure` (diastolic): mean of the available readings BPXSY1-4 / BPXDI1-4 (BPXOSY1-3 / BPXODI1-3 for the 2017-2020 oscillometric protocol); readings of 0 are treated as missing (none occur in these files, so this is a no-op safeguard).
- `acr` (mg/g): URDACT where reported (2009-10 onward); 2005-06 and 2007-08 have no URDACT, so ACR = URXUMA (ug/mL) / URXUCR (mg/dL) x 100. Only the first urine collection is used.
- `mec_weight`: WTMEC2YR (2-year cycles) or WTMECPRP (2017-2020).

## Non-missing counts per cycle

| column | D | E | F | G | H | I | P |
|---|---|---|---|---|---|---|---|
| age | 10348 | 10149 | 10537 | 9756 | 10175 | 9971 | 15560 |
| sex | 10348 | 10149 | 10537 | 9756 | 10175 | 9971 | 15560 |
| pregnant | 3173 | 1153 | 1334 | 1123 | 1215 | 1195 | 1691 |
| serum_creatinine | 6434 | 6376 | 6860 | 5976 | 6553 | 6255 | 9475 |
| acr | 7843 | 7878 | 8398 | 7636 | 8052 | 8280 | 12509 |
| bun | 6434 | 6375 | 6860 | 5975 | 6553 | 6255 | 9473 |
| blood_glucose_random | 6434 | 6377 | 6860 | 5976 | 6553 | 6257 | 9473 |
| sodium | 6434 | 6375 | 6860 | 5974 | 6553 | 6257 | 9476 |
| potassium | 6433 | 6374 | 6860 | 5973 | 6552 | 6257 | 9466 |
| chloride | 6434 | 6374 | 6860 | 5974 | 6553 | 6257 | 9476 |
| bicarbonate | 6349 | 6375 | 6860 | 5974 | 6553 | 6257 | 9473 |
| serum_albumin | 6434 | 6377 | 6860 | 5976 | 6553 | 6256 | 9477 |
| uric_acid | 6433 | 6375 | 6859 | 5974 | 6551 | 6254 | 9473 |
| calcium | 6434 | 6375 | 6860 | 5974 | 6511 | 6257 | 9473 |
| phosphorus | 6432 | 6377 | 6856 | 5976 | 6552 | 6256 | 9473 |
| hemoglobin | 8400 | 8267 | 8799 | 7953 | 8544 | 8117 | 12156 |
| packed_cell_volume | 8400 | 8267 | 8799 | 7953 | 8544 | 8117 | 12156 |
| red_blood_cell_count | 8400 | 8267 | 8799 | 7953 | 8544 | 8117 | 12156 |
| white_blood_cell_count | 8400 | 8266 | 8798 | 7953 | 8544 | 8117 | 12156 |
| platelets | 8400 | 8267 | 8799 | 7952 | 8544 | 8117 | 12156 |
| hba1c | 6493 | 6427 | 6930 | 6145 | 6643 | 6326 | 9737 |
| bmi | 8949 | 8861 | 9412 | 8602 | 9055 | 8756 | 13137 |
| diabetes_mellitus | 9813 | 9657 | 10103 | 9357 | 9763 | 9571 | 14978 |
| sbp | 7359 | 7350 | 7818 | 7055 | 7531 | 7363 | 10353 |
| blood_pressure | 7359 | 7350 | 7818 | 7055 | 7531 | 7363 | 10353 |
| mec_weight | 10348 | 10149 | 10537 | 9756 | 10175 | 9971 | 15560 |

## ACR source per cycle

| cycle | URDACT | computed | missing |
|---|---|---|---|
| D | 0 | 7843 | 2505 |
| E | 0 | 7878 | 2271 |
| F | 8398 | 0 | 2139 |
| G | 7636 | 0 | 2120 |
| H | 8052 | 0 | 2123 |
| I | 8280 | 0 | 1691 |
| P | 12509 | 0 | 3051 |

## 2005-06 serum creatinine before/after recalibration

| variable | count | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|---|
| serum_creatinine_raw | 6434 | 0.8899 | 0.3887 | 0.4 | 0.7 | 0.8 | 1 | 17.8 |
| serum_creatinine | 6434 | 0.8543 | 0.3802 | 0.3752 | 0.6686 | 0.7664 | 0.962 | 17.39 |

## Biological plausibility (hard ranges from `src/config.py::VALIDATION_RULES`)

Values are reported, not removed, to keep the cohort definition exact. `blood_urea` is not present because NHANES reports BUN (`bun`), not urea.

| column | present | n_checked | hard_range | hard_violations | soft_warnings |
|---|---|---|---|---|---|
| age | True | 76496 | (0, 120) | 0 | 18981 |
| blood_pressure | True | 54829 | (30, 300) | 851 | 0 |
| serum_creatinine | True | 47929 | (0, 30) | 0 | 0 |
| blood_urea | False | 0 | (0, 300) | 0 | 0 |
| sodium | True | 47929 | (100, 200) | 1 | 0 |
| potassium | True | 47915 | (1, 10) | 0 | 0 |
| hemoglobin | True | 62236 | (1, 25) | 0 | 0 |
| blood_glucose_random | True | 47930 | (20, 1000) | 1 | 0 |
| packed_cell_volume | True | 62236 | (0, 80) | 0 | 0 |
| white_blood_cell_count | True | 62234 | (1000, 50000) | 10 | 0 |
| red_blood_cell_count | True | 62236 | (0, 10) | 0 | 0 |

**Total hard-range violations: 863** (soft warnings: 18981).
