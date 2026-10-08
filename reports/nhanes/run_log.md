# Ren AI v2 run log

Every configuration run on NHANES, in order, including the ones that did worse. Appended automatically by the scripts; decisions are recorded by hand.

## Run 2026-10-08T07:54:10+00:00 - scripts/nhanes_05_train_cv.py

- train N = 31,584 (cycles 2005-2016), CKD prevalence = 0.1678
- StratifiedKFold(n_splits=5, shuffle=True, random_state=42) on train cycles only; scoring = ROC-AUC; seed = 42; sklearn 1.8.0
- selection rule: highest mean CV ROC-AUC among ['model_A_full', 'model_B_full']

| model_set | family | params | n_features | fold AUCs | mean AUC | sd | seconds |
|---|---|---|---:|---|---:|---:|---:|
| model_A_full | logistic | C=0.001 | 25 | 0.8139, 0.8146, 0.8153, 0.8292, 0.8074 | 0.8161 | 0.0080 | 0.3 |
| model_A_full | logistic | C=0.01 | 25 | 0.8145, 0.8159, 0.8186, 0.8298, 0.8118 | 0.8181 | 0.0070 | 0.3 |
| model_A_full | logistic | C=0.1 | 25 | 0.8192, 0.8227, 0.8236, 0.8328, 0.8199 | 0.8237 | 0.0055 | 0.4 |
| model_A_full | logistic | C=1.0 | 25 | 0.8220, 0.8282, 0.8248, 0.8334, 0.8268 | 0.8270 | 0.0042 | 0.4 |
| model_A_full | logistic | C=10.0 | 25 | 0.8217, 0.8289, 0.8241, 0.8325, 0.8277 | 0.8270 | 0.0042 | 0.4 |
| model_B_full | hgb | learning_rate=0.05, max_depth=3, max_iter=100 | 25 | 0.8532, 0.8651, 0.8497, 0.8610, 0.8514 | 0.8561 | 0.0067 | 1.3 |
| model_B_full | hgb | learning_rate=0.05, max_depth=3, max_iter=300 | 25 | 0.8613, 0.8704, 0.8563, 0.8674, 0.8598 | 0.8630 | 0.0057 | 3.3 |
| model_B_full | hgb | learning_rate=0.05, max_depth=6, max_iter=100 | 25 | 0.8632, 0.8737, 0.8534, 0.8682, 0.8618 | 0.8641 | 0.0076 | 2.9 |
| model_B_full | hgb | learning_rate=0.05, max_depth=6, max_iter=300 | 25 | 0.8636, 0.8670, 0.8492, 0.8688, 0.8601 | 0.8617 | 0.0078 | 9.7 |
| model_B_full | hgb | learning_rate=0.1, max_depth=3, max_iter=100 | 25 | 0.8593, 0.8707, 0.8550, 0.8675, 0.8587 | 0.8623 | 0.0066 | 0.8 |
| model_B_full | hgb | learning_rate=0.1, max_depth=3, max_iter=300 | 25 | 0.8629, 0.8665, 0.8525, 0.8690, 0.8597 | 0.8621 | 0.0064 | 3.2 |
| model_B_full | hgb | learning_rate=0.1, max_depth=6, max_iter=100 | 25 | 0.8632, 0.8695, 0.8515, 0.8693, 0.8598 | 0.8627 | 0.0075 | 3.6 |
| model_B_full | hgb | learning_rate=0.1, max_depth=6, max_iter=300 | 25 | 0.8581, 0.8581, 0.8395, 0.8640, 0.8496 | 0.8539 | 0.0095 | 9.4 |
| baseline_egfr | logistic | C=0.001 | 1 | 0.7643, 0.7573, 0.7585, 0.7704, 0.7519 | 0.7605 | 0.0071 | 0.1 |
| baseline_egfr | logistic | C=0.01 | 1 | 0.7643, 0.7573, 0.7585, 0.7704, 0.7519 | 0.7605 | 0.0071 | 0.1 |
| baseline_egfr | logistic | C=0.1 | 1 | 0.7643, 0.7573, 0.7585, 0.7704, 0.7519 | 0.7605 | 0.0071 | 0.1 |
| baseline_egfr | logistic | C=1.0 | 1 | 0.7643, 0.7573, 0.7585, 0.7704, 0.7519 | 0.7605 | 0.0071 | 0.1 |
| baseline_egfr | logistic | C=10.0 | 1 | 0.7643, 0.7573, 0.7585, 0.7704, 0.7519 | 0.7605 | 0.0071 | 0.0 |
| baseline_demographic | logistic | C=0.001 | 5 | 0.7726, 0.7788, 0.7751, 0.7778, 0.7594 | 0.7728 | 0.0079 | 0.1 |
| baseline_demographic | logistic | C=0.01 | 5 | 0.7731, 0.7790, 0.7752, 0.7780, 0.7595 | 0.7729 | 0.0079 | 0.1 |
| baseline_demographic | logistic | C=0.1 | 5 | 0.7731, 0.7790, 0.7752, 0.7780, 0.7595 | 0.7730 | 0.0079 | 0.1 |
| baseline_demographic | logistic | C=1.0 | 5 | 0.7731, 0.7790, 0.7752, 0.7780, 0.7595 | 0.7730 | 0.0079 | 0.1 |
| baseline_demographic | logistic | C=10.0 | 5 | 0.7731, 0.7790, 0.7752, 0.7780, 0.7595 | 0.7730 | 0.0079 | 0.1 |
| model_A_harmonized | logistic | C=0.001 | 14 | 0.7994, 0.7972, 0.7935, 0.8143, 0.7895 | 0.7988 | 0.0094 | 0.2 |
| model_A_harmonized | logistic | C=0.01 | 14 | 0.8019, 0.8006, 0.7960, 0.8177, 0.7929 | 0.8018 | 0.0096 | 0.2 |
| model_A_harmonized | logistic | C=0.1 | 14 | 0.8073, 0.8081, 0.8000, 0.8228, 0.7992 | 0.8075 | 0.0095 | 0.3 |
| model_A_harmonized | logistic | C=1.0 | 14 | 0.8092, 0.8122, 0.8004, 0.8239, 0.8026 | 0.8096 | 0.0093 | 0.3 |
| model_A_harmonized | logistic | C=10.0 | 14 | 0.8090, 0.8125, 0.8001, 0.8235, 0.8029 | 0.8096 | 0.0092 | 0.3 |

Chosen per model set (frozen, fit on all train rows, threshold = Youden's J on train OOF):

- **model_A_full**: logistic {'C': 1.0} -> CV AUC 0.8270 (sd 0.0042); OOF AUC 0.8270; threshold 0.5461 (OOF sens 0.677, spec 0.858)
- **model_B_full**: hgb {'learning_rate': 0.05, 'max_depth': 6, 'max_iter': 100} -> CV AUC 0.8641 (sd 0.0076); OOF AUC 0.8639; threshold 0.1401 (OOF sens 0.727, spec 0.849)
- **baseline_egfr**: logistic {'C': 0.001} -> CV AUC 0.7605 (sd 0.0071); OOF AUC 0.7604; threshold 0.6603 (OOF sens 0.504, spec 0.948)
- **baseline_demographic**: logistic {'C': 10.0} -> CV AUC 0.7730 (sd 0.0079); OOF AUC 0.7727; threshold 0.5336 (OOF sens 0.682, spec 0.758)
- **model_A_harmonized**: logistic {'C': 1.0} -> CV AUC 0.8096 (sd 0.0093); OOF AUC 0.8095; threshold 0.5883 (OOF sens 0.636, spec 0.880)

**FINAL MODEL: model_B_full** (frozen before any test-set evaluation).

## Decision record (written by hand before any test-set evaluation)

- Label, cohort, feature set and temporal split were fixed in `src/nhanes/constants.py` before the first model was fitted (commits 87d6c40, d394240, add25c2).
- Selection rule, fixed in advance: the final model is the configuration with the highest mean 5-fold CV ROC-AUC on the 2005-2016 training cycles among Model A (logistic) and Model B (gradient boosting).
- Outcome of the rule: **model_B_full** (HistGradientBoosting, learning_rate 0.05, max_depth 6, max_iter 100; CV AUC 0.8641) beats model_A_full (logistic, C = 1.0; CV AUC 0.8270). Model A remains the interpretable secondary model and is reported alongside, not used for selection.
- Thresholds: Youden's J on train out-of-fold predictions (model_B_full 0.1401; model_A_full 0.5461; baselines and the harmonised model as listed above). The test set is never used to choose a threshold.
- Log-transform policy (log1p of creatinine, BUN, glucose, BUN:creatinine ratio) copies v1 `DEFAULT_LOG_COLS`; Model B is invariant to it.
- Model B uses no class weighting (ROC-AUC is unaffected); Model A keeps `class_weight="balanced"` as in v1, so its predicted probabilities are expected to run high (reported, not corrected).
- Configurations that did worse are kept in `cv_results.csv` and the table above.
- The 2017-Mar 2020 cycle is evaluated once by `scripts/nhanes_06_evaluate_test.py`, immediately after this note. Nothing above changes afterwards, whatever the result.

## Test evaluation 2026-10-08T07:57:47+00:00 - scripts/nhanes_06_evaluate_test.py

Single pass on the 2017-2020 cycle (n = 8,038) with the models frozen at 2026-10-08T07:54:10+00:00. Results in reports/nhanes/test_metrics.json. No model, feature, label, cohort or split decision was changed after this point.

## External validation 2026-10-08T07:58:03+00:00 - scripts/nhanes_07_external_uci.py

Frozen model_A_harmonized applied unchanged to the 399 UCI patients (eGFR as male primary, female sensitivity). Results in reports/nhanes/external_metrics.json.

## Post-hoc UCI diagnostics 2026-10-08T08:01:05+00:00 - scripts/nhanes_09_uci_diagnostics.py

Exploratory, not pre-registered, nothing retrained: the frozen harmonised Model A applied to UCI subsets to explain the external result. See reports/nhanes/external_diagnostics.json.

## v3 transfer model: pre-specification (written before the second look at UCI)

- Disclosure: UCI has been evaluated once (v2 harmonised Model A, AUC 0.641, see external_metrics.json).
  The post-hoc diagnosis (external_diagnostics.json) found two mechanisms: standardised rare-missingness
  indicators (-55 log-odds for a missing potassium) and extrapolation on hospital-range values (z up to 26).
- v3 changes, fixed now: (1) no missing-value indicators; (2) features clipped to the NHANES-train
  0.1%-99.9% range before scaling; (3) a gradient-boosting candidate on
  the same 14 shared features. Same 14 features, same log policy, same label, same NHANES train rows.
- Selection rule: highest mean 5-fold CV ROC-AUC on NHANES train among ['transfer_lr_v3', 'transfer_hgb_14'].
  UCI plays no part in selection. `transfer_lr_v3_noclip` (no indicators, no clipping) is an ablation, reported only.
- Thresholds: Youden's J on NHANES-train out-of-fold predictions.
- UCI is then evaluated exactly once more by scripts/nhanes_11_transfer_v3_evaluate.py; whatever the
  number is, it is reported next to the v2 number as a disclosed second look.

## v3 training run 2026-10-08T08:32:18+00:00 - scripts/nhanes_10_transfer_v3_train.py

| model_set | family | params | fold AUCs | mean AUC | sd |
|---|---|---|---|---:|---:|
| transfer_lr_v3 | transfer_logistic | C=0.001 | 0.7997, 0.7980, 0.7939, 0.8144, 0.7900 | 0.7992 | 0.0093 |
| transfer_lr_v3 | transfer_logistic | C=0.01 | 0.8022, 0.8013, 0.7963, 0.8178, 0.7934 | 0.8022 | 0.0095 |
| transfer_lr_v3 | transfer_logistic | C=0.1 | 0.8071, 0.8087, 0.8001, 0.8231, 0.7994 | 0.8077 | 0.0096 |
| transfer_lr_v3 | transfer_logistic | C=1.0 | 0.8087, 0.8125, 0.8002, 0.8243, 0.8026 | 0.8097 | 0.0095 |
| transfer_lr_v3 | transfer_logistic | C=10.0 | 0.8084, 0.8130, 0.7998, 0.8241, 0.8030 | 0.8097 | 0.0095 |
| transfer_lr_v3_noclip | transfer_logistic_noclip | C=0.001 | 0.7996, 0.7980, 0.7936, 0.8144, 0.7896 | 0.7990 | 0.0094 |
| transfer_lr_v3_noclip | transfer_logistic_noclip | C=0.01 | 0.8022, 0.8014, 0.7961, 0.8178, 0.7930 | 0.8021 | 0.0096 |
| transfer_lr_v3_noclip | transfer_logistic_noclip | C=0.1 | 0.8074, 0.8092, 0.8001, 0.8230, 0.7993 | 0.8078 | 0.0095 |
| transfer_lr_v3_noclip | transfer_logistic_noclip | C=1.0 | 0.8091, 0.8131, 0.8005, 0.8239, 0.8026 | 0.8098 | 0.0093 |
| transfer_lr_v3_noclip | transfer_logistic_noclip | C=10.0 | 0.8089, 0.8134, 0.8001, 0.8236, 0.8029 | 0.8098 | 0.0093 |
| transfer_hgb_14 | hgb | learning_rate=0.05, max_depth=3, max_iter=100 | 0.8504, 0.8529, 0.8376, 0.8516, 0.8444 | 0.8474 | 0.0064 |
| transfer_hgb_14 | hgb | learning_rate=0.05, max_depth=3, max_iter=300 | 0.8571, 0.8542, 0.8407, 0.8553, 0.8492 | 0.8513 | 0.0066 |
| transfer_hgb_14 | hgb | learning_rate=0.05, max_depth=6, max_iter=100 | 0.8554, 0.8547, 0.8411, 0.8537, 0.8479 | 0.8506 | 0.0061 |
| transfer_hgb_14 | hgb | learning_rate=0.05, max_depth=6, max_iter=300 | 0.8538, 0.8459, 0.8331, 0.8504, 0.8423 | 0.8451 | 0.0080 |
| transfer_hgb_14 | hgb | learning_rate=0.1, max_depth=3, max_iter=100 | 0.8553, 0.8549, 0.8410, 0.8542, 0.8483 | 0.8508 | 0.0061 |
| transfer_hgb_14 | hgb | learning_rate=0.1, max_depth=3, max_iter=300 | 0.8583, 0.8495, 0.8393, 0.8536, 0.8474 | 0.8496 | 0.0071 |
| transfer_hgb_14 | hgb | learning_rate=0.1, max_depth=6, max_iter=100 | 0.8537, 0.8511, 0.8373, 0.8520, 0.8456 | 0.8480 | 0.0067 |
| transfer_hgb_14 | hgb | learning_rate=0.1, max_depth=6, max_iter=300 | 0.8428, 0.8382, 0.8245, 0.8433, 0.8313 | 0.8360 | 0.0080 |

- **transfer_lr_v3** (secondary candidate): transfer_logistic {'C': 1.0} -> CV AUC 0.8097; threshold 0.5812 (OOF sens 0.642, spec 0.873)
- **transfer_hgb_14** (final): hgb {'learning_rate': 0.05, 'max_depth': 3, 'max_iter': 300} -> CV AUC 0.8513; threshold 0.1612 (OOF sens 0.681, spec 0.874)
- **transfer_lr_v3_noclip** (ablation): transfer_logistic_noclip {'C': 1.0} -> CV AUC 0.8098; threshold 0.5862 (OOF sens 0.637, spec 0.878)

**FINAL v3 transfer model: transfer_hgb_14** (frozen before the second UCI evaluation).

## v3 external evaluation 2026-10-08T08:33:03+00:00 - scripts/nhanes_11_transfer_v3_evaluate.py

Second, disclosed look at UCI with the models frozen at 2026-10-08T08:32:18+00:00. Results in external_metrics_v3.json; v2 first-look AUC 0.641 stays on record.

