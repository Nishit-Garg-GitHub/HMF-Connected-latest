# HMF Electrooxidation to FDCA — Integrated ML Research Report

## Research architecture
Literature dataset (154 records) → cleaning and feature engineering → K-means (K=3,4,5,6) → fold-local class balancing (ROS, SMOTE, ADASYN) → classifier comparison → learning curves → SHAP/feature importance → per-target regression with validation gate → regression SHAP and literature check → hydroxide-restricted NSGA-II.

This report distinguishes paper-inspired methodology from the results obtained on the current literature dataset. It does not claim that screening scores are experimental validation.

## 1. Literature dataset and preprocessing
- Source records: **154**.
- Publication groups: **99**.
- Preprocessed descriptors: **67 columns before exclusions**.
- The feature matrix excludes all four measured targets and outcome-derived columns.
- The agreed experimental input groups are TEMPO, temperature, reactant monomer, HMF concentration, catalyst, electrolyte, potential vs RHE and cell type.
- Chemical/catalyst/electrolyte descriptors are represented by engineered numerical columns.

## 2. K-means clustering
| K | Records clustered | Cluster counts |
|---:|---:|---|
| 3 | 111 | `{0: 97, 1: 5, 2: 9}` |
| 4 | 111 | `{0: 97, 1: 1, 2: 4, 3: 9}` |
| 5 | 111 | `{0: 86, 1: 4, 2: 14, 3: 6, 4: 1}` |
| 6 | 111 | `{0: 81, 1: 3, 2: 4, 3: 16, 4: 1, 5: 6}` |

K-means is performed on conversion/yield outcome space for descriptive performance grouping. Cluster labels are never included as input features for the regression model.

## 3. Classification and sampling comparison
| target_kind    |   k | model            | sampler   |   macro_f1_mean |   macro_f1_sd |   macro_f1_ci95_lo |   macro_f1_ci95_hi |   accuracy_mean | leakage_flag   |
|:---------------|----:|:-----------------|:----------|----------------:|--------------:|-------------------:|-------------------:|----------------:|:---------------|
| kmeans_cluster |   3 | GradientBoosting | ROS       |        0.658038 |     0.0598963 |           0.633314 |           0.682762 |        0.891532 | False          |
| kmeans_cluster |   3 | GradientBoosting | none      |        0.638451 |     0.0582192 |           0.614419 |           0.662482 |        0.902342 | False          |
| kmeans_cluster |   3 | DecisionTree     | ROS       |        0.616435 |     0.0626346 |           0.59058  |           0.642289 |        0.853694 | False          |
| kmeans_cluster |   3 | DecisionTree     | none      |        0.581442 |     0.0659339 |           0.554225 |           0.608658 |        0.841802 | False          |
| kmeans_cluster |   3 | SVM              | none      |        0.568749 |     0.0296056 |           0.556528 |           0.580969 |        0.863063 | False          |
| kmeans_cluster |   3 | Ensemble         | ROS       |        0.550924 |     0.0353224 |           0.536344 |           0.565505 |        0.882883 | False          |
| kmeans_cluster |   3 | GradientBoosting | SMOTE     |        0.549304 |     0.0467738 |           0.529997 |           0.568611 |        0.892973 | False          |
| kmeans_cluster |   3 | Ensemble         | none      |        0.543893 |     0.028843  |           0.531987 |           0.555799 |        0.879279 | False          |
| kmeans_cluster |   3 | GradientBoosting | ADASYN    |        0.542368 |     0.0463217 |           0.523248 |           0.561489 |        0.893333 | False          |
| kmeans_cluster |   3 | Ensemble         | ADASYN    |        0.541134 |     0.0311639 |           0.52827  |           0.553998 |        0.881802 | False          |
| kmeans_cluster |   3 | Ensemble         | SMOTE     |        0.534013 |     0.0366964 |           0.518866 |           0.549161 |        0.881441 | False          |
| kmeans_cluster |   3 | NeuralNetwork    | SMOTE     |        0.5221   |     0.0496531 |           0.501604 |           0.542596 |        0.853694 | False          |
| kmeans_cluster |   3 | NeuralNetwork    | ROS       |        0.519653 |     0.0538924 |           0.497407 |           0.541898 |        0.86018  | False          |
| kmeans_cluster |   3 | NeuralNetwork    | ADASYN    |        0.519063 |     0.0381273 |           0.503325 |           0.534801 |        0.850811 | False          |
| kmeans_cluster |   3 | SVM              | ROS       |        0.509285 |     0.0606824 |           0.484236 |           0.534333 |        0.870991 | False          |
| kmeans_cluster |   3 | DecisionTree     | ADASYN    |        0.489624 |     0.0658764 |           0.462432 |           0.516817 |        0.851532 | False          |
| kmeans_cluster |   3 | NeuralNetwork    | none      |        0.486556 |     0.050749  |           0.465607 |           0.507504 |        0.876396 | False          |
| kmeans_cluster |   3 | DecisionTree     | SMOTE     |        0.485907 |     0.0698417 |           0.457078 |           0.514737 |        0.860901 | False          |
| kmeans_cluster |   3 | RandomForest     | ROS       |        0.479552 |     0.0807917 |           0.446203 |           0.512901 |        0.886126 | False          |
| kmeans_cluster |   3 | SVM              | SMOTE     |        0.479262 |     0.0495151 |           0.458823 |           0.4997   |        0.868108 | False          |

Selected screening candidate: **GradientBoosting + ROS**, K=3, mean macro-F1=0.65803834684149, 95% CI=[0.633, 0.683], dummy baseline mean macro-F1=0.3280, permutation test p-value=0.1386.
Samplers: no sampling, RandomOverSampler (ROS), SMOTE and ADASYN. Sampling occurs inside training folds only. Infeasible sampler/fold combinations are reported rather than silently replaced.
Evaluation uses pooled out-of-fold predictions per repeat across >=20 publication-grouped repeats before calculating macro-F1.
Scores above 0.8 are flagged for leakage audit. Predicted performance reflects unseen-paper generalization.

## 4. Fold-local balancing audit
|   k | sampler   | status     | before                                             | after                                                  | reason                                                                                       |
|----:|:----------|:-----------|:---------------------------------------------------|:-------------------------------------------------------|:---------------------------------------------------------------------------------------------|
|   3 | ROS       | ok         | {"0": 64, "1": 3, "2": 5}                          | {"0": 64, "1": 64, "2": 64}                            | nan                                                                                          |
|   3 | SMOTE     | ok         | {"0": 64, "1": 3, "2": 5}                          | {"0": 64, "1": 64, "2": 64}                            | nan                                                                                          |
|   3 | ADASYN    | ok         | {"0": 64, "1": 3, "2": 5}                          | {"0": 64, "1": 63, "2": 63}                            | nan                                                                                          |
|   3 | ROS       | ok         | {"0": 65, "1": 3, "2": 7}                          | {"0": 65, "1": 65, "2": 65}                            | nan                                                                                          |
|   3 | SMOTE     | ok         | {"0": 65, "1": 3, "2": 7}                          | {"0": 65, "1": 65, "2": 65}                            | nan                                                                                          |
|   3 | ADASYN    | ok         | {"0": 65, "1": 3, "2": 7}                          | {"0": 65, "1": 66, "2": 65}                            | nan                                                                                          |
|   3 | ROS       | ok         | {"0": 65, "1": 4, "2": 6}                          | {"0": 65, "1": 65, "2": 65}                            | nan                                                                                          |
|   3 | SMOTE     | ok         | {"0": 65, "1": 4, "2": 6}                          | {"0": 65, "1": 65, "2": 65}                            | nan                                                                                          |
|   3 | ADASYN    | ok         | {"0": 65, "1": 4, "2": 6}                          | {"0": 65, "1": 64, "2": 66}                            | nan                                                                                          |
|   4 | ROS       | ok         | {"0": 64, "1": 1, "2": 3, "3": 6}                  | {"0": 64, "1": 64, "2": 64, "3": 64}                   | nan                                                                                          |
|   4 | SMOTE     | infeasible | {"0": 64, "1": 1, "2": 3, "3": 6}                  | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 1, n_samples = 1 |
|   4 | ADASYN    | infeasible | {"0": 64, "1": 1, "2": 3, "3": 6}                  | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 1, n_samples = 1 |
|   4 | ROS       | ok         | {"0": 65, "2": 2, "3": 6}                          | {"0": 65, "2": 65, "3": 65}                            | nan                                                                                          |
|   4 | SMOTE     | infeasible | {"0": 65, "2": 2, "3": 6}                          | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   4 | ADASYN    | infeasible | {"0": 65, "2": 2, "3": 6}                          | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   4 | ROS       | ok         | {"0": 65, "1": 1, "2": 3, "3": 6}                  | {"0": 65, "1": 65, "2": 65, "3": 65}                   | nan                                                                                          |
|   4 | SMOTE     | infeasible | {"0": 65, "1": 1, "2": 3, "3": 6}                  | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 1, n_samples = 1 |
|   4 | ADASYN    | infeasible | {"0": 65, "1": 1, "2": 3, "3": 6}                  | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 1, n_samples = 1 |
|   5 | ROS       | ok         | {"0": 57, "1": 2, "2": 10, "3": 4, "4": 1}         | {"0": 57, "1": 57, "2": 57, "3": 57, "4": 57}          | nan                                                                                          |
|   5 | SMOTE     | infeasible | {"0": 57, "1": 2, "2": 10, "3": 4, "4": 1}         | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   5 | ADASYN    | infeasible | {"0": 57, "1": 2, "2": 10, "3": 4, "4": 1}         | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   5 | ROS       | ok         | {"0": 58, "1": 3, "2": 9, "3": 4}                  | {"0": 58, "1": 58, "2": 58, "3": 58}                   | nan                                                                                          |
|   5 | SMOTE     | ok         | {"0": 58, "1": 3, "2": 9, "3": 4}                  | {"0": 58, "1": 58, "2": 58, "3": 58}                   | nan                                                                                          |
|   5 | ADASYN    | ok         | {"0": 58, "1": 3, "2": 9, "3": 4}                  | {"0": 58, "1": 59, "2": 61, "3": 60}                   | nan                                                                                          |
|   5 | ROS       | ok         | {"0": 57, "1": 3, "2": 9, "3": 4, "4": 1}          | {"0": 57, "1": 57, "2": 57, "3": 57, "4": 57}          | nan                                                                                          |
|   5 | SMOTE     | infeasible | {"0": 57, "1": 3, "2": 9, "3": 4, "4": 1}          | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 1, n_samples = 1 |
|   5 | ADASYN    | infeasible | {"0": 57, "1": 3, "2": 9, "3": 4, "4": 1}          | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 1, n_samples = 1 |
|   6 | ROS       | ok         | {"0": 54, "1": 2, "2": 2, "3": 10, "4": 1, "5": 4} | {"0": 54, "1": 54, "2": 54, "3": 54, "4": 54, "5": 54} | nan                                                                                          |
|   6 | SMOTE     | infeasible | {"0": 54, "1": 2, "2": 2, "3": 10, "4": 1, "5": 4} | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   6 | ADASYN    | infeasible | {"0": 54, "1": 2, "2": 2, "3": 10, "4": 1, "5": 4} | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   6 | ROS       | ok         | {"0": 54, "1": 2, "2": 3, "3": 11, "4": 1, "5": 4} | {"0": 54, "1": 54, "2": 54, "3": 54, "4": 54, "5": 54} | nan                                                                                          |
|   6 | SMOTE     | infeasible | {"0": 54, "1": 2, "2": 3, "3": 11, "4": 1, "5": 4} | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   6 | ADASYN    | infeasible | {"0": 54, "1": 2, "2": 3, "3": 11, "4": 1, "5": 4} | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   6 | ROS       | ok         | {"0": 54, "1": 2, "2": 3, "3": 11, "5": 4}         | {"0": 54, "1": 54, "2": 54, "3": 54, "5": 54}          | nan                                                                                          |
|   6 | SMOTE     | infeasible | {"0": 54, "1": 2, "2": 3, "3": 11, "5": 4}         | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   6 | ADASYN    | infeasible | {"0": 54, "1": 2, "2": 3, "3": 11, "5": 4}         | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |

## 5. Learning curves and generalisation
- Unseen-paper macro-F1 at the full training fraction (stage 05, publication-grouped holdouts):

| K | Model | Validation macro-F1 | Training macro-F1 |
|---|---|---|---|
| 3 | DecisionTree | 0.302 | 0.941 |
| 3 | NeuralNetwork | 0.510 | 0.953 |
| 3 | RandomForest | 0.317 | 0.948 |
| 3 | SVM | 0.539 | 0.806 |
| 4 | DecisionTree | 0.466 | 0.754 |
| 4 | NeuralNetwork | 0.330 | 0.913 |
| 4 | RandomForest | 0.236 | 0.927 |
| 4 | SVM | 0.294 | 0.681 |
| 5 | DecisionTree | 0.437 | 0.621 |
| 5 | NeuralNetwork | 0.297 | 0.924 |
| 5 | RandomForest | 0.250 | 0.930 |
| 5 | SVM | 0.226 | 0.720 |

## 6. SHAP / feature importance
| feature                           |   importance_mean |   importance_sd |
|:----------------------------------|------------------:|----------------:|
| Electrolyte_Electrolyte_Anion_OH  |         0.170191  |       0.081994  |
| Alkaline_Electrolyte              |         0.170191  |       0.081994  |
| Electrolyte_Electrolyte_Anion_SO4 |         0.0956229 |       0         |
| Electrolyte_Electrolyte_Anion_nan |         0.0688893 |       0.040569  |
| TEMPO_binary                      |         0.0688893 |       0.040569  |
| Electrolyte_pH                    |         0.0637486 |       0.0450771 |
| Temperature_C                     |         0.0318743 |       0.0450771 |
| Catalyst_N_Elements               |         0.0318743 |       0.0450771 |
| Catalyst_Phosphide                |         0         |       0         |
| Catalyst_Selenide                 |         0         |       0         |
| Catalyst_Sulfide                  |         0         |       0         |
| Catalyst_Nitride                  |         0         |       0         |
Feature importance is associational and should not be interpreted as causal evidence.

## 7. Multi-output regression
### Primary — Per-target regression (Huber gradient boosting + extra-trees)
| target          |   n_records |   model_r2_mean |   model_r2_sd |   model_mae_mean |   baseline_mae_mean |   model_rmse_mean |
|:----------------|------------:|----------------:|--------------:|-----------------:|--------------------:|------------------:|
| Conversion_pct  |         117 |        0.185068 |     0.0653342 |          5.07135 |             4.55377 |           10.8802 |
| FE_pct          |         119 |        0.389214 |     0.0985214 |          7.37726 |             8.56711 |           11.725  |
| Selectivity_pct |          91 |        0.301103 |     0.084978  |          8.6483  |             9.47843 |           14.3406 |
| Yield_pct       |         132 |        0.443472 |     0.069532  |          8.72853 |            11.1237  |           15.0996 |

### Secondary — Joint multi-output regression (complete rows only)
| target          |   model_r2_mean |   model_r2_sd |   model_mae_mean |   baseline_mae_mean |   model_rmse_mean |
|:----------------|----------------:|--------------:|-----------------:|--------------------:|------------------:|
| Conversion_pct  |      -1.76276   |      3.65406  |          3.40973 |             3.3148  |           6.61856 |
| FE_pct          |       0.0958719 |      0.665223 |          6.43742 |             7.88615 |           9.43555 |
| Selectivity_pct |      -0.0577629 |      0.748444 |          5.81977 |             6.25713 |           8.9844  |
| Yield_pct       |       0.1449    |      0.851276 |          5.19021 |             6.9818  |           8.40384 |


Regression validation gate: **PASSED**.
Gate criteria: Repeated publication-grouped GroupKFold (10 repeats x 5 folds). For each required target (Yield_pct, FE_pct) the mean MAE must beat the training-fold MEDIAN baseline, win a majority of folds and have mean pooled out-of-fold R2 >= 0.0. Other targets are informational.
Per-target models train on all records with each target observed, preventing sample starvation from requiring all 4 targets simultaneously.

## 8. Genetic algorithm / NSGA-II
- Status: **completed_model_generated_candidates**.
- Model-generated Pareto candidates: **99**.
- NSGA-II runs only if the regressors for its objectives (yield and FE) beat the median baseline (MAE, fold wins, pooled R2 >= 0). Conversion is not an objective because it fails this test.
- Any Pareto candidates are model-generated hypotheses and require experimental validation.

### Constrained scenario: Ni(OH)2 / Cu(OH)2 anode (top 10)
- Conditions bounded to the range reported for real Ni/Cu hydroxide anodes: {'Potential_vs_RHE_V': [1.4, 1.69], 'Temperature_C': [25.0, 40.0]}.
|   rank | catalyst   | support   | electrolyte   |   electrolyte_M |   HMF_mM |   potential_V_RHE |   temperature_C |   predicted_Yield_pct |   member_spread_Yield_pct |   predicted_FE_pct |   observed_Yield_pct |
|-------:|:-----------|:----------|:--------------|----------------:|---------:|------------------:|----------------:|----------------------:|--------------------------:|-------------------:|---------------------:|
|      1 | Ni(OH)2    | foam      | KOH           |            1.16 |     10   |             1.4   |            25   |                97.809 |                      0.27 |             97.162 |                 88.3 |
|      2 | Ni(OH)2    | foam      | KOH           |            1.97 |     10.3 |             1.402 |            25   |                97.705 |                      0.48 |             97.351 |                 88.3 |
|      3 | Ni(OH)2    | carbon    | KOH           |            1.27 |     10   |             1.4   |            25   |                97.811 |                      0.26 |             96.302 |                 88.3 |
|      4 | Ni(OH)2    | carbon    | KOH           |            1.07 |     10   |             1.4   |            25   |                97.829 |                      0.23 |             96.282 |                 88.3 |
|      5 | Ni(OH)2    | foam      | KOH           |            2.98 |     13   |             1.4   |            25   |                96.985 |                      0.23 |             97.353 |                 88.3 |
|      6 | Ni(OH)2    | foam      | KOH           |            1.49 |     10.1 |             1.406 |            25   |                97.747 |                      0.39 |             97.37  |                 88.3 |
|      7 | Ni(OH)2    | foam      | KOH           |            2.58 |     11.3 |             1.408 |            25   |                97.71  |                      0.47 |             97.37  |                 88.3 |
|      8 | Ni(OH)2    | foam      | KOH           |            2.32 |     11.4 |             1.4   |            25.1 |                97.754 |                      0.38 |             97.365 |                 88.3 |
|      9 | Ni(OH)2    | foam      | KOH           |            2.04 |     10.5 |             1.404 |            25.1 |                97.759 |                      0.37 |             97.206 |                 88.3 |
|     10 | Ni(OH)2    | foam      | KOH           |            2.65 |     10.1 |             1.409 |            25   |                97.781 |                      0.32 |             97.185 |                 88.3 |

Nearest observed hydroxide record: Ni(OH)2/NF (1.0 M KOH | 1.45 V vs. RHE), measured yield 88.3%, FE 90.6%.

### Top 10 — Cu(OH)2 anode
|   rank | catalyst   | support   | electrolyte   |   electrolyte_M |   HMF_mM |   potential_V_RHE |   temperature_C |   predicted_Yield_pct |   member_spread_Yield_pct |   predicted_FE_pct |   observed_Yield_pct |
|-------:|:-----------|:----------|:--------------|----------------:|---------:|------------------:|----------------:|----------------------:|--------------------------:|-------------------:|---------------------:|
|      1 | Cu(OH)2    | foam      | KOH           |            1.11 |     11.4 |             1.4   |            25   |                91.053 |                      7.81 |             97.223 |                 93.3 |
|      2 | Cu(OH)2    | foam      | KOH           |            1.37 |     11.6 |             1.401 |            25   |                90.598 |                      8.2  |             97.071 |                 93.3 |
|      3 | Cu(OH)2    | foam      | KOH           |            1.37 |      7.3 |             1.402 |            25   |                91.148 |                      7.62 |             96.867 |                 93.3 |
|      4 | Cu(OH)2    | foam      | KOH           |            1.44 |    138.8 |             1.404 |            25   |                89.723 |                      9.16 |             97.136 |                 93.3 |
|      5 | Cu(OH)2    | foam      | KOH           |            1.68 |     31.6 |             1.405 |            25   |                89.728 |                      9.15 |             97.155 |                 93.3 |
|      6 | Cu(OH)2    | foam      | KOH           |            1.45 |      7.2 |             1.404 |            25.1 |                91.199 |                      7.52 |             96.717 |                 93.3 |
|      7 | Cu(OH)2    | foam      | KOH           |            1.09 |     13.6 |             1.408 |            25.1 |                90.67  |                      7.8  |             97.242 |                 93.3 |
|      8 | Cu(OH)2    | foam      | KOH           |            1.73 |     12.3 |             1.41  |            25   |                90.963 |                      9.12 |             97.259 |                 93.3 |
|      9 | Cu(OH)2    | foam      | KOH           |            1.48 |     12.2 |             1.407 |            25.1 |                90.549 |                      8.29 |             97.246 |                 93.3 |
|     10 | Cu(OH)2    | foam      | KOH           |            1.38 |     11   |             1.402 |            25.2 |                91.053 |                      7.81 |             97.229 |                 93.3 |

### Top 10 — Ni-Cu hydroxide anode
|   rank | catalyst        | support   | electrolyte   |   electrolyte_M |   HMF_mM |   potential_V_RHE |   temperature_C |   predicted_Yield_pct |   member_spread_Yield_pct |   predicted_FE_pct |   observed_Yield_pct |
|-------:|:----------------|:----------|:--------------|----------------:|---------:|------------------:|----------------:|----------------------:|--------------------------:|-------------------:|---------------------:|
|      1 | Ni-Cu hydroxide | foam      | KOH           |            0.95 |     10.1 |             1.4   |            25   |                97.025 |                      1.24 |             96.415 |                 93.3 |
|      2 | Ni-Cu hydroxide | foam      | KOH           |            0.95 |     10.1 |             1.4   |            25   |                97.201 |                      0.89 |             95.982 |                 93.3 |
|      3 | Ni-Cu hydroxide | foam      | KOH           |            2.55 |     15.1 |             1.4   |            25   |                96.329 |                      0.48 |             96.657 |                 93.3 |
|      4 | Ni-Cu hydroxide | foam      | KOH           |            2.44 |     14.3 |             1.4   |            25.1 |                96.464 |                      0.75 |             96.657 |                 93.3 |
|      5 | Ni-Cu hydroxide | foam      | KOH           |            0.95 |     10.1 |             1.401 |            25.1 |                96.933 |                      1.43 |             96.67  |                 93.3 |
|      6 | Ni-Cu hydroxide | foam      | KOH           |            2.15 |     15.7 |             1.405 |            25   |                96.324 |                      0.49 |             96.673 |                 93.3 |
|      7 | Ni-Cu hydroxide | foam      | KOH           |            0.92 |     10.1 |             1.406 |            25   |                97.001 |                      1.29 |             96.437 |                 93.3 |
|      8 | Ni-Cu hydroxide | foam      | KOH           |            2.69 |     13.9 |             1.402 |            25.1 |                96.458 |                      0.76 |             96.662 |                 93.3 |
|      9 | Ni-Cu hydroxide | foam      | KOH           |            0.94 |     10.1 |             1.407 |            25   |                97.019 |                      1.26 |             96.45  |                 93.3 |
|     10 | Ni-Cu hydroxide | foam      | KOH           |            2.24 |     12.3 |             1.408 |            25   |                96.832 |                      1.63 |             96.679 |                 93.3 |

### Robust ranking (E >= 1.45 V vs RHE)
|   rank | catalyst        | support   | electrolyte   |   electrolyte_M |   HMF_mM |   potential_V_RHE |   temperature_C |   predicted_Yield_pct |   member_spread_Yield_pct |   predicted_FE_pct |   observed_Yield_pct |
|-------:|:----------------|:----------|:--------------|----------------:|---------:|------------------:|----------------:|----------------------:|--------------------------:|-------------------:|---------------------:|
|      1 | Ni(OH)2         | foam      | KOH           |            2.69 |     12.1 |             1.45  |            25   |                97.567 |                      0.09 |             97.576 |                 88.3 |
|      2 | Ni(OH)2         | foam      | KOH           |            2.97 |     12.4 |             1.451 |            25   |                97.213 |                      0.27 |             97.602 |                 88.3 |
|      3 | Ni(OH)2         | foam      | KOH           |            2.9  |     11.7 |             1.454 |            25   |                97.273 |                      0.15 |             97.592 |                 88.3 |
|      4 | Ni(OH)2         | foam      | KOH           |            2.97 |     11.2 |             1.454 |            25   |                97.273 |                      0.15 |             97.597 |                 88.3 |
|      5 | Ni(OH)2         | foam      | KOH           |            2.91 |     12.5 |             1.451 |            25   |                96.871 |                      0.67 |             97.592 |                 88.3 |
|      6 | Ni(OH)2         | foam      | KOH           |            2.75 |     11.9 |             1.452 |            25.1 |                97.273 |                      0.15 |             97.592 |                 88.3 |
|      7 | Ni(OH)2         | foam      | KOH           |            2.23 |     11.4 |             1.451 |            25.1 |                97.276 |                      0.14 |             97.592 |                 88.3 |
|      8 | Ni-Cu hydroxide | foam      | KOH           |            2.52 |     10.2 |             1.45  |            25   |                96.896 |                      1.43 |             96.885 |                 93.3 |
|      9 | Ni-Cu hydroxide | foam      | KOH           |            2.42 |     10.1 |             1.451 |            25   |                97.033 |                      1.15 |             96.213 |                 93.3 |
|     10 | Ni-Cu hydroxide | foam      | KOH           |            1.43 |     10.1 |             1.451 |            25   |                97.141 |                      0.94 |             96.101 |                 93.3 |

### Regression SHAP versus literature
| factor                              | target    | expected   | data_direction   |   importance_rank | verdict             |
|:------------------------------------|:----------|:-----------|:-----------------|------------------:|:--------------------|
| Ni-containing anode (Ni(OH)2/NiOOH) | Yield_pct | +          | +                |                 2 | consistent          |
| Ni-containing anode (Ni(OH)2/NiOOH) | FE_pct    | +          | +                |                20 | not distinguishable |
| Hydroxide / oxyhydroxide phase      | Yield_pct | +          | +                |                29 | not distinguishable |
| Hydroxide / oxyhydroxide phase      | FE_pct    | +          | -                |                13 | not distinguishable |
| Cu-containing anode (Cu(OH)2/CuOOH) | Yield_pct | +          | -                |                27 | not distinguishable |
| Cu-containing anode (Cu(OH)2/CuOOH) | FE_pct    | +          | -                |                32 | not distinguishable |
| Alkaline electrolyte                | Yield_pct | +          | +                |                11 | not distinguishable |
| Alkaline electrolyte                | FE_pct    | +          | +                |                11 | not distinguishable |
| HMF concentration                   | Yield_pct | -          | -                |                 8 | consistent          |

## 9. Reproducibility and limitations
- All four numerical outcomes are excluded from the input matrix.
- Publication-grouped validation is used to reduce leakage between records from the same paper.
- Imputation, variance filtering, scaling and resampling are fitted within training folds.
- Electrode potential vs RHE is parsed only from explicitly RHE-referenced records; cell voltage is kept separate and excluded from the agreed input feature set.
- Invalid percentage targets outside 0–100 are excluded from the strict four-output regression rather than clipped or fabricated.
- The dataset is small and literature-derived; missingness, heterogeneous reporting and publication bias limit generalisation.
- A favourable F1 or R² does not establish causal catalyst effects.
- NSGA-II outputs must be confirmed experimentally before being described as optimal catalysts or operating conditions.
