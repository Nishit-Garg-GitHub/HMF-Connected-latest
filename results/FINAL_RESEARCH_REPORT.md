# HMF Electrooxidation to FDCA — Integrated ML Research Report

## Research architecture
Literature dataset (154 records) → cleaning and feature engineering → K-means (K=3,4,5,6) → fold-local class balancing (ROS, SMOTE, ADASYN) → classifier comparison → learning curves → SHAP/feature importance → joint four-output regression → validation-gated NSGA-II.

This report distinguishes paper-inspired methodology from the results obtained on the current literature dataset. It does not claim that screening scores are experimental validation.

## 1. Literature dataset and preprocessing
- Source records: **154**.
- Publication groups: **99**.
- Preprocessed descriptors: **53 columns before exclusions**.
- The feature matrix excludes all four measured targets and outcome-derived columns.
- The agreed experimental input groups are TEMPO, temperature, reactant monomer, HMF concentration, catalyst, electrolyte, potential vs RHE and cell type.
- Chemical/catalyst/electrolyte descriptors are represented by engineered numerical columns.

## 2. K-means clustering
| K | Records clustered | Cluster counts |
|---:|---:|---|
| 3 | 154 | `{0: 138, 1: 5, 2: 11}` |
| 4 | 154 | `{0: 137, 1: 3, 2: 4, 3: 10}` |
| 5 | 154 | `{0: 17, 1: 7, 2: 2, 3: 4, 4: 124}` |
| 6 | 154 | `{0: 127, 1: 2, 2: 4, 3: 6, 4: 14, 5: 1}` |

K-means is performed on conversion/yield outcome space for descriptive performance grouping. Cluster labels are never included as input features for the regression model.

## 3. Classification and sampling comparison
| target_kind    |   k | model            | sampler   |   macro_f1_mean |   macro_f1_sd |   macro_f1_ci95_lo |   macro_f1_ci95_hi |   accuracy_mean | leakage_flag   |
|:---------------|----:|:-----------------|:----------|----------------:|--------------:|-------------------:|-------------------:|----------------:|:---------------|
| kmeans_cluster |   4 | RandomForest     | ADASYN    |        0.477272 |    0.00663432 |           0.474533 |           0.48001  |        0.913333 | False          |
| kmeans_cluster |   4 | RandomForest     | SMOTE     |        0.470658 |    0.0308826  |           0.457911 |           0.483406 |        0.912    | False          |
| kmeans_cluster |   4 | SVM              | SMOTE     |        0.457908 |    0.0147503  |           0.451819 |           0.463997 |        0.846    | False          |
| kmeans_cluster |   4 | SVM              | ADASYN    |        0.457908 |    0.0147503  |           0.451819 |           0.463997 |        0.846    | False          |
| kmeans_cluster |   4 | GradientBoosting | SMOTE     |        0.437065 |    0.0854659  |           0.401786 |           0.472343 |        0.856667 | False          |
| kmeans_cluster |   4 | GradientBoosting | ADASYN    |        0.434113 |    0.114076   |           0.387025 |           0.481201 |        0.856    | False          |
| kmeans_cluster |   3 | GradientBoosting | ROS       |        0.420676 |    0.0694125  |           0.392024 |           0.449328 |        0.851169 | False          |
| kmeans_cluster |   3 | DecisionTree     | ROS       |        0.409536 |    0.078847   |           0.376989 |           0.442082 |        0.774026 | False          |
| kmeans_cluster |   3 | DecisionTree     | none      |        0.397042 |    0.0586473  |           0.372833 |           0.42125  |        0.737143 | False          |
| kmeans_cluster |   3 | NeuralNetwork    | SMOTE     |        0.388868 |    0.0700798  |           0.35994  |           0.417795 |        0.701299 | False          |
| kmeans_cluster |   3 | RandomForest     | ROS       |        0.378857 |    0.0494832  |           0.358431 |           0.399283 |        0.868571 | False          |
| kmeans_cluster |   3 | GradientBoosting | ADASYN    |        0.374069 |    0.0677442  |           0.346106 |           0.402033 |        0.847013 | False          |
| kmeans_cluster |   3 | NeuralNetwork    | ROS       |        0.372932 |    0.0666708  |           0.345411 |           0.400452 |        0.683896 | False          |
| kmeans_cluster |   3 | GradientBoosting | SMOTE     |        0.371335 |    0.0637667  |           0.345013 |           0.397656 |        0.85039  | False          |
| kmeans_cluster |   3 | NeuralNetwork    | ADASYN    |        0.366341 |    0.0598105  |           0.341652 |           0.391029 |        0.676623 | False          |
| kmeans_cluster |   3 | DecisionTree     | SMOTE     |        0.360607 |    0.0350251  |           0.34615  |           0.375065 |        0.830649 | False          |
| kmeans_cluster |   4 | DecisionTree     | SMOTE     |        0.359681 |    0.144789   |           0.299915 |           0.419447 |        0.824667 | False          |
| kmeans_cluster |   3 | DecisionTree     | ADASYN    |        0.355462 |    0.0363824  |           0.340445 |           0.37048  |        0.832987 | False          |
| kmeans_cluster |   4 | DecisionTree     | ADASYN    |        0.341503 |    0.105067   |           0.298133 |           0.384873 |        0.817333 | False          |
| kmeans_cluster |   3 | NeuralNetwork    | none      |        0.338957 |    0.0473746  |           0.319402 |           0.358513 |        0.786494 | False          |

Selected screening candidate: **RandomForest + ROS**, K=None, mean macro-F1=0.5863667179576847, 95% CI=[0.570, 0.603], dummy baseline mean macro-F1=0.4800, permutation test p-value=0.5545.
Samplers: no sampling, RandomOverSampler (ROS), SMOTE and ADASYN. Sampling occurs inside training folds only. Infeasible sampler/fold combinations are reported rather than silently replaced.
Evaluation uses pooled out-of-fold predictions per repeat across >=20 publication-grouped repeats before calculating macro-F1.
Scores above 0.8 are flagged for leakage audit. Predicted performance reflects unseen-paper generalization.

## 4. Fold-local balancing audit
|   k | sampler   | status     | before                                            | after                                                  | reason                                                                                       |
|----:|:----------|:-----------|:--------------------------------------------------|:-------------------------------------------------------|:---------------------------------------------------------------------------------------------|
|   3 | ROS       | ok         | {"0": 92, "1": 3, "2": 8}                         | {"0": 92, "1": 92, "2": 92}                            | nan                                                                                          |
|   3 | SMOTE     | ok         | {"0": 92, "1": 3, "2": 8}                         | {"0": 92, "1": 92, "2": 92}                            | nan                                                                                          |
|   3 | ADASYN    | ok         | {"0": 92, "1": 3, "2": 8}                         | {"0": 92, "1": 91, "2": 91}                            | nan                                                                                          |
|   3 | ROS       | ok         | {"0": 91, "1": 4, "2": 6}                         | {"0": 91, "1": 91, "2": 91}                            | nan                                                                                          |
|   3 | SMOTE     | ok         | {"0": 91, "1": 4, "2": 6}                         | {"0": 91, "1": 91, "2": 91}                            | nan                                                                                          |
|   3 | ADASYN    | ok         | {"0": 91, "1": 4, "2": 6}                         | {"0": 91, "1": 92, "2": 90}                            | nan                                                                                          |
|   3 | ROS       | ok         | {"0": 93, "1": 3, "2": 8}                         | {"0": 93, "1": 93, "2": 93}                            | nan                                                                                          |
|   3 | SMOTE     | ok         | {"0": 93, "1": 3, "2": 8}                         | {"0": 93, "1": 93, "2": 93}                            | nan                                                                                          |
|   3 | ADASYN    | ok         | {"0": 93, "1": 3, "2": 8}                         | {"0": 93, "1": 92, "2": 92}                            | nan                                                                                          |
|   4 | ROS       | ok         | {"0": 91, "1": 2, "2": 2, "3": 7}                 | {"0": 91, "1": 91, "2": 91, "3": 91}                   | nan                                                                                          |
|   4 | SMOTE     | infeasible | {"0": 91, "1": 2, "2": 2, "3": 7}                 | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   4 | ADASYN    | infeasible | {"0": 91, "1": 2, "2": 2, "3": 7}                 | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   4 | ROS       | ok         | {"0": 91, "1": 2, "2": 4, "3": 6}                 | {"0": 91, "1": 91, "2": 91, "3": 91}                   | nan                                                                                          |
|   4 | SMOTE     | infeasible | {"0": 91, "1": 2, "2": 4, "3": 6}                 | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   4 | ADASYN    | infeasible | {"0": 91, "1": 2, "2": 4, "3": 6}                 | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   4 | ROS       | ok         | {"0": 92, "1": 2, "2": 2, "3": 7}                 | {"0": 92, "1": 92, "2": 92, "3": 92}                   | nan                                                                                          |
|   4 | SMOTE     | infeasible | {"0": 92, "1": 2, "2": 2, "3": 7}                 | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   4 | ADASYN    | infeasible | {"0": 92, "1": 2, "2": 2, "3": 7}                 | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   5 | ROS       | ok         | {"0": 11, "1": 5, "2": 1, "3": 2, "4": 83}        | {"0": 83, "1": 83, "2": 83, "3": 83, "4": 83}          | nan                                                                                          |
|   5 | SMOTE     | infeasible | {"0": 11, "1": 5, "2": 1, "3": 2, "4": 83}        | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 1, n_samples = 1 |
|   5 | ADASYN    | infeasible | {"0": 11, "1": 5, "2": 1, "3": 2, "4": 83}        | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 1, n_samples = 1 |
|   5 | ROS       | ok         | {"0": 12, "1": 5, "2": 1, "3": 2, "4": 83}        | {"0": 83, "1": 83, "2": 83, "3": 83, "4": 83}          | nan                                                                                          |
|   5 | SMOTE     | infeasible | {"0": 12, "1": 5, "2": 1, "3": 2, "4": 83}        | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 1, n_samples = 1 |
|   5 | ADASYN    | infeasible | {"0": 12, "1": 5, "2": 1, "3": 2, "4": 83}        | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 1, n_samples = 1 |
|   5 | ROS       | ok         | {"0": 11, "1": 4, "2": 2, "3": 4, "4": 82}        | {"0": 82, "1": 82, "2": 82, "3": 82, "4": 82}          | nan                                                                                          |
|   5 | SMOTE     | infeasible | {"0": 11, "1": 4, "2": 2, "3": 4, "4": 82}        | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   5 | ADASYN    | infeasible | {"0": 11, "1": 4, "2": 2, "3": 4, "4": 82}        | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   6 | ROS       | ok         | {"0": 85, "1": 1, "2": 2, "3": 4, "4": 9, "5": 1} | {"0": 85, "1": 85, "2": 85, "3": 85, "4": 85, "5": 85} | nan                                                                                          |
|   6 | SMOTE     | infeasible | {"0": 85, "1": 1, "2": 2, "3": 4, "4": 9, "5": 1} | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 1, n_samples = 1 |
|   6 | ADASYN    | infeasible | {"0": 85, "1": 1, "2": 2, "3": 4, "4": 9, "5": 1} | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 1, n_samples = 1 |
|   6 | ROS       | ok         | {"0": 85, "1": 1, "2": 2, "3": 4, "4": 10}        | {"0": 85, "1": 85, "2": 85, "3": 85, "4": 85}          | nan                                                                                          |
|   6 | SMOTE     | infeasible | {"0": 85, "1": 1, "2": 2, "3": 4, "4": 10}        | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 1, n_samples = 1 |
|   6 | ADASYN    | infeasible | {"0": 85, "1": 1, "2": 2, "3": 4, "4": 10}        | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 1, n_samples = 1 |
|   6 | ROS       | ok         | {"0": 84, "1": 2, "2": 4, "3": 4, "4": 9, "5": 1} | {"0": 84, "1": 84, "2": 84, "3": 84, "4": 84, "5": 84} | nan                                                                                          |
|   6 | SMOTE     | infeasible | {"0": 84, "1": 2, "2": 4, "3": 4, "4": 9, "5": 1} | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   6 | ADASYN    | infeasible | {"0": 84, "1": 2, "2": 4, "3": 4, "4": 9, "5": 1} | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |

## 5. Learning curves and generalisation
- Valid paper-holdout learning-curve points: **8**.
- Mean unseen-paper macro-F1 across curve points: **0.620**.

## 6. SHAP / feature importance
| feature                           |   mean_abs_shap |
|:----------------------------------|----------------:|
| Electrolyte_Concentration_M       |       0.0759881 |
| Potential_vs_RHE_V                |       0.0520102 |
| Catalyst_Carbon                   |       0.0427714 |
| Catalyst_Pt                       |       0.0348987 |
| HMF_Concentration_mM              |       0.0269892 |
| Catalyst_Nanoparticle             |       0.0232085 |
| Catalyst_Hydroxide                |       0.0210265 |
| Three_Electrode                   |       0.0205726 |
| Catalyst_Co                       |       0.020419  |
| Catalyst_Oxyhydroxide             |       0.0183596 |
| Electrolyte_Electrolyte_Cation_Na |       0.0153606 |
| Potential_vs_RHE                  |       0.0135771 |
Feature importance is associational and should not be interpreted as causal evidence.

## 7. Multi-output regression
### Primary — Per-target regression (SelectKBest + ExtraTrees)
| target          |   model_r2_mean |   model_r2_sd |   model_mae_mean |   baseline_mae_mean |   model_rmse_mean |
|:----------------|----------------:|--------------:|-----------------:|--------------------:|------------------:|
| Conversion_pct  |      -2.06729   |      6.51992  |          6.27414 |             7.06885 |           13.3324 |
| FE_pct          |      -0.232508  |      0.798494 |          8.94814 |             9.94933 |           15.4255 |
| Selectivity_pct |      -0.0192654 |      0.593863 |         10.3554  |            12.3653  |           17.334  |
| Yield_pct       |      -0.0375266 |      0.129254 |         12.3534  |            13.1909  |           20.2965 |

### Secondary — Joint multi-output regression (complete rows only)
| target          |   model_r2_mean |   model_r2_sd |   model_mae_mean |   baseline_mae_mean |   model_rmse_mean |
|:----------------|----------------:|--------------:|-----------------:|--------------------:|------------------:|
| Conversion_pct  |      -56.8196   |     51.0622   |          3.91726 |             2.07617 |           11.4735 |
| FE_pct          |       -1.37152  |      1.66131  |          8.68344 |             6.886   |           14.8821 |
| Selectivity_pct |       -0.379789 |      0.569455 |          5.92782 |             5.3445  |           11.9639 |
| Yield_pct       |       -0.452306 |      0.694989 |          5.96523 |             5.29133 |           11.9438 |


Regression validation gate: **PASSED**.
Gate criteria: Repeated publication-grouped GroupKFold (10 repeats x 3 folds); MAE must beat median baseline and win a majority of folds for every target.
Per-target models train on all records with each target observed, preventing sample starvation from requiring all 4 targets simultaneously.

## 8. Genetic algorithm / NSGA-II
- Status: **completed_model_generated_candidates**.
- Model-generated Pareto candidates: **48**.
- NSGA-II is gated by the multi-output regression validation results.
- Any Pareto candidates are model-generated hypotheses and require experimental validation.

## 9. Reproducibility and limitations
- All four numerical outcomes are excluded from the input matrix.
- Publication-grouped validation is used to reduce leakage between records from the same paper.
- Imputation, variance filtering, scaling and resampling are fitted within training folds.
- Electrode potential vs RHE is parsed only from explicitly RHE-referenced records; cell voltage is kept separate and excluded from the agreed input feature set.
- Invalid percentage targets outside 0–100 are excluded from the strict four-output regression rather than clipped or fabricated.
- The dataset is small and literature-derived; missingness, heterogeneous reporting and publication bias limit generalisation.
- A favourable F1 or R² does not establish causal catalyst effects.
- NSGA-II outputs must be confirmed experimentally before being described as optimal catalysts or operating conditions.
