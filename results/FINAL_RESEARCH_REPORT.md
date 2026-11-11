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
| 3 | 102 | `{0: 90, 1: 3, 2: 9}` |
| 4 | 102 | `{0: 90, 1: 2, 2: 1, 3: 9}` |
| 5 | 102 | `{0: 80, 1: 6, 2: 1, 3: 13, 4: 2}` |
| 6 | 102 | `{0: 69, 1: 3, 2: 1, 3: 2, 4: 6, 5: 21}` |

K-means is performed on conversion/yield outcome space for descriptive performance grouping. Cluster labels are never included as input features for the regression model.

## 3. Classification and sampling comparison
| target_kind    |   k | model         | sampler   |   n_valid |   macro_f1_mean |   macro_f1_sd |   weighted_f1_mean |   accuracy_mean |   balanced_accuracy_mean |
|:---------------|----:|:--------------|:----------|----------:|----------------:|--------------:|-------------------:|----------------:|-------------------------:|
| kmeans_cluster |   4 | RandomForest  | ROS       |         6 |        0.481139 |    0.21502    |           0.861863 |        0.877863 |                 0.459259 |
| kmeans_cluster |   3 | DecisionTree  | ROS       |         6 |        0.476831 |    0.0930677  |           0.848332 |        0.81813  |                 0.509099 |
| kmeans_cluster |   4 | SVM           | none      |         6 |        0.436086 |    0.0845948  |           0.863732 |        0.897173 |                 0.431481 |
| kmeans_cluster |   3 | RandomForest  | ROS       |         6 |        0.434775 |    0.105886   |           0.867123 |        0.886701 |                 0.425862 |
| kmeans_cluster |   4 | SVM           | ROS       |         6 |        0.431413 |    0.0847675  |           0.860185 |        0.892271 |                 0.42963  |
| kmeans_cluster |   4 | DecisionTree  | ROS       |         6 |        0.426993 |    0.177742   |           0.838435 |        0.828397 |                 0.483796 |
| kmeans_cluster |   3 | SVM           | ROS       |         6 |        0.417748 |    0.0830373  |           0.860609 |        0.892032 |                 0.411107 |
| kmeans_cluster |   3 | SVM           | none      |         6 |        0.395551 |    0.0915969  |           0.854787 |        0.886982 |                 0.392588 |
| kmeans_cluster |   3 | DecisionTree  | none      |         6 |        0.370354 |    0.0957569  |           0.799016 |        0.784206 |                 0.405156 |
| kmeans_cluster |   3 | RandomForest  | none      |         6 |        0.350116 |    0.0566656  |           0.840338 |        0.87759  |                 0.360181 |
| kmeans_cluster |   4 | DecisionTree  | none      |         6 |        0.341051 |    0.0690206  |           0.77911  |        0.743562 |                 0.40463  |
| kmeans_cluster |   3 | NeuralNetwork | ROS       |         6 |        0.338126 |    0.0649634  |           0.718617 |        0.67032  |                 0.388261 |
| kmeans_cluster |   4 | RandomForest  | none      |         6 |        0.320522 |    0.0702664  |           0.816802 |        0.857652 |                 0.335185 |
| kmeans_cluster |   3 | NeuralNetwork | none      |         6 |        0.291535 |    0.00733349 |           0.771995 |        0.769903 |                 0.290895 |
| kmeans_cluster |   6 | RandomForest  | ROS       |         6 |        0.286485 |    0.0781774  |           0.594409 |        0.622642 |                 0.314683 |
| kmeans_cluster |   6 | DecisionTree  | ROS       |         6 |        0.286105 |    0.0667777  |           0.5627   |        0.561879 |                 0.333696 |
| kmeans_cluster |   4 | NeuralNetwork | ROS       |         6 |        0.284087 |    0.0821442  |           0.746888 |        0.691537 |                 0.377778 |
| kmeans_cluster |   5 | RandomForest  | ROS       |         6 |        0.275806 |    0.0703074  |           0.690038 |        0.710784 |                 0.295758 |
| kmeans_cluster |   4 | NeuralNetwork | none      |         6 |        0.27015  |    0.0549129  |           0.766815 |        0.759159 |                 0.328241 |
| kmeans_cluster |   6 | RandomForest  | none      |         6 |        0.264726 |    0.116583   |           0.58654  |        0.637488 |                 0.282005 |

Selected screening candidate: **DecisionTree + SMOTE**, K=None, mean macro-F1=0.6285509398934352.
Samplers: no sampling, RandomOverSampler (ROS), SMOTE and ADASYN. Sampling occurs inside training folds only. Infeasible sampler/fold combinations are reported rather than silently replaced.
These are screening results. A high F1 is not guaranteed; model choice should be based on repeated unseen-paper performance and class support, not a single best fold.

## 4. Fold-local balancing audit
|   k | sampler   | status     | before                                             | after                                                  | reason                                                                                       |
|----:|:----------|:-----------|:---------------------------------------------------|:-------------------------------------------------------|:---------------------------------------------------------------------------------------------|
|   3 | ROS       | ok         | {"0": 60, "1": 2, "2": 6}                          | {"0": 60, "1": 60, "2": 60}                            | nan                                                                                          |
|   3 | SMOTE     | infeasible | {"0": 60, "1": 2, "2": 6}                          | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   3 | ADASYN    | infeasible | {"0": 60, "1": 2, "2": 6}                          | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   3 | ROS       | ok         | {"0": 61, "1": 2, "2": 6}                          | {"0": 61, "1": 61, "2": 61}                            | nan                                                                                          |
|   3 | SMOTE     | infeasible | {"0": 61, "1": 2, "2": 6}                          | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   3 | ADASYN    | infeasible | {"0": 61, "1": 2, "2": 6}                          | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   3 | ROS       | ok         | {"0": 59, "1": 2, "2": 6}                          | {"0": 59, "1": 59, "2": 59}                            | nan                                                                                          |
|   3 | SMOTE     | infeasible | {"0": 59, "1": 2, "2": 6}                          | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   3 | ADASYN    | infeasible | {"0": 59, "1": 2, "2": 6}                          | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   4 | ROS       | ok         | {"0": 60, "1": 2, "2": 1, "3": 6}                  | {"0": 60, "1": 60, "2": 60, "3": 60}                   | nan                                                                                          |
|   4 | SMOTE     | infeasible | {"0": 60, "1": 2, "2": 1, "3": 6}                  | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   4 | ADASYN    | infeasible | {"0": 60, "1": 2, "2": 1, "3": 6}                  | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   4 | ROS       | ok         | {"0": 60, "1": 1, "3": 6}                          | {"0": 60, "1": 60, "3": 60}                            | nan                                                                                          |
|   4 | SMOTE     | infeasible | {"0": 60, "1": 1, "3": 6}                          | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 1, n_samples = 1 |
|   4 | ADASYN    | infeasible | {"0": 60, "1": 1, "3": 6}                          | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 1, n_samples = 1 |
|   4 | ROS       | ok         | {"0": 60, "1": 1, "2": 1, "3": 6}                  | {"0": 60, "1": 60, "2": 60, "3": 60}                   | nan                                                                                          |
|   4 | SMOTE     | infeasible | {"0": 60, "1": 1, "2": 1, "3": 6}                  | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 1, n_samples = 1 |
|   4 | ADASYN    | infeasible | {"0": 60, "1": 1, "2": 1, "3": 6}                  | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 1, n_samples = 1 |
|   5 | ROS       | ok         | {"0": 54, "1": 4, "3": 9, "4": 1}                  | {"0": 54, "1": 54, "3": 54, "4": 54}                   | nan                                                                                          |
|   5 | SMOTE     | infeasible | {"0": 54, "1": 4, "3": 9, "4": 1}                  | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 1, n_samples = 1 |
|   5 | ADASYN    | infeasible | {"0": 54, "1": 4, "3": 9, "4": 1}                  | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 1, n_samples = 1 |
|   5 | ROS       | ok         | {"0": 53, "1": 4, "2": 1, "3": 9, "4": 1}          | {"0": 53, "1": 53, "2": 53, "3": 53, "4": 53}          | nan                                                                                          |
|   5 | SMOTE     | infeasible | {"0": 53, "1": 4, "2": 1, "3": 9, "4": 1}          | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 1, n_samples = 1 |
|   5 | ADASYN    | infeasible | {"0": 53, "1": 4, "2": 1, "3": 9, "4": 1}          | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 1, n_samples = 1 |
|   5 | ROS       | ok         | {"0": 53, "1": 4, "2": 1, "3": 8, "4": 2}          | {"0": 53, "1": 53, "2": 53, "3": 53, "4": 53}          | nan                                                                                          |
|   5 | SMOTE     | infeasible | {"0": 53, "1": 4, "2": 1, "3": 8, "4": 2}          | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 1, n_samples = 1 |
|   5 | ADASYN    | infeasible | {"0": 53, "1": 4, "2": 1, "3": 8, "4": 2}          | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 1, n_samples = 1 |
|   6 | ROS       | ok         | {"0": 46, "1": 2, "2": 1, "3": 2, "4": 4, "5": 14} | {"0": 46, "1": 46, "2": 46, "3": 46, "4": 46, "5": 46} | nan                                                                                          |
|   6 | SMOTE     | infeasible | {"0": 46, "1": 2, "2": 1, "3": 2, "4": 4, "5": 14} | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   6 | ADASYN    | infeasible | {"0": 46, "1": 2, "2": 1, "3": 2, "4": 4, "5": 14} | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   6 | ROS       | ok         | {"0": 46, "1": 2, "2": 1, "3": 1, "4": 4, "5": 14} | {"0": 46, "1": 46, "2": 46, "3": 46, "4": 46, "5": 46} | nan                                                                                          |
|   6 | SMOTE     | infeasible | {"0": 46, "1": 2, "2": 1, "3": 1, "4": 4, "5": 14} | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   6 | ADASYN    | infeasible | {"0": 46, "1": 2, "2": 1, "3": 1, "4": 4, "5": 14} | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   6 | ROS       | ok         | {"0": 46, "1": 2, "3": 1, "4": 4, "5": 14}         | {"0": 46, "1": 46, "3": 46, "4": 46, "5": 46}          | nan                                                                                          |
|   6 | SMOTE     | infeasible | {"0": 46, "1": 2, "3": 1, "4": 4, "5": 14}         | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |
|   6 | ADASYN    | infeasible | {"0": 46, "1": 2, "3": 1, "4": 4, "5": 14}         | {}                                                     | Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 2 |

## 5. Learning curves and generalisation
- Valid paper-holdout learning-curve points: **8**.
- Mean unseen-paper macro-F1 across curve points: **0.620**.

## 6. SHAP / feature importance
| feature                            |   mean_abs_shap |
|:-----------------------------------|----------------:|
| Electrolyte_Concentration_M        |      0.306373   |
| HMF_Concentration_mM               |      0.0879577  |
| Catalyst_Carbon                    |      0.0708827  |
| Catalyst_Fe                        |      0.0511058  |
| Potential_vs_RHE_V                 |      0.0402414  |
| Catalyst_LDH                       |      0.0194686  |
| Temperature_C                      |      0.0155843  |
| Electrolyte_Electrolyte_Cation_nan |      0.0111828  |
| Potential_vs_RHE                   |      0.00587176 |
| Electrolyte_Electrolyte_Cation_Li  |      0.00525364 |
| TEMPO_binary                       |      0          |
| Alkaline_Electrolyte               |      0          |
Feature importance is associational and should not be interpreted as causal evidence.

## 7. Multi-output regression
| target          |   model_r2_mean |   model_r2_sd |   model_mae_mean |   baseline_mae_mean |   model_rmse_mean |
|:----------------|----------------:|--------------:|-----------------:|--------------------:|------------------:|
| Conversion_pct  |      -56.8189   |     51.0613   |          3.91703 |             2.07617 |           11.4735 |
| FE_pct          |       -1.37152  |      1.66131  |          8.68344 |             6.886   |           14.8821 |
| Selectivity_pct |       -0.379789 |      0.569455 |          5.92782 |             5.3445  |           11.9639 |
| Yield_pct       |       -0.452306 |      0.694989 |          5.96523 |             5.29133 |           11.9438 |

Complete valid records used for the shared four-output comparison: **60** across **47** papers.
Rows excluded for out-of-range percentage targets: **0**.
Regression validation gate: **NOT PASSED**.
One MultiOutputRegressor predicts Conversion, FDCA Yield, Faradaic Efficiency and Selectivity simultaneously. Missing target measurements are not fabricated; the strict shared comparison uses records with all four targets observed.

## 8. Genetic algorithm / NSGA-II
- Status: **blocked**.
- Reason: Multi-output regression did not beat baseline on all four targets.
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
