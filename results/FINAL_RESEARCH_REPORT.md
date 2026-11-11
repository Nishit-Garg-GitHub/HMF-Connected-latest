# HMF Electrooxidation to FDCA — Integrated ML Research Report

## Research architecture
Literature dataset (154 records) → cleaning and feature engineering → K-means (K=3,4,5,6) → fold-local class balancing (ROS, SMOTE, ADASYN) → classifier comparison → learning curves → SHAP/feature importance → joint four-output regression → validation-gated NSGA-II.

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
| 3 | 154 | `{0: 138, 1: 5, 2: 11}` |
| 4 | 154 | `{0: 137, 1: 3, 2: 4, 3: 10}` |
| 5 | 154 | `{0: 17, 1: 7, 2: 2, 3: 4, 4: 124}` |
| 6 | 154 | `{0: 127, 1: 2, 2: 4, 3: 6, 4: 14, 5: 1}` |

K-means is performed on conversion/yield outcome space for descriptive performance grouping. Cluster labels are never included as input features for the regression model.

## 3. Classification and sampling comparison
| target_kind    |   k | model            | sampler   |   macro_f1_mean |   macro_f1_sd |   macro_f1_ci95_lo |   macro_f1_ci95_hi |   accuracy_mean | leakage_flag   |
|:---------------|----:|:-----------------|:----------|----------------:|--------------:|-------------------:|-------------------:|----------------:|:---------------|
| kmeans_cluster |   3 | GradientBoosting | none      |        0.625501 |     0.0595084 |           0.600937 |           0.650065 |        0.905714 | False          |
| kmeans_cluster |   3 | GradientBoosting | SMOTE     |        0.604259 |     0.080735  |           0.570934 |           0.637585 |        0.905974 | False          |
| kmeans_cluster |   3 | GradientBoosting | ADASYN    |        0.602803 |     0.0973239 |           0.562629 |           0.642976 |        0.905714 | False          |
| kmeans_cluster |   3 | GradientBoosting | ROS       |        0.590515 |     0.0434347 |           0.572586 |           0.608443 |        0.875584 | False          |
| kmeans_cluster |   3 | NeuralNetwork    | ROS       |        0.535104 |     0.0950644 |           0.495863 |           0.574345 |        0.875325 | False          |
| kmeans_cluster |   3 | NeuralNetwork    | ADASYN    |        0.519645 |     0.0839152 |           0.485007 |           0.554284 |        0.867013 | False          |
| kmeans_cluster |   3 | DecisionTree     | none      |        0.518887 |     0.0762407 |           0.487416 |           0.550357 |        0.839221 | False          |
| kmeans_cluster |   3 | NeuralNetwork    | SMOTE     |        0.515372 |     0.0839641 |           0.480713 |           0.550031 |        0.866494 | False          |
| kmeans_cluster |   3 | DecisionTree     | ROS       |        0.511631 |     0.0789258 |           0.479052 |           0.54421  |        0.847013 | False          |
| kmeans_cluster |   3 | SVM              | none      |        0.485886 |     0.0927914 |           0.447584 |           0.524188 |        0.872208 | False          |
| kmeans_cluster |   3 | ExtraTrees       | SMOTE     |        0.484388 |     0.0255426 |           0.473844 |           0.494931 |        0.911169 | False          |
| kmeans_cluster |   3 | RandomForest     | ROS       |        0.480831 |     0.0905955 |           0.443436 |           0.518227 |        0.898701 | False          |
| kmeans_cluster |   3 | ExtraTrees       | ADASYN    |        0.478056 |     0.0272057 |           0.466826 |           0.489286 |        0.91013  | False          |
| kmeans_cluster |   3 | ExtraTrees       | none      |        0.456594 |     0.046887  |           0.43724  |           0.475948 |        0.907273 | False          |
| kmeans_cluster |   3 | DecisionTree     | ADASYN    |        0.453957 |     0.0654942 |           0.426922 |           0.480992 |        0.88026  | False          |
| kmeans_cluster |   3 | DecisionTree     | SMOTE     |        0.450015 |     0.05622   |           0.426808 |           0.473221 |        0.878701 | False          |
| kmeans_cluster |   3 | ExtraTrees       | ROS       |        0.444364 |     0.0407617 |           0.427538 |           0.46119  |        0.903636 | False          |
| kmeans_cluster |   3 | NeuralNetwork    | none      |        0.438214 |     0.0728097 |           0.40816  |           0.468269 |        0.887013 | False          |
| kmeans_cluster |   3 | SVM              | SMOTE     |        0.435596 |     0.0987806 |           0.394821 |           0.476371 |        0.888312 | False          |
| kmeans_cluster |   4 | DecisionTree     | SMOTE     |        0.427478 |     0.0835209 |           0.393003 |           0.461954 |        0.88026  | False          |

Selected screening candidate: **NeuralNetwork + SMOTE**, K=None, mean macro-F1=0.7306760875189986, 95% CI=[0.717, 0.744], dummy baseline mean macro-F1=0.4800, permutation test p-value=0.4653.
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
|   3 | ADASYN    | ok         | {"0": 93, "1": 3, "2": 8}                         | {"0": 93, "1": 92, "2": 91}                            | nan                                                                                          |
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
- Unseen-paper macro-F1 at the full training fraction (stage 05, publication-grouped holdouts):

| K | Model | Validation macro-F1 | Training macro-F1 |
|---|---|---|---|
| 3 | DecisionTree | 0.481 | 0.920 |
| 3 | NeuralNetwork | 0.537 | 0.959 |
| 3 | RandomForest | 0.470 | 0.939 |
| 3 | SVM | 0.729 | 0.834 |
| 4 | DecisionTree | 0.286 | 0.556 |
| 4 | NeuralNetwork | 0.299 | 0.969 |
| 4 | RandomForest | 0.228 | 0.959 |
| 4 | SVM | 0.223 | 0.909 |
| 5 | DecisionTree | 0.209 | 0.740 |
| 5 | NeuralNetwork | 0.315 | 0.774 |
| 5 | RandomForest | 0.231 | 0.874 |
| 5 | SVM | 0.250 | 0.734 |

## 6. SHAP / feature importance
| feature                           |   mean_abs_shap |
|:----------------------------------|----------------:|
| Catalyst_Conductive_Glass         |       0.0897748 |
| Catalyst_Cu                       |       0.0636189 |
| Catalyst_V                        |       0.0576132 |
| Counter_Electrode_Carbon          |       0.0483421 |
| Catalyst_Carbon                   |       0.0388169 |
| missingindicator_Temperature_C    |       0.0359286 |
| Catalyst_Hydroxide                |       0.0348796 |
| Electrolyte_Electrolyte_Anion_nan |       0.0274185 |
| Catalyst_N_Elements               |       0.0254851 |
| Catalyst_Oxyhydroxide             |       0.0244946 |
| Catalyst_Nitride                  |       0.023863  |
| Three_Electrode                   |       0.0232732 |
Feature importance is associational and should not be interpreted as causal evidence.

## 7. Multi-output regression
### Primary — Per-target regression (SelectKBest + ExtraTrees)
| target          |   model_r2_mean |   model_r2_sd |   model_mae_mean |   baseline_mae_mean |   model_rmse_mean |
|:----------------|----------------:|--------------:|-----------------:|--------------------:|------------------:|
| Conversion_pct  |      -1.73258   |      4.86348  |          6.40674 |             7.06885 |           13.513  |
| FE_pct          |      -0.0655545 |      0.484988 |          8.60037 |             9.94933 |           14.9696 |
| Selectivity_pct |       0.0299772 |      0.405537 |          9.97005 |            12.3653  |           17.2024 |
| Yield_pct       |       0.0440759 |      0.170476 |         11.8639  |            13.1909  |           19.4966 |

### Secondary — Joint multi-output regression (complete rows only)
| target          |   model_r2_mean |   model_r2_sd |   model_mae_mean |   baseline_mae_mean |   model_rmse_mean |
|:----------------|----------------:|--------------:|-----------------:|--------------------:|------------------:|
| Conversion_pct  |      -3.57886   |     3.19244   |          2.55179 |             2.07617 |           7.67329 |
| FE_pct          |      -0.375536  |     0.655528  |          7.28042 |             6.886   |          12.4328  |
| Selectivity_pct |      -0.0213833 |     0.131564  |          5.82484 |             5.3445  |          10.9836  |
| Yield_pct       |      -0.0489266 |     0.0892687 |          5.74454 |             5.29133 |          11.1053  |


Regression validation gate: **PASSED**.
Gate criteria: Repeated publication-grouped GroupKFold (10 repeats x 3 folds); MAE must beat median baseline and win a majority of folds for every target.
Per-target models train on all records with each target observed, preventing sample starvation from requiring all 4 targets simultaneously.

## 8. Genetic algorithm / NSGA-II
- Status: **completed_model_generated_candidates**.
- Model-generated Pareto candidates: **80**.
- NSGA-II is gated by the multi-output regression validation results.
- Any Pareto candidates are model-generated hypotheses and require experimental validation.

### Constrained scenario: Ni(OH)2 / Cu(OH)2 anode (top 10)
- Conditions bounded to the range reported for real Ni/Cu hydroxide anodes: {'Potential_vs_RHE_V': [1.4, 1.69], 'Temperature_C': [25.0, 40.0]}.
|   rank | catalyst   | support   | electrolyte   |   electrolyte_M |   HMF_mM |   potential_V_RHE |   temperature_C |   predicted_Conversion_pct |   predicted_Yield_pct |   predicted_FE_pct |   predicted_Selectivity_pct |
|-------:|:-----------|:----------|:--------------|----------------:|---------:|------------------:|----------------:|---------------------------:|----------------------:|-------------------:|----------------------------:|
|      1 | Ni(OH)2    | foam      | KOH           |            1.49 |     10   |             1.443 |            25   |                      99.09 |                 95.16 |              96.32 |                       97.41 |
|      2 | Ni(OH)2    | foam      | KOH           |            1.34 |     11.5 |             1.438 |            29.4 |                      99.01 |                 95.44 |              96.23 |                       96.87 |
|      3 | Ni(OH)2    | foam      | KOH           |            0.95 |      9.5 |             1.422 |            25.1 |                      99.19 |                 95.44 |              95.09 |                       97.43 |
|      4 | Ni(OH)2    | foam      | KOH           |            1    |     10.6 |             1.4   |            25   |                      98.34 |                 95.44 |              95.87 |                       97.18 |
|      5 | Ni(OH)2    | foam      | KOH           |            0.6  |     11.5 |             1.436 |            25.1 |                      99.21 |                 94.39 |              94.7  |                       97.43 |
|      6 | Ni(OH)2    | foam      | KOH           |            0.61 |     10   |             1.423 |            35.7 |                      99.21 |                 94.94 |              94.7  |                       96.87 |
|      7 | Ni(OH)2    | foam      | KOH           |            1.05 |     13.3 |             1.429 |            25   |                      99.19 |                 92.7  |              96.21 |                       97.43 |
|      8 | Ni(OH)2    | foam      | KOH           |            1.04 |      6   |             1.426 |            25.9 |                      99.19 |                 92.39 |              96.21 |                       96.87 |
|      9 | Ni(OH)2    | foam      | KOH           |            0.74 |     10.5 |             1.408 |            26.6 |                      98.46 |                 94.94 |              94.58 |                       96.62 |
|     10 | Ni(OH)2    | foam      | KOH           |            1    |     10   |             1.437 |            25   |                      95.73 |                 95.16 |              96.23 |                       97.43 |

Nearest observed hydroxide record: Ni(OH)2/NF (1.0 M KOH | 1.45 V vs. RHE), measured yield 88.3%, FE 90.6%.

## 9. Reproducibility and limitations
- All four numerical outcomes are excluded from the input matrix.
- Publication-grouped validation is used to reduce leakage between records from the same paper.
- Imputation, variance filtering, scaling and resampling are fitted within training folds.
- Electrode potential vs RHE is parsed only from explicitly RHE-referenced records; cell voltage is kept separate and excluded from the agreed input feature set.
- Invalid percentage targets outside 0–100 are excluded from the strict four-output regression rather than clipped or fabricated.
- The dataset is small and literature-derived; missingness, heterogeneous reporting and publication bias limit generalisation.
- A favourable F1 or R² does not establish causal catalyst effects.
- NSGA-II outputs must be confirmed experimentally before being described as optimal catalysts or operating conditions.
