# HMF Electrooxidation → FDCA: Connected Fast Research Pipeline

This repository runs the research workflow in dependency order. Each stage
consumes the data/artifacts created by the preceding stage. The original
standalone research scripts are retained byte-for-byte under
`archive_legacy/scripts/` for provenance and comparison.

## Folder structure

```text
HMF_Connected_Fast_Pipeline/
├── scripts/
│   ├── 01_preprocess_data.py
│   ├── 02_kmeans.py
│   ├── 03_oversampling.py
│   ├── 04_model_evaluation.py
│   ├── 05_learning_curves.py
│   ├── 06_shap_analysis.py
│   ├── 07_regression.py
│   ├── 08_nsga2.py
│   ├── 09_final_report.py
│   └── common.py
├── data/
│   ├── raw/
│   └── processed/
├── results/
├── figures/
├── archive_legacy/scripts/
├── run_all.py
└── requirements.txt
```

## Run on Windows / VS Code

Open the project root (the folder containing `run_all.py`) in VS Code.

```powershell
# Activate your existing environment
.\.venv\Scripts\Activate.ps1

# Install any missing packages
pip install -r requirements.txt

# Run all nine stages sequentially
python run_all.py
```

The runner executes one stage at a time and stops if a stage fails. It writes
one log per stage under `results/`. To resume from a later stage after fixing
an issue, use (for example):

```powershell
python run_all.py --start 04 --no-clean
```

## Research workflow

1. **Preprocessing:** cleans the 154 literature records, engineers chemical
   and process descriptors, and writes the aligned cleaned/preprocessed files.
2. **K-means:** performs clustering directly on Conversion and FDCA Yield for
   K = 3, 4, 5 and 6. The cluster labels are saved and passed to later stages.
3. **Oversampling audit:** checks RandomOverSampler (ROS), SMOTE and ADASYN on
   paper-grouped training folds. It never oversamples validation records.
4. **Model comparison:** compares Decision Tree, SVM, Random Forest, Neural
   Network, Gradient Boosting and Extra Trees under no sampling, ROS, SMOTE
   and ADASYN. It reports K-means class
   prediction and a secondary paper-inspired binary high-performance screen.
5. **Learning curves:** evaluates the selected candidate on increasing
   fractions of publication groups.
6. **SHAP:** explains the selected classifier on a bounded sample. Kernel SHAP
   or Tree SHAP is used where supported; permutation importance is the fallback.
7. **Regression:** one multi-output model predicts Conversion, FDCA Yield,
   Faradaic Efficiency and Selectivity simultaneously using rows with all four
   measured outputs. Missing targets are not fabricated.
8. **NSGA-II:** runs only if the joint regression model beats the median
   baseline on all four outputs under publication-grouped validation.
9. **Final report:** compiles the outputs without retraining models.

## Runtime design

- Classification screening uses 2 repeats × 3 publication-grouped folds.
- The four classifier families use compact screening configurations.
- Sampling is inside the training-fold pipeline only.
- Learning curves and SHAP reuse the selected classifier instead of launching
  another full model grid.
- Regression uses 3 publication-grouped folds and 160 ExtraTrees per output.
- NSGA-II uses a small population/generation budget and is skipped when the
  regression validation gate fails.

These settings are intended for fast screening. Increase repeats/folds and
model complexity only after reviewing class support and selecting a stable
candidate. Do not present screening scores as final external validation.

## Important interpretation notes

- The K-means classes are highly imbalanced in this dataset. Some clusters
  contain very few observations, so macro-F1 can be unstable. ROS/SMOTE/ADASYN
  are evaluated honestly; no synthetic rows enter validation folds.
- The secondary binary screen defines Performance as min(Conversion, Yield)
  when Conversion is reported, otherwise Yield; Class 1 is Performance ≥90%.
  This is a screening threshold, not a universal chemical definition of a
  successful catalyst.
- The primary predictive task is **one multi-output regression model** for all
  four continuous outcomes.
- The strict shared regression dataset contains only records where all four
  outcomes are measured. This run found 60 complete records from 47 papers.
- NSGA-II runs only when the regression validation gate passes
  (`results/regression_validation_gate.json`); a failed gate blocks it as a
  scientific safeguard, not a pipeline failure.
- A favourable F1/R² does not prove causation. SHAP/permutation values are
  associational. Any model-generated catalyst proposal requires experiments.

## Main outputs

- `results/classification_summary.csv`
- `results/oversampling_audit.csv`
- `results/classification_incomplete_combinations.csv` (sampler/K pairs not ranked)
- `results/learning_curves_summary.csv`
- `results/feature_importance_selected_model.csv`
- `results/multioutput_regression_summary.csv`
- `results/regression_validation_gate.json`
- `results/nsga2_status.json`
- `results/FINAL_RESEARCH_REPORT.md`
- Figures are under `figures/`. `figures/classification/model_sampler_*.png`
  (including `model_sampler_binary90.png`) plot the same pooled scores as
  `classification_summary.csv`; `figures/shap/Figure_12_SHAP_Bar.png` and
  `Figure_13_SHAP_Beeswarm.png` explain the best Stage 04 model for K=3 and K=5.

## Classification update (descriptor fixes + tuned models)

Pooled out-of-fold macro-F1, 25 repeats × 5 publication-grouped folds, best
sampler per model:

| Target | DT | SVM | RF | NN | GB | ET |
|---|---|---|---|---|---|---|
| Binary ≥90% (before) | 0.557 | 0.534 | 0.586 | 0.507 | 0.575 | — |
| Binary ≥90% (now) | 0.600 | 0.695 | 0.678 | **0.731** | 0.661 | 0.720 |
| K=3 (before) | 0.410 | 0.298 | 0.379 | 0.389 | 0.421 | — |
| K=3 (now) | 0.519 | 0.486 | 0.481 | 0.535 | **0.626** | 0.484 |

What changed:
- **Stage 01 descriptors.** Catalyst flags are parsed from the anode/working
  electrode only. Before, the Pt *cathode* gave 74 "Pt catalysts", and
  phosphide/nitride/sulfide flags were always 0. Element symbols are matched on
  formulas (Ni3S2, CoNiP, VN, Ni₂P). New descriptors: Rh/V/Bi, element count,
  noble-metal and bimetallic flags, boride, foam/conductive-glass substrate,
  counter-electrode type (cell group) and electrolyte pH. "23 ± 2 °C" now
  parses as 23 °C (it used to give 2 °C).
- **Preprocessing.** Fold-local median imputation now appends missing-value
  indicators (e.g. temperature not reported), shared by stages 04–06.
- **Models.** Per-family hyperparameters come from a small grid. The NN no
  longer early-stops on a 10% split of a tiny fold. Extra Trees was added.
- **Evaluation fix.** SMOTE/ADASYN neighbours adapt to the smallest training
  class. A repeat with any failed fold is now `partial` and is not ranked, and
  a pair needs ≥80% complete repeats. Before, K=4 SMOTE/ADASYN "scores" came
  from about 1 of 5 folds, so earlier K=4 values (e.g. 0.477) were not
  comparable. Honest K=4 scores are 0.29–0.43.

Caveats: hyperparameters were selected on the same 154 records, so the scores
are somewhat optimistic. No model averages ≥0.76 across the 25 repeats; the
best binary NN reaches 0.76–0.78 on individual repeats. The within-paper label
permutation test is not significant (p≈0.47), because most descriptors are
constant within a paper. The model clearly beats the stratified dummy (0.48),
but its signal is mostly between papers.

The project keeps the full earlier scripts in `archive_legacy/`; the active
scripts are the connected, runtime-optimised implementation.
