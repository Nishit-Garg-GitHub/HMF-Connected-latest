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
4. **Model comparison:** compares Decision Tree, SVM, Random Forest and Neural
   Network under no sampling, ROS, SMOTE and ADASYN. It reports K-means class
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
- The current regression model did not beat the median baseline on every
  target. Therefore NSGA-II is correctly blocked. This is a scientific safeguard,
  not a pipeline failure.
- A favourable F1/R² does not prove causation. SHAP/permutation values are
  associational. Any model-generated catalyst proposal requires experiments.

## Main outputs

- `results/classification_summary.csv`
- `results/oversampling_audit.csv`
- `results/learning_curve_selected_model.csv`
- `results/feature_importance_selected_model.csv`
- `results/multioutput_regression_summary.csv`
- `results/regression_validation_gate.json`
- `results/nsga2_status.json`
- `results/FINAL_RESEARCH_REPORT.md`
- Figures are under `figures/`.

The project keeps the full earlier scripts in `archive_legacy/`; the active
scripts are the connected, runtime-optimised implementation.
