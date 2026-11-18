# HMF Electrooxidation → FDCA: Connected Research Pipeline

This repository runs the research workflow in dependency order. Each stage
consumes artifacts created by the preceding stage. The original standalone
research scripts are retained under `archive_legacy/scripts/` for provenance.

## Folder structure

```text
scripts/
  descriptors.py            # pure text parsers used by stage 01 (unit-tested)
  common.py                 # shared features, labels, models, validation helpers
  01_preprocess_data.py     # cleaning + descriptors (+ derived yield, flagged)
  02_kmeans.py              # K-means on (Conversion, Yield), K = 3-6
  03_oversampling.py        # ROS / SMOTE / ADASYN fold audit
  04_model_evaluation.py    # 7 classifier families x 4 samplers, grouped CV
  05_learning_curves.py
  06_shap_analysis.py       # classifier SHAP
  07_regression.py          # per-target regressors + validation gate + nested check
  08_regression_shap.py     # regression Tree SHAP, bootstraps, literature check
  09_nsga2.py               # validation-gated NSGA-II (Ni / Cu / Ni-Cu hydroxide)
  10_diagnostic_figures.py  # workflow schematic + diagnostic figures
  11_final_report.py        # Markdown report
  12_latex_report.py        # research-paper style LaTeX + PDF
tests/                      # python -m unittest discover -s tests -v
data/raw/                   # the single copy of the literature workbook
data/processed/  results/  figures/  report/  archive_legacy/
run_all.py  requirements.txt
```

## Run

```bash
pip install -r requirements.txt
python run_all.py                       # all 12 stages sequentially
python run_all.py --start 07 --no-clean # resume from a stage
```

The runner stops on the first failing stage and writes one log per stage under
`results/`. Stage 12 compiles the PDF with `tectonic` or `pdflatex` if present.

## Data usage (why not all 154 records everywhere)

| Analysis | Records | Reason |
|---|---|---|
| Descriptors / SHAP / regression | all 154 (each target uses every record that reports it: 132 yield, 119 FE, 117 conversion, 91 selectivity) | a target is never imputed |
| K-means and K-class classification | 111 | both Conversion and Yield must be measured; no label is built from an imputed value |
| Binary ≥ 90 % screen | 128 | FDCA is the major product and yield is reported or derived |

Yield is derived as Conversion × Selectivity / 100 for 9 records whose major
product is FDCA and whose yield is not reported (flagged `Yield_Derived`).
Values outside 0–100 % are treated as missing, never clipped.
Set `HMF_CLUSTER_RECORDS=all` for the median-imputed 154-record K-means
sensitivity variant.

## Validation design

- Publication-grouped cross-validation (StratifiedGroupKFold / GroupKFold by
  paper, 99 papers); imputation, scaling and oversampling are fitted inside
  training folds only.
- Classification: pooled out-of-fold macro-F1, 25 repeats × 5 folds, compared
  with a stratified dummy on identical splits; the selected candidate is the one
  with the largest lift over its own dummy. Two permutation tests are reported
  (within-paper and global).
- Regression: pooled out-of-fold R² and MAE against a training-fold **median**
  baseline, 10 repeats × 5 folds, plus a nested (selection-corrected) check.
  The gate applies to the targets NSGA-II optimises (Yield, FE).
- NSGA-II runs only when the gate passes; conversion is not an objective
  because its regressor does not beat the baseline.

## Current results (reproduce with `python run_all.py`)

Classification, pooled out-of-fold macro-F1, best sampler per model:

| Task | DT | SVM | RF | NN | GB | ET | ENS | dummy |
|---|---|---|---|---|---|---|---|---|
| Binary ≥ 90 % | 0.599 | 0.733 | 0.700 | 0.715 | 0.680 | 0.726 | **0.741** | 0.479 |
| K = 3 (111 records) | 0.616 | 0.569 | 0.480 | 0.522 | **0.658** | 0.453 | 0.551 | 0.328 |

For K ≥ 4 ranked scores are 0.20–0.36 (singleton clusters make SMOTE/ADASYN
infeasible; those combinations are reported as infeasible, not scored).
The selected candidate (K = 3 GB + ROS) has a global permutation p = 0.0099 and
a within-paper p = 0.14: the signal is real but mostly between papers. These
scores are optimistic because the best of ≈ 100 model/sampler/task
combinations is reported on the records used to choose hyper-parameters.

Regression (paper-grouped, pooled out-of-fold):

| Target | R² | MAE | median-baseline MAE | nested R² | gate |
|---|---|---|---|---|---|
| FDCA yield | 0.44 | 8.7 | 11.1 | 0.45 | required, passed |
| Faradaic efficiency | 0.39 | 7.4 | 8.6 | 0.39 | required, passed |
| Selectivity | 0.30 | 8.6 | 9.5 | 0.30 | informational |
| Conversion | 0.19 | 5.1 | 4.6 | 0.01 | informational, fails: not predictable |

Regression SHAP (Tree SHAP, 30 paper bootstraps) versus the experimental
literature (`results/literature_check.csv`): a Ni-containing anode is the second
most important yield descriptor with a stable positive sign, consistent with the
Ni(OH)₂/NiOOH literature, and HMF concentration lowers yield. Cu, the
hydroxide phase and alkaline electrolyte are **not distinguishable** in this
corpus (few Cu records; nearly all records alkaline), so the data cannot rank Ni
against Cu. The largest single contribution is the conductive-glass (FTO/ITO)
support, almost certainly a study-design confound rather than a catalyst effect.
All records use HMF, so the reactant cannot be assessed.
The literature references are recorded in the paper with the authors/titles
flagged for completion from the full text; verify them before submission.

NSGA-II (population 100, 1000 generations; objectives: max yield and FE, min
potential and temperature) was run for Ni(OH)₂, Cu(OH)₂ and Ni–Cu hydroxide
anodes within the potential/temperature range reported for such anodes.
Candidates are written to `results/nsga2_top10_ni_cu_hydroxide.csv` (and
`nsga2_top10_robust_potential.csv` for E ≥ 1.45 V). Ni-containing fronts are
predicted above the Cu(OH)₂ front, but this reflects the training data and must
not be read as Cu being inferior. Candidates collect at the lower bounds of E
and T, lie far from any measured record, and are predicted ≈ 9 points above
their nearest measured record; they are **hypotheses for laboratory testing**,
not validated optima.

The two descriptive oversampling figures (stage 03 performance-space plot and stage 05 oversampled-NN
learning curves) duplicate single-record classes before SMOTE (`common.balance_with_singleton_fallback`).
ADASYN stays blank wherever imbalanced-learn cannot define it (classes of 1–4 records with no majority-class
neighbours). Stage 04 classification scores do not use this fallback.

## Interpretation notes

- K-means classes are imbalanced; macro-F1 for rare clusters is unstable.
- The binary threshold (performance ≥ 90 %) is a screening definition.
- SHAP values are associational, not causal catalyst effects.
- A passing internal gate is not experimental validation.

## Main outputs

- `results/classification_summary.csv`, `best_classification_candidate.json`
- `results/per_target_regression_summary.csv`, `per_target_regression_nested.csv`,
  `regression_validation_gate.json`
- `results/shap_regression_*.csv`, `literature_check.csv`
- `results/nsga2_*.csv`, `nsga2_status.json`
- `results/FINAL_RESEARCH_REPORT.md`, `report/HMF_FDCA_ML_Report.pdf`
- Figures under `figures/` (`diagnostics/`, `shap/`, `nsga2/`, `classification/`,
  `learning_curves/`, `paper_style/`, `KMeans*/`).

`figures/KMeans_Figure6/HMF_KMeans_Figure6_Style.png` reproduces the layout of
Figure 6 of Li et al., ACS Sustainable Chem. Eng. 2025, 13, 13940, which this
workflow adapts to electro-oxidation.
