# HMF–FDCA ML pipeline — corrected scripts

This folder contains the ten corrected Python scripts. Keep the filenames unchanged and
place all ten files together in the project's `scripts/` directory, replacing the
corresponding originals.

## Important changes
- Electrode potential is parsed numerically only when the source explicitly says `vs. RHE`.
  Cell voltage is extracted separately and is not used as an electrode-potential feature.
- Electrolyte cation extraction now recognises formula strings such as KOH and Na2SO4.
- Shared feature selection explicitly excludes measured outcomes and keeps the RHE numeric
  descriptor rather than the ambiguous legacy `Potential_V` field.
- K-means validates the required outcomes and creates its results directory before saving
  row-aligned labels.
- Oversampling feasibility is audited for every publication-grouped training fold.
- Classification, learning curves and SHAP use the same fold-local zero-variance filtering,
  imputation and scaling contract.
- Fold class support is saved, and infeasible grouped splits/samplers are reported rather
  than silently substituted.
- Regression no longer clips invalid observed percentages into the valid range. Complete
  four-target rows must be finite and between 0 and 100; exclusions are counted.
- The regression gate requires MAE improvement over the median baseline on every target
  and a majority of fold wins. NSGA-II remains blocked unless this evidence passes.
- NSGA-II candidates are imputed before search and electrolyte/reactant one-hot groups are
  repaired to avoid impossible category combinations. A feature-contract check prevents
  using a model trained on a different preprocessing matrix.

## Validation performed
- All ten scripts passed Python syntax compilation.
- Stage 01 was run against the uploaded 154-row workbook.
- Stages 02 through 09 were run in an isolated project-shaped test directory.
- All stages completed. The regression gate correctly blocked NSGA-II because the
  multi-output model did not outperform the median baseline on every target.
- This validates execution on the available environment and workbook; it does not establish
  scientific predictive performance or experimental validation.

## Run order
Run from the `scripts` directory, with the source workbook in `data/raw/` (or the project
root / scripts folder under its original filename):

```powershell
python 01_preprocess_data.py
python 02_kmeans.py
python 03_oversampling.py
python 04_model_evaluation.py
python 05_learning_curves.py
python 06_shap_analysis.py
python 07_regression.py
python 08_nsga2.py
python 09_final_report.py
```

`common.py` must remain in the same `scripts` directory as the stage scripts.
