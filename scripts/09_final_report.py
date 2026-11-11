"""Stage 09 — assemble the connected pipeline's audit and research report.

This report reads outputs produced by stages 01–08. It does not rerun model
training. It records data coverage, K-means outputs, classification F1,
resampling feasibility, learning curves, feature interpretation, four-target
regression metrics and the NSGA-II validation decision. Screening scores are
reported as screening estimates, not as guaranteed or experimentally
validated performance.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common import (
    RESULTS_DIR, DATA_DIR, FIGURES_DIR, TARGETS, TARGET_LABELS,
    read_processed, read_cleaned, stage_status,
)

REPORT_PATH = RESULTS_DIR / "FINAL_RESEARCH_REPORT.md"


def read_json(path: Path, default=None):
    """Read an optional JSON artifact without hiding malformed content."""
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def table_markdown(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    """Format a compact Markdown table, with a readable empty-state message."""
    if frame.empty:
        return "_No valid rows were produced by this stage._"
    if columns:
        columns = [column for column in columns if column in frame.columns]
        frame = frame[columns]
    return frame.to_markdown(index=False)


def dataset_section() -> list[str]:
    """Summarise the original and cleaned record counts."""
    processed = read_processed()
    cleaned = read_cleaned()
    lines = [
        "## 1. Literature dataset and preprocessing",
        f"- Source records: **{len(processed)}**.",
        f"- Publication groups: **{cleaned['Paper Title'].nunique()}**."
        if "Paper Title" in cleaned else "- Publication groups: not available.",
        f"- Preprocessed descriptors: **{len(processed.columns) - len(TARGETS)} columns before exclusions**.",
        "- The feature matrix excludes all four measured targets and outcome-derived columns.",
        "- The agreed experimental input groups are TEMPO, temperature, reactant monomer, HMF concentration, catalyst, electrolyte, potential vs RHE and cell type.",
        "- Chemical/catalyst/electrolyte descriptors are represented by engineered numerical columns.",
        "",
    ]
    return lines


def clustering_section() -> list[str]:
    """Report K-means class support for all requested K values."""
    lines = ["## 2. K-means clustering", "| K | Records clustered | Cluster counts |", "|---:|---:|---|"]
    for k in (3, 4, 5, 6):
        path = RESULTS_DIR / f"kmeans_labels_K{k}.csv"
        if path.exists():
            frame = pd.read_csv(path)
            counts = frame.cluster.value_counts().sort_index().to_dict()
            lines.append(f"| {k} | {len(frame)} | `{counts}` |")
        else:
            lines.append(f"| {k} | — | Not produced |")
    lines.extend(["", "K-means is performed on conversion/yield outcome space for descriptive performance grouping. Cluster labels are never included as input features for the regression model.", ""])
    return lines


def classification_section() -> list[str]:
    """Summarise four model families and four sampling alternatives."""
    path = RESULTS_DIR / "classification_summary.csv"
    best = read_json(RESULTS_DIR / "best_classification_candidate.json")
    lines = ["## 3. Classification and sampling comparison"]
    if path.exists():
        summary = pd.read_csv(path)
        lines.append(table_markdown(summary.head(20), [
            "target_kind", "k", "model", "sampler", "n_valid", "macro_f1_mean",
            "macro_f1_sd", "weighted_f1_mean", "accuracy_mean",
            "balanced_accuracy_mean",
        ]))
    else:
        lines.append("_Classification summary was not generated._")
    lines.extend([
        "",
        f"Selected screening candidate: **{best.get('model', 'N/A')} + {best.get('sampler', 'N/A')}**, K={best.get('k', 'N/A')}, mean macro-F1={best.get('macro_f1_mean', 'N/A')}.",
        "Samplers: no sampling, RandomOverSampler (ROS), SMOTE and ADASYN. Sampling occurs inside training folds only. Infeasible sampler/fold combinations are reported rather than silently replaced.",
        "These are screening results. A high F1 is not guaranteed; model choice should be based on repeated unseen-paper performance and class support, not a single best fold.",
        "",
    ])
    return lines


def balancing_section() -> list[str]:
    """Summarise sampler feasibility from stage 03."""
    path = RESULTS_DIR / "oversampling_audit.csv"
    lines = ["## 4. Fold-local balancing audit"]
    if path.exists():
        frame = pd.read_csv(path)
        lines.append(table_markdown(frame, ["k", "sampler", "status", "before", "after", "reason"]))
    else:
        lines.append("_Oversampling audit was not generated._")
    lines.append("")
    return lines


def learning_shap_section() -> list[str]:
    """Report generalisation and interpretation artifacts."""
    lines = ["## 5. Learning curves and generalisation"]
    curve_path = RESULTS_DIR / "learning_curve_selected_model.csv"
    if curve_path.exists():
        curve = pd.read_csv(curve_path)
        if "test_macro_f1" in curve:
            valid = curve.dropna(subset=["test_macro_f1"])
            mean_score = valid.test_macro_f1.mean() if len(valid) else float("nan")
            lines.append(f"- Valid paper-holdout learning-curve points: **{len(valid)}**.")
            lines.append(f"- Mean unseen-paper macro-F1 across curve points: **{mean_score:.3f}**.")
    else:
        lines.append("- Learning curve not generated.")
    lines.extend(["", "## 6. SHAP / feature importance"])
    importance = RESULTS_DIR / "feature_importance_selected_model.csv"
    if importance.exists():
        table = pd.read_csv(importance).head(12)
        lines.append(table_markdown(table))
        lines.append("Feature importance is associational and should not be interpreted as causal evidence.")
    else:
        lines.append("_Feature-importance analysis not generated._")
    lines.append("")
    return lines


def regression_section() -> list[str]:
    """Report the single joint four-output regression model and its gate."""
    lines = ["## 7. Multi-output regression"]
    path = RESULTS_DIR / "multioutput_regression_summary.csv"
    gate = read_json(RESULTS_DIR / "regression_validation_gate.json")
    if path.exists():
        summary = pd.read_csv(path)
        lines.append(table_markdown(summary, [
            "target", "model_r2_mean", "model_r2_sd", "model_mae_mean",
            "baseline_mae_mean", "model_rmse_mean",
        ]))
    else:
        lines.append("_Four-output regression did not complete._")
    lines.extend([
        "",
        f"Complete valid records used for the shared four-output comparison: **{gate.get('n_complete_records', 'N/A')}** across **{gate.get('n_papers', 'N/A')}** papers.",
        f"Rows excluded for out-of-range percentage targets: **{gate.get('n_rows_excluded_invalid_percentage', 'N/A')}**.",
        f"Regression validation gate: **{'PASSED' if gate.get('passed') else 'NOT PASSED'}**.",
        "One MultiOutputRegressor predicts Conversion, FDCA Yield, Faradaic Efficiency and Selectivity simultaneously. Missing target measurements are not fabricated; the strict shared comparison uses records with all four targets observed.",
        "",
    ])
    return lines


def optimisation_section() -> list[str]:
    """Document whether NSGA-II ran or was correctly blocked."""
    status = read_json(RESULTS_DIR / "nsga2_status.json")
    lines = ["## 8. Genetic algorithm / NSGA-II"]
    lines.append(f"- Status: **{status.get('status', 'not run')}**.")
    if status.get("reason"):
        lines.append(f"- Reason: {status['reason']}")
    if status.get("n_pareto_candidates") is not None:
        lines.append(f"- Model-generated Pareto candidates: **{status['n_pareto_candidates']}**.")
    lines.extend([
        "- NSGA-II is gated by the multi-output regression validation results.",
        "- Any Pareto candidates are model-generated hypotheses and require experimental validation.",
        "",
    ])
    return lines


def main() -> None:
    """Write the final report from the outputs of all previous stages."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HMF Electrooxidation to FDCA — Integrated ML Research Report", "",
        "## Research architecture", "Literature dataset (154 records) → cleaning and feature engineering → K-means (K=3,4,5,6) → fold-local class balancing (ROS, SMOTE, ADASYN) → classifier comparison → learning curves → SHAP/feature importance → joint four-output regression → validation-gated NSGA-II.",
        "",
        "This report distinguishes paper-inspired methodology from the results obtained on the current literature dataset. It does not claim that screening scores are experimental validation.",
        "",
    ]
    lines.extend(dataset_section())
    lines.extend(clustering_section())
    lines.extend(classification_section())
    lines.extend(balancing_section())
    lines.extend(learning_shap_section())
    lines.extend(regression_section())
    lines.extend(optimisation_section())
    lines.extend([
        "## 9. Reproducibility and limitations",
        "- All four numerical outcomes are excluded from the input matrix.",
        "- Publication-grouped validation is used to reduce leakage between records from the same paper.",
        "- Imputation, variance filtering, scaling and resampling are fitted within training folds.",
        "- Electrode potential vs RHE is parsed only from explicitly RHE-referenced records; cell voltage is kept separate and excluded from the agreed input feature set.",
        "- Invalid percentage targets outside 0–100 are excluded from the strict four-output regression rather than clipped or fabricated.",
        "- The dataset is small and literature-derived; missingness, heterogeneous reporting and publication bias limit generalisation.",
        "- A favourable F1 or R² does not establish causal catalyst effects.",
        "- NSGA-II outputs must be confirmed experimentally before being described as optimal catalysts or operating conditions.",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    stage_status("09_final_report", "completed", {"report": str(REPORT_PATH.relative_to(REPORT_PATH.parents[1]))})
    print(f"Final research report written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
