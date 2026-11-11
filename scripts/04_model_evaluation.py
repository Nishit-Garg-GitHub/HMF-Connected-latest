"""Stage 04 — paper-grouped classifier comparison for K-means classes.

The reference-paper sequence is retained: Decision Tree, SVM, Random Forest
and Neural Network are compared under no sampling, ROS, SMOTE and ADASYN.
K-means labels from stage 02 are the class targets. All imputation, scaling
and sampling happen inside each training fold. This prevents the common
mistake of balancing the full dataset before cross-validation.

FAST SCREENING DEFAULT: two repeats x three publication-grouped folds. This
is a screening estimate, not a final nested-CV publication claim. Increase
REPEATS/FOLDS only after selecting a stable candidate.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score)
from sklearn.feature_selection import VarianceThreshold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

from common import (
    RESULTS_DIR, FIGURES_DIR, build_feature_matrix, grouped_splits,
    load_cluster_labels, make_classifier, paper_groups, read_cleaned, read_processed,
    save_figure, stage_status, write_json,
)

K_VALUES = (3, 4, 5, 6)
MODEL_NAMES = ("DecisionTree", "SVM", "RandomForest", "NeuralNetwork")
SAMPLERS = ("none", "ROS", "SMOTE", "ADASYN")
REPEATS = 2
FOLDS = 3
SEED_LIST = (42, 2027)


def sampler_for(name: str, seed: int):
    """Create a sampler; k-neighbours are kept small for literature data."""
    if name == "none":
        return None
    if name == "ROS":
        return RandomOverSampler(random_state=seed)
    if name == "SMOTE":
        return SMOTE(k_neighbors=2, random_state=seed)
    if name == "ADASYN":
        return ADASYN(n_neighbors=2, random_state=seed)
    raise ValueError(name)



def make_pipeline(model_name: str, sampler_name: str, seed: int, n_features: int = 15):
    """Build a fold-local preprocessing/sampling/model pipeline."""

    steps = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
    ]

    sampler = sampler_for(sampler_name, seed)

    if sampler is not None:
        steps.append(("sampler", sampler))

    steps.append(("model", make_classifier(model_name, seed)))

    return ImbPipeline(steps)



def evaluate_fold(X: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray,
                  test_idx: np.ndarray, model_name: str, sampler_name: str,
                  seed: int, k: int, fold: int) -> dict:
    """Fit one candidate on training papers and score unseen papers."""
    record = {
        "k": k, "seed": seed, "fold": fold, "model": model_name,
        "sampler": sampler_name, "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)), "status": "ok",
    }
    try:
        pipe = make_pipeline(model_name, sampler_name, seed, X.shape[1])
        pipe.fit(X.iloc[train_idx], y[train_idx])
        prediction = pipe.predict(X.iloc[test_idx])
        record.update({
            "macro_f1": float(f1_score(y[test_idx], prediction,
                                        average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y[test_idx], prediction,
                                           average="weighted", zero_division=0)),
            "accuracy": float(accuracy_score(y[test_idx], prediction)),
            "precision_macro": float(precision_score(
                y[test_idx], prediction, average="macro", zero_division=0
            )),
            "recall_macro": float(recall_score(
                y[test_idx], prediction, average="macro", zero_division=0
            )),
            "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], prediction)),
            "classes_train": int(len(np.unique(y[train_idx]))),
            "classes_test": int(len(np.unique(y[test_idx]))),
            "train_class_counts": json.dumps({
                str(int(label)): int(count)
                for label, count in zip(*np.unique(y[train_idx], return_counts=True))
            }),
            "test_class_counts": json.dumps({
                str(int(label)): int(count)
                for label, count in zip(*np.unique(y[test_idx], return_counts=True))
            }),
        })
    except Exception as exc:
        record.update({"status": "infeasible", "reason": str(exc)[:500]})
        record.update({
            "macro_f1": np.nan, "weighted_f1": np.nan, "accuracy": np.nan,
            "precision_macro": np.nan, "recall_macro": np.nan,
            "balanced_accuracy": np.nan,
        })
    return record


def run_k(k: int, X_all: pd.DataFrame, groups_all: np.ndarray) -> list[dict]:
    """Evaluate every model/sampler pair for one K value."""
    try:
        label_frame = load_cluster_labels(k)
    except Exception as exc:
        return [{"k": k, "status": "infeasible", "reason": str(exc),
                 "model": model, "sampler": sampler, "macro_f1": np.nan}
                for model in MODEL_NAMES for sampler in SAMPLERS]
    row_ids = label_frame["source_row"].astype(int).to_numpy()
    y = label_frame["cluster"].astype(int).to_numpy()
    X = X_all.iloc[row_ids].reset_index(drop=True)
    groups = groups_all[row_ids]
    output: list[dict] = []
    for seed in SEED_LIST[:REPEATS]:
        try:
            folds = grouped_splits(y, groups, seed, n_splits=FOLDS)
        except ValueError as exc:
            output.extend({
                "k": k, "seed": seed, "model": model_name, "sampler": sampler_name,
                "status": "infeasible", "reason": str(exc), "macro_f1": np.nan,
            } for model_name in MODEL_NAMES for sampler_name in SAMPLERS)
            continue
        for fold_no, (train_idx, test_idx) in enumerate(folds, start=1):
            print(
                f"K={k} | repeat={seed} | fold={fold_no}/{len(folds)} "
                f"| {len(MODEL_NAMES)} models x {len(SAMPLERS)} samplers",
                flush=True,
            )
            for model_name in MODEL_NAMES:
                for sampler_name in SAMPLERS:
                    result = evaluate_fold(
                        X, y, train_idx, test_idx, model_name,
                        sampler_name, seed, k, fold_no,
                    )
                    output.append(result)
    return output



def run_binary_high_performance(X_all: pd.DataFrame, groups_all: np.ndarray) -> list[dict]:
    """Evaluate the paper-inspired >=90% FDCA performance screening target.

    Performance is min(Conversion, Yield) when conversion is reported; when
    conversion is missing, measured FDCA yield is used. This binary task is a
    secondary classification screen only. The primary scientific model remains
    the four-output numerical regression in stage 07.
    """
    raw = read_cleaned()
    if "Major_Product_Clean" in raw.columns:
        fdca = raw["Major_Product_Clean"].astype(str).str.contains(
            r"FDCA|FURANDICARBOXYLIC", case=False, regex=True, na=False
        )
    else:
        fdca = pd.Series(True, index=raw.index)
    yield_values = pd.to_numeric(raw["Yield_pct"], errors="coerce")
    conversion = pd.to_numeric(raw["Conversion_pct"], errors="coerce")
    keep = fdca & yield_values.notna()
    row_ids = np.flatnonzero(keep.to_numpy())
    yld = yield_values.iloc[row_ids].clip(0, 100).to_numpy()
    conv = conversion.iloc[row_ids].clip(0, 100).to_numpy()
    performance = np.where(np.isfinite(conv), np.minimum(conv, yld), yld)
    y = (performance >= 90.0).astype(int)
    X = X_all.iloc[row_ids].reset_index(drop=True)
    groups = groups_all[row_ids]
    output = []
    print(f"Binary >=90% FDCA screen: {len(y)} records; class counts={np.bincount(y).tolist()}")
    for seed in SEED_LIST[:REPEATS]:
        try:
            folds = grouped_splits(y, groups, seed, n_splits=FOLDS)
        except ValueError as exc:
            for model_name in MODEL_NAMES:
                for sampler_name in SAMPLERS:
                    output.append({
                        "k": 0, "seed": seed, "model": model_name,
                        "sampler": sampler_name, "status": "infeasible",
                        "reason": str(exc), "macro_f1": np.nan,
                        "target_kind": "binary_high_performance_90",
                    })
            continue
        for fold_no, (train_idx, test_idx) in enumerate(folds, start=1):
            for model_name in MODEL_NAMES:
                for sampler_name in SAMPLERS:
                    result = evaluate_fold(
                        X, y, train_idx, test_idx, model_name,
                        sampler_name, seed, 0, fold_no,
                    )
                    result["target_kind"] = "binary_high_performance_90"
                    output.append(result)
    return output


def choose_best_candidate(cluster_summary: pd.DataFrame,
                          binary_summary: pd.DataFrame) -> dict:
    """Choose by mean paper-grouped macro-F1, preserving the target identity."""
    candidates = []
    if not cluster_summary.empty:
        for _, row in cluster_summary.iterrows():
            candidates.append({
                "target_kind": "kmeans_cluster",
                "k": int(row["k"]), "model": str(row["model"]),
                "sampler": str(row["sampler"]),
                "macro_f1_mean": float(row["macro_f1_mean"]),
                "macro_f1_sd": None if pd.isna(row["macro_f1_sd"]) else float(row["macro_f1_sd"]),
                "n_valid_folds": int(row["n_valid"]),
            })
    if not binary_summary.empty:
        for _, row in binary_summary.iterrows():
            candidates.append({
                "target_kind": "binary_high_performance_90",
                "k": None, "model": str(row["model"]),
                "sampler": str(row["sampler"]),
                "macro_f1_mean": float(row["macro_f1_mean"]),
                "macro_f1_sd": None if pd.isna(row["macro_f1_sd"]) else float(row["macro_f1_sd"]),
                "n_valid_folds": int(row["n_valid"]),
            })
    if not candidates:
        return {"status": "no_valid_candidate"}
    best = max(candidates, key=lambda item: item["macro_f1_mean"])
    best["status"] = "screening_candidate"
    return best


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Summarise mean and standard deviation without hiding failed folds."""
    valid = results[results["status"] == "ok"].copy()
    summary = valid.groupby(["k", "model", "sampler"], as_index=False).agg(
        n_valid=("macro_f1", "count"), macro_f1_mean=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", "std"), weighted_f1_mean=("weighted_f1", "mean"),
        accuracy_mean=("accuracy", "mean"),
        precision_macro_mean=("precision_macro", "mean"),
        precision_macro_sd=("precision_macro", "std"),
        recall_macro_mean=("recall_macro", "mean"),
        recall_macro_sd=("recall_macro", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
    )
    return summary.sort_values("macro_f1_mean", ascending=False)


def plot_model_comparison(summary: pd.DataFrame) -> None:
    """Create a K-specific comparison figure for the four sampling methods."""
    if summary.empty:
        return
    for k in K_VALUES:
        subset = summary[summary["k"] == k]
        if subset.empty:
            continue
        pivot = subset.pivot(index="model", columns="sampler", values="macro_f1_mean")
        fig, ax = plt.subplots(figsize=(9, 5))
        pivot.reindex(MODEL_NAMES).plot(kind="bar", ax=ax, width=0.82)
        ax.set_title(f"K={k}: model and sampling comparison")
        ax.set_ylabel("Paper-grouped macro-F1")
        ax.set_xlabel("Classifier")
        ax.set_ylim(0, 1)
        ax.legend(title="Training-fold sampler", ncol=2)
        ax.grid(axis="y", alpha=0.25)
        save_figure(fig, FIGURES_DIR / "classification" / f"model_sampler_K{k}.png")




def generate_paper_style_outputs(cluster_results: pd.DataFrame) -> None:
    """Generate paper-inspired Figure 9, Table 1 and Figure 10.

    Compares ROS, ADASYN and SMOTE using only successful, held-out,
    publication-grouped validation folds. Missing combinations are displayed
    as N/A, never as zero or fabricated values. The no-sampling baseline is
    retained in the main classification summary but excluded from these
    explicitly post-oversampling outputs.
    """
    output_dir = FIGURES_DIR / "paper_style"
    output_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    required = {
        "k", "model", "sampler", "status", "accuracy",
        "precision_macro", "recall_macro", "macro_f1",
    }
    missing = sorted(required - set(cluster_results.columns))
    if missing:
        raise RuntimeError(
            f"Cannot generate paper-style figures; missing metrics: {missing}"
        )

    models = ("DecisionTree", "SVM", "RandomForest", "NeuralNetwork")
    model_labels = {
        "DecisionTree": "DT", "SVM": "SVM",
        "RandomForest": "RF", "NeuralNetwork": "NN",
    }
    colors = {
        "DecisionTree": "#e8a0b5",
        "SVM": "#8ecac8",
        "RandomForest": "#b5a4dc",
        "NeuralNetwork": "#8fb8e8",
    }
    # Include all three oversampling strategies requested by the user.
    methods = ("ROS", "ADASYN", "SMOTE")
    k_values = (3, 4, 5)
    metric_columns = ("accuracy", "precision_macro", "recall_macro", "macro_f1")

    valid = cluster_results[
        cluster_results["status"].eq("ok")
        & cluster_results["sampler"].isin(methods)
        & cluster_results["k"].isin(k_values)
        & cluster_results["model"].isin(models)
    ].copy()

    for column in metric_columns:
        valid[column] = pd.to_numeric(valid[column], errors="coerce")
    valid = valid.dropna(subset=list(metric_columns))

    if valid.empty:
        raise RuntimeError(
            "No valid ROS/ADASYN/SMOTE fold scores were found for K=3, 4, 5. "
            "Check classification_kmeans_fold_scores.csv."
        )

    observed_summary = (
        valid.groupby(["k", "sampler", "model"], as_index=False)
        .agg(
            n_valid_folds=("macro_f1", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            precision_mean=("precision_macro", "mean"),
            precision_sd=("precision_macro", "std"),
            recall_mean=("recall_macro", "mean"),
            recall_sd=("recall_macro", "std"),
            f1_mean=("macro_f1", "mean"),
            f1_sd=("macro_f1", "std"),
        )
    )

    # Full display grid ensures every K/model/sampler combination is present.
    grid = pd.MultiIndex.from_product(
        [k_values, methods, models], names=["k", "sampler", "model"]
    ).to_frame(index=False)
    summary = grid.merge(
        observed_summary, on=["k", "sampler", "model"], how="left"
    )
    summary["n_valid_folds"] = summary["n_valid_folds"].fillna(0).astype(int)

    # Save the exact aggregated values used by all figures and the table.
    summary.to_csv(RESULTS_DIR / "paper_oversampling_summary_all_methods.csv",
                   index=False)
    valid.to_csv(RESULTS_DIR / "paper_oversampling_valid_fold_rows.csv",
                 index=False)

    # ---------------- Figure 9: F1-score comparison ----------------
    # Three panels (ROS, ADASYN, SMOTE), each showing K=3,4,5 and four models.
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), sharey=True)
    x = np.arange(len(k_values))
    width = 0.19

    for ax, method in zip(axes, methods):
        subset = summary[summary["sampler"].eq(method)]
        for j, model in enumerate(models):
            values, errors = [], []
            for k in k_values:
                row = subset[
                    (subset["k"] == k) & (subset["model"] == model)
                ]
                value = row["f1_mean"].iloc[0]
                sd = row["f1_sd"].iloc[0]
                values.append(float(value) if pd.notna(value) else np.nan)
                errors.append(float(sd) if pd.notna(sd) else 0.0)

            bars = ax.bar(
                x + (j - 1.5) * width, values, width,
                yerr=errors, capsize=2, label=model_labels[model],
                color=colors[model], edgecolor="white", linewidth=0.7,
                error_kw={"elinewidth": 1.0},
            )
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        min(value + 0.014, 1.035), f"{value:.3f}",
                        ha="center", va="bottom", fontsize=7, rotation=90,
                    )

        for idx, k in enumerate(k_values):
            group = subset[subset["k"].eq(k)]
            if group["f1_mean"].notna().sum() == 0:
                ax.text(
                    idx, 0.045, "N/A\n(no valid folds)",
                    ha="center", va="bottom", fontsize=8, color="dimgray",
                    bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
                )

        ax.set_xticks(x, [f"K={k}" for k in k_values])
        ax.set_xlim(-0.55, len(k_values) - 0.45)
        ax.set_ylim(0, 1.08)
        ax.set_title(method, fontweight="bold", fontsize=13)
        ax.set_ylabel("Macro-F1 score")
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper right")

    fig.suptitle(
        "Figure 9. F1-scores of the four models after oversampling",
        fontweight="bold", fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(
            output_dir / f"Figure_9_Oversampling_F1.{ext}",
            dpi=600 if ext == "png" else None, bbox_inches="tight",
        )
    plt.close(fig)

    # ---------------- Table 1: all three samplers ----------------
    table = summary.rename(columns={
        "k": "K", "sampler": "Oversampling", "model": "Model",
        "accuracy_mean": "Accuracy", "precision_mean": "Precision",
        "recall_mean": "Recall", "f1_mean": "F1-score",
        "n_valid_folds": "Valid folds",
    }).copy()
    table["Model"] = table["Model"].map(model_labels)
    table = table[
        ["K", "Model", "Oversampling", "Accuracy", "Precision",
         "Recall", "F1-score", "Valid folds"]
    ].sort_values(
        ["K", "Oversampling", "Model"],
        key=lambda col: col.map(
            {"ROS": 0, "ADASYN": 1, "SMOTE": 2}
        ) if col.name == "Oversampling" else col,
        kind="stable",
    )
    table.to_csv(RESULTS_DIR / "paper_table_1_metrics.csv", index=False)

    display = table.copy()
    for column in ("Accuracy", "Precision", "Recall", "F1-score"):
        display[column] = display[column].map(
            lambda value: "N/A" if pd.isna(value) else f"{value:.4f}"
        )

    fig_height = max(8.0, 0.27 * len(display) + 2.0)
    fig, ax = plt.subplots(figsize=(13, fig_height))
    ax.axis("off")
    tbl = ax.table(
        cellText=display.values, colLabels=display.columns,
        cellLoc="center", colLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.2)
    tbl.scale(1, 1.25)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#355c7d")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f1f4f7" if row % 2 else "white")
            if col == 6:
                cell.set_text_props(weight="bold")
            if display.iloc[row - 1]["Valid folds"] == 0:
                cell.set_text_props(color="dimgray", style="italic")
    ax.set_title(
        "Table 1. Model performance after oversampling "
        "(mean held-out publication-grouped scores)",
        fontsize=13, fontweight="bold", pad=15,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "Table_1_Performance_Metrics.png",
        dpi=600, bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "Table_1_Performance_Metrics.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # ---------------- Figure 10: metric profiles, all K and samplers ----------------
    # Three vertically stacked panels keep all K blocks and metric groups readable.
    metric_specs = (
        ("Accuracy", "accuracy_mean"),
        ("Precision", "precision_mean"),
        ("Recall", "recall_mean"),
        ("F1-score", "f1_mean"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True, sharey=True)
    bar_width = 0.17
    block_gap = 0.9

    for ax, method in zip(axes, methods):
        centres, labels = [], []
        block_ranges = []
        cursor = 0.0
        subset_all = summary[summary["sampler"].eq(method)]

        for k in k_values:
            block_start = cursor
            block = subset_all[subset_all["k"].eq(k)]
            has_data = block["f1_mean"].notna().any()

            for metric_label, metric_column in metric_specs:
                centres.append(cursor)
                labels.append(metric_label)
                for model_index, model in enumerate(models):
                    row = block[block["model"].eq(model)]
                    value = row[metric_column].iloc[0]
                    if pd.notna(value):
                        ax.bar(
                            cursor + (model_index - 1.5) * bar_width,
                            float(value), width=bar_width,
                            color=colors[model], edgecolor="white",
                            linewidth=0.5,
                            label=model_labels[model] if (
                                k == k_values[0] and metric_label == "Accuracy"
                            ) else None,
                            zorder=2,
                        )
                cursor += 1.0

            block_end = cursor - 1.0
            block_ranges.append(
                (block_start - 0.48, block_end + 0.48, k, has_data)
            )
            cursor += block_gap

        for block_index, (left, right, k, has_data) in enumerate(block_ranges):
            ax.axvspan(
                left, right,
                color="#eaf1f7" if block_index % 2 == 0 else "#f8eeee",
                alpha=0.65, zorder=0,
            )
            ax.text(
                (left + right) / 2, 1.025, f"K = {k}",
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )
            if not has_data:
                ax.text(
                    (left + right) / 2, 0.50, "N/A\n(no valid folds)",
                    ha="center", va="center", fontsize=9, color="dimgray",
                    bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
                )

        ax.set_xticks(centres, labels, rotation=30, ha="right")
        ax.set_ylim(0, 1.10)
        ax.set_xlim(block_ranges[0][0], block_ranges[-1][1])
        ax.set_ylabel("Mean held-out score")
        ax.set_title(method, fontweight="bold", fontsize=13, loc="left")
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper center")

    axes[-1].set_xlabel("Evaluation metric within each K-means cluster configuration")
    fig.suptitle(
        "Figure 10. Performance evaluation after oversampling",
        fontweight="bold", fontsize=15, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    for ext in ("png", "pdf"):
        fig.savefig(
            output_dir / f"Figure_10_Performance_Metrics.{ext}",
            dpi=600 if ext == "png" else None, bbox_inches="tight",
        )
    plt.close(fig)

    manifest = {
        "source": str(RESULTS_DIR / "classification_kmeans_fold_scores.csv"),
        "valid_fold_rows_used": int(len(valid)),
        "summary_rows": int(len(observed_summary)),
        "display_grid_rows": int(len(summary)),
        "K_values_displayed": list(k_values),
        "models": list(models),
        "oversampling_methods": list(methods),
        "metrics": list(metric_columns),
        "missing_combinations": summary.loc[
            summary["n_valid_folds"].eq(0),
            ["k", "sampler", "model"]
        ].to_dict(orient="records"),
        "metric_definition": (
            "Arithmetic mean of successful held-out fold scores; error bars "
            "show standard deviation across successful folds."
        ),
        "caveat": (
            "K-means labels are descriptive classes. These figures evaluate "
            "classification of those labels, not direct prediction of measured "
            "FDCA yield or causal catalyst performance. N/A means no valid "
            "fold estimate and is not a zero score."
        ),
    }
    (RESULTS_DIR / "paper_style_figures_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        "Generated Figure 9, Table 1 and Figure 10 for ROS, ADASYN and SMOTE "
        f"with K=3,4,5 displayed. Valid held-out fold rows: {len(valid)}; "
        f"displayed combinations: {len(summary)}."
    )

def main() -> None:
    """Run K-means class models plus the paper-inspired binary screen."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    frame = read_processed()
    X, features, _ = build_feature_matrix(frame)
    groups = paper_groups()
    print("=" * 78)
    print("STAGE 04 — PAPER-STYLE CLASSIFICATION MODEL COMPARISON")
    print(f"Rows={len(frame)} | descriptors={len(features)} | repeats={REPEATS} | folds={FOLDS}")
    start = time.time()
    cluster_rows: list[dict] = []
    for k in K_VALUES:
        cluster_rows.extend(run_k(k, X, groups))
    cluster_results = pd.DataFrame(cluster_rows)
    binary_rows = run_binary_high_performance(X, groups)
    binary_results = pd.DataFrame(binary_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cluster_results.to_csv(RESULTS_DIR / "classification_kmeans_fold_scores.csv", index=False)
    binary_results.to_csv(RESULTS_DIR / "classification_binary_fold_scores.csv", index=False)
    # Paper-style figures are generated directly in Stage 04 from these actual fold results.
    generate_paper_style_outputs(cluster_results)
    cluster_summary = summarise(cluster_results)
    binary_summary = summarise(binary_results)
    cluster_summary.to_csv(RESULTS_DIR / "classification_kmeans_summary.csv", index=False)
    binary_summary.to_csv(RESULTS_DIR / "classification_binary_summary.csv", index=False)
    cluster_summary.insert(0, "target_kind", "kmeans_cluster")
    binary_summary.insert(0, "target_kind", "binary_high_performance_90")
    combined = pd.concat([cluster_summary, binary_summary], ignore_index=True)
    combined.to_csv(RESULTS_DIR / "classification_summary.csv", index=False)
    plot_model_comparison(summarise(cluster_results))
    best = choose_best_candidate(summarise(cluster_results), summarise(binary_results))
    write_json(RESULTS_DIR / "best_classification_candidate.json", best)
    elapsed = time.time() - start
    stage_status("04_model_evaluation", "completed", {
        "best_candidate": best, "elapsed_seconds": round(elapsed, 2),
        "kmeans_fold_rows": int(len(cluster_results)),
        "binary_fold_rows": int(len(binary_results)),
        "models": list(MODEL_NAMES), "samplers": list(SAMPLERS),
        "note": "K-means classes are descriptive; binary >=90% is a secondary screening target.",
        "fold_diagnostics": "Per-fold train/test class counts are stored in classification_*_fold_scores.csv.",
        "feature_selection": "Fold-local zero-variance filtering only; no univariate outcome-driven feature ranking.",
    })
    print("\nTop K-means classification candidates:")
    print(summarise(cluster_results).head(8).to_string(index=False))
    print("\nTop >=90% binary screening candidates:")
    print(summarise(binary_results).head(8).to_string(index=False))
    print(f"Selected candidate: {best}")
    print(f"Classification stage finished in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    main()
