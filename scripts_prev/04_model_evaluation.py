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
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.feature_selection import SelectKBest, f_classif

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
            "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], prediction)),
            "classes_train": int(len(np.unique(y[train_idx]))),
            "classes_test": int(len(np.unique(y[test_idx]))),
        })
    except Exception as exc:
        record.update({"status": "infeasible", "reason": str(exc)[:500]})
        record.update({"macro_f1": np.nan, "weighted_f1": np.nan,
                       "accuracy": np.nan, "balanced_accuracy": np.nan})
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
        folds = grouped_splits(y, groups, seed, n_splits=FOLDS)
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
        folds = grouped_splits(y, groups, seed, n_splits=FOLDS)
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
    })
    print("\nTop K-means classification candidates:")
    print(summarise(cluster_results).head(8).to_string(index=False))
    print("\nTop >=90% binary screening candidates:")
    print(summarise(binary_results).head(8).to_string(index=False))
    print(f"Selected candidate: {best}")
    print(f"Classification stage finished in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    main()
