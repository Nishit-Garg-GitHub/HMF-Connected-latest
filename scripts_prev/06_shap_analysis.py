"""Stage 06 — explain the selected classifier with SHAP or a safe fallback.

The stage reuses the winning K/model/sampler from stage 04 and the same
K-means class labels. A publication-grouped split is used for fitting. SHAP
values are calculated only for a bounded held-out sample to control runtime.
If SHAP is unavailable or the selected model is unsupported, permutation
importance is saved instead and the report states which method was used.
Feature effects are associational, not causal explanations.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

from common import (
    RESULTS_DIR, FIGURES_DIR, build_feature_matrix, make_classifier,
    paper_groups, read_cleaned, read_processed, save_figure, stage_status,
)

MAX_EXPLAIN_ROWS = 40
RANDOM_STATE = 42


def load_candidate() -> dict:
    """Load the selected model from the classifier stage."""
    path = RESULTS_DIR / "best_classification_candidate.json"
    if not path.exists():
        raise FileNotFoundError("No selected classifier. Run stage 04 first.")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "screening_candidate":
        raise RuntimeError("There is no valid classifier candidate to explain.")
    return result


def candidate_labels(candidate: dict):
    """Read the selected target labels without deriving them from predictors."""
    if candidate.get("target_kind") == "binary_high_performance_90":
        raw = read_cleaned()
        if "Major_Product_Clean" in raw:
            fdca = raw["Major_Product_Clean"].astype(str).str.contains(
                r"FDCA|FURANDICARBOXYLIC", case=False, regex=True, na=False
            )
        else:
            fdca = pd.Series(True, index=raw.index)
        yld = pd.to_numeric(raw["Yield_pct"], errors="coerce")
        conv = pd.to_numeric(raw["Conversion_pct"], errors="coerce")
        keep = fdca & yld.notna()
        row_ids = np.flatnonzero(keep.to_numpy())
        performance = np.where(
            conv.iloc[row_ids].notna().to_numpy(),
            np.minimum(conv.iloc[row_ids].fillna(100).clip(0, 100).to_numpy(),
                       yld.iloc[row_ids].clip(0, 100).to_numpy()),
            yld.iloc[row_ids].clip(0, 100).to_numpy(),
        )
        return row_ids, (performance >= 90).astype(int)
    labels = pd.read_csv(RESULTS_DIR / f"kmeans_labels_K{int(candidate['k'])}.csv")
    return labels.source_row.astype(int).to_numpy(), labels.cluster.astype(int).to_numpy()


def sampler_for(name: str):
    """Recreate the chosen training-only balancing method."""
    if name == "none":
        return None
    if name == "ROS":
        return RandomOverSampler(random_state=RANDOM_STATE)
    if name == "SMOTE":
        return SMOTE(k_neighbors=2, random_state=RANDOM_STATE)
    if name == "ADASYN":
        return ADASYN(n_neighbors=2, random_state=RANDOM_STATE)
    raise ValueError(name)


def make_pipe(candidate: dict, n_features: int = 15):
    """Build the exact selected model family with fold-local preprocessing."""
    steps = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=min(15, n_features))),
    ]
    sampler = sampler_for(candidate["sampler"])
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", make_classifier(candidate["model"], RANDOM_STATE)))
    return ImbPipeline(steps)


def group_holdout(groups: np.ndarray):
    """Create a reproducible 20% publication holdout."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    return next(splitter.split(np.zeros(len(groups)), groups=groups))


def permutation_fallback(pipe, X_test, y_test, feature_names):
    """Calculate held-out permutation importance when SHAP cannot run."""
    result = permutation_importance(
        pipe, X_test, y_test, scoring="f1_macro", n_repeats=3,
        random_state=RANDOM_STATE, n_jobs=1,
    )
    table = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_sd": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    return table, "permutation_importance"


def shap_or_fallback(pipe, X_train, X_test, y_test, feature_names):
    """Use TreeExplainer for tree models; otherwise use permutation fallback."""
    model_name = pipe.named_steps["model"].__class__.__name__
    try:
        import shap
        transformed_train = pipe.named_steps["imputer"].transform(X_train)
        transformed_train = pipe.named_steps["scaler"].transform(transformed_train)
        transformed_train = pipe.named_steps["select"].transform(transformed_train)
        transformed_test = pipe.named_steps["imputer"].transform(X_test)
        transformed_test = pipe.named_steps["scaler"].transform(transformed_test)
        transformed_test = pipe.named_steps["select"].transform(transformed_test)
        selected_names = np.asarray(feature_names)[pipe.named_steps["select"].get_support()]
        estimator = pipe.named_steps["model"]
        if model_name in {"DecisionTreeClassifier", "RandomForestClassifier"}:
            explainer = shap.TreeExplainer(estimator)
            values = explainer.shap_values(transformed_test)
            if isinstance(values, list):
                stacked = np.asarray(values)
                # SHAP versions commonly return (classes, rows, features).
                matrix = np.mean(np.abs(stacked), axis=(0, 1))
            else:
                array = np.asarray(values)
                if array.ndim == 3:
                    # Newer SHAP returns (rows, features, classes).
                    matrix = np.mean(np.abs(array), axis=(0, 2))
                elif array.ndim == 2:
                    matrix = np.mean(np.abs(array), axis=0)
                else:
                    raise ValueError(f"Unexpected SHAP array shape: {array.shape}")
            method = "shap_tree_explainer"
        else:
            # Kernel SHAP is model-agnostic. Bound background/explanation rows
            # and nsamples to keep SVM/MLP explanations tractable.
            background = transformed_train[:min(12, len(transformed_train))]
            explain = transformed_test[:min(12, len(transformed_test))]
            explainer = shap.KernelExplainer(estimator.predict_proba, background)
            values = explainer.shap_values(explain, nsamples=80)
            if isinstance(values, list):
                matrix = np.mean(np.abs(np.asarray(values)), axis=(0, 1))
            else:
                array = np.asarray(values)
                matrix = np.mean(np.abs(array), axis=0)
            method = "shap_kernel_explainer"
        table = pd.DataFrame({
            "feature": selected_names,
            "mean_abs_shap": np.asarray(matrix).reshape(-1)[:len(selected_names)],
        }).sort_values("mean_abs_shap", ascending=False)
        return table, method
    except Exception as exc:
        warnings.warn(f"SHAP failed; using permutation importance: {exc}")
        return permutation_fallback(pipe, X_test, y_test, feature_names)


def plot_importance(table: pd.DataFrame, method: str) -> None:
    """Save a clean top-feature bar chart."""
    value_col = "mean_abs_shap" if "mean_abs_shap" in table else "importance_mean"
    top = table.head(15).sort_values(value_col)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.feature, top[value_col])
    ax.set_xlabel("Mean absolute SHAP value" if value_col == "mean_abs_shap"
                  else "Decrease in held-out macro-F1")
    ax.set_title(f"Feature importance — {method.replace('_', ' ')}")
    ax.grid(axis="x", alpha=0.2)
    save_figure(fig, FIGURES_DIR / "shap" / "selected_model_feature_importance.png")


def main() -> None:
    """Fit the selected candidate and explain a bounded held-out sample."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    candidate = load_candidate()
    frame = read_processed()
    X_all, feature_names, _ = build_feature_matrix(frame)
    row_ids, y = candidate_labels(candidate)
    X = X_all.iloc[row_ids].reset_index(drop=True)
    groups = paper_groups()[row_ids]
    train_idx, test_idx = group_holdout(groups)
    if len(np.unique(y[train_idx])) < 2:
        raise RuntimeError("Training holdout contains fewer than two classes.")
    pipe = make_pipe(candidate, X.shape[1])
    pipe.fit(X.iloc[train_idx], y[train_idx])
    selected = np.arange(len(test_idx))[:MAX_EXPLAIN_ROWS]
    explain_idx = test_idx[selected]
    table, method = shap_or_fallback(
        pipe, X.iloc[train_idx], X.iloc[explain_idx], y[explain_idx], feature_names
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / "feature_importance_selected_model.csv", index=False)
    plot_importance(table, method)
    stage_status("06_shap_analysis", "completed", {
        "candidate": candidate, "method": method,
        "train_papers": int(len(np.unique(groups[train_idx]))),
        "heldout_papers": int(len(np.unique(groups[test_idx]))),
        "explained_rows": int(len(explain_idx)),
        "interpretation": "Associational feature importance; not causal evidence.",
    })
    print(f"Feature interpretation completed using {method}.")
    print(table.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
