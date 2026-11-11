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
from sklearn.feature_selection import VarianceThreshold
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
        ("variance", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
    ]
    sampler = sampler_for(candidate["sampler"])
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", make_classifier(candidate["model"], RANDOM_STATE)))
    return ImbPipeline(steps)


def group_holdout(groups: np.ndarray, y: np.ndarray | None = None):
    """Create a reproducible 20% publication holdout."""
    if len(np.unique(groups)) < 3:
        raise ValueError("SHAP/importance holdout requires at least three independent papers.")
    splitter = GroupShuffleSplit(n_splits=20, test_size=0.20, random_state=RANDOM_STATE)
    best = None
    best_score = -1
    for train_idx, test_idx in splitter.split(np.zeros(len(groups)), groups=groups):
        if not len(train_idx) or not len(test_idx):
            continue
        train_classes = len(np.unique(y[train_idx])) if y is not None else 1
        test_classes = len(np.unique(y[test_idx])) if y is not None else 1
        score = 1000 * train_classes + 10 * test_classes
        if score > best_score:
            best, best_score = (train_idx, test_idx), score
    if best is None:
        raise ValueError("Unable to construct a publication-grouped holdout.")
    return best


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
        transformed_train = pipe.named_steps["variance"].transform(transformed_train)
        transformed_train = pipe.named_steps["scaler"].transform(transformed_train)
        transformed_test = pipe.named_steps["imputer"].transform(X_test)
        transformed_test = pipe.named_steps["variance"].transform(transformed_test)
        transformed_test = pipe.named_steps["scaler"].transform(transformed_test)
        selected_names = np.asarray(feature_names)[pipe.named_steps["variance"].get_support()]
        estimator = pipe.named_steps["model"]
        if model_name in {"DecisionTreeClassifier", "RandomForestClassifier", "GradientBoostingClassifier"}:
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



# Paper-inspired SHAP settings: best Stage 04 multiclass model, Random Forest + ROS.
PAPER_K_VALUES = (3, 5)
SHAP_BACKGROUND_ROWS = 40
SHAP_EXPLAIN_ROWS = 100
SHAP_NSAMPLES = 100
TOP_SHAP_FEATURES = 10


def make_paper_rf_ros_pipe():
    """Build the Stage 04 winning Random Forest + ROS pipeline.

    ROS is fitted only on the training publications. It can handle singleton
    classes by duplication; it does not create independent observations.
    """
    return ImbPipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("sampler", RandomOverSampler(random_state=RANDOM_STATE)),
        ("model", make_classifier("RandomForest", RANDOM_STATE)),
    ])


def group_holdout_for_ros(groups: np.ndarray, y: np.ndarray):
    """Find a publication-grouped split retaining every class in training."""
    if len(np.unique(groups)) < 3:
        raise ValueError("At least three independent publications are required.")
    splitter = GroupShuffleSplit(
        n_splits=250, test_size=0.20, random_state=RANDOM_STATE
    )
    best = None
    best_score = -1
    for train_idx, test_idx in splitter.split(np.zeros(len(groups)), y, groups):
        train_counts = pd.Series(y[train_idx]).value_counts()
        # ROS can operate with one training observation per class, but all
        # classes must remain represented in the training publications.
        if len(train_counts) != len(np.unique(y)) or int(train_counts.min()) < 1:
            continue
        test_classes = len(np.unique(y[test_idx]))
        # Prefer broad class coverage in the held-out papers without sacrificing
        # the minimum training support required by SMOTE.
        score = 100 * test_classes + int(train_counts.min())
        if score > best_score:
            best, best_score = (train_idx, test_idx), score
    if best is None:
        raise ValueError(
            "No publication-grouped split retains every K-means class in training; "
            "RF+ROS SHAP cannot be fitted without leakage."
        )
    return best


def normalize_multiclass_shap(values, n_rows: int, n_features: int):
    """Normalize SHAP's list/array output to (rows, features, classes)."""
    if isinstance(values, list):
        arrays = [np.asarray(v) for v in values]
        arrays = [a.reshape(n_rows, n_features) for a in arrays]
        return np.stack(arrays, axis=2)

    array = np.asarray(values)
    if array.ndim == 2:
        if array.shape != (n_rows, n_features):
            raise ValueError(f"Unexpected SHAP matrix shape: {array.shape}")
        return array[:, :, None]
    if array.ndim != 3:
        raise ValueError(f"Unexpected SHAP array shape: {array.shape}")

    # Current SHAP commonly returns (rows, features, classes).
    if array.shape[0] == n_rows and array.shape[1] == n_features:
        return array
    # Some versions return (classes, rows, features).
    if array.shape[1] == n_rows and array.shape[2] == n_features:
        return np.transpose(array, (1, 2, 0))
    # Other versions may return (rows, classes, features).
    if array.shape[0] == n_rows and array.shape[2] == n_features:
        return np.transpose(array, (0, 2, 1))
    raise ValueError(f"Cannot identify rows/features in SHAP shape {array.shape}")


def select_grouped_shap_sample(X_test: pd.DataFrame, y_test: np.ndarray,
                               max_rows: int, seed: int):
    """Take a reproducible, class-aware subset of held-out rows for explanation."""
    rng = np.random.default_rng(seed)
    chosen = []
    classes = np.unique(y_test)
    # Reserve a few explanation rows for each class where available.
    per_class = max(1, max_rows // max(1, len(classes)))
    for class_id in classes:
        ids = np.flatnonzero(y_test == class_id)
        take = min(len(ids), per_class)
        if take:
            chosen.extend(rng.choice(ids, size=take, replace=False).tolist())
    if len(chosen) < min(max_rows, len(y_test)):
        remaining = np.setdiff1d(np.arange(len(y_test)), np.asarray(chosen, dtype=int))
        extra = min(max_rows - len(chosen), len(remaining))
        if extra:
            chosen.extend(rng.choice(remaining, size=extra, replace=False).tolist())
    chosen = np.asarray(chosen[:max_rows], dtype=int)
    return X_test.iloc[chosen].copy(), y_test[chosen]


def make_beeswarm_panel(ax, shap_matrix: np.ndarray, feature_values: np.ndarray,
                        feature_names: list[str], title: str):
    """Draw a SHAP beeswarm-style panel using actual SHAP values and feature values."""
    from matplotlib.colors import Normalize
    import matplotlib as mpl

    mean_abs = np.mean(np.abs(shap_matrix), axis=0)
    top_idx = np.argsort(mean_abs)[-TOP_SHAP_FEATURES:][::-1]
    cmap = plt.get_cmap("coolwarm")
    norm = Normalize(vmin=0, vmax=1)

    for y_pos, feature_idx in enumerate(top_idx):
        vals = shap_matrix[:, feature_idx]
        raw = feature_values[:, feature_idx]
        finite = np.isfinite(raw)
        if finite.any():
            lo, hi = np.nanpercentile(raw[finite], [5, 95])
            if hi > lo:
                colors = np.clip((raw - lo) / (hi - lo), 0, 1)
            else:
                colors = np.full(len(raw), 0.5)
        else:
            colors = np.full(len(raw), 0.5)

        # Deterministic jitter gives the familiar beeswarm density appearance.
        rng = np.random.default_rng(RANDOM_STATE + feature_idx)
        jitter = rng.normal(0, 0.075, size=len(vals))
        ax.scatter(
            vals, y_pos + jitter, c=colors, cmap=cmap, norm=norm,
            s=12, alpha=0.75, edgecolors="none", rasterized=True,
        )

    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.18)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.045)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.set_label("Feature value", fontsize=8)


def run_paper_rf_ros_shap(X_all: pd.DataFrame, feature_names: list[str]):
    """Generate Figure 12/13 for the Stage 04 winner: RF+ROS, K=3 and K=5."""
    try:
        import shap  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "The requested Figure 12/13 SHAP analyses require the 'shap' package. "
            "Install it in the project virtual environment with: "
            "python -m pip install shap"
        ) from exc

    importance_tables = {}
    beeswarm_payload = {}
    audit = {
        "model": "RandomForest",
        "sampler": "ROS",
        "k_values": list(PAPER_K_VALUES),
        "split": "publication-grouped 80/20 holdout; ROS fitted on training papers only",
        "background_rows_requested": SHAP_BACKGROUND_ROWS,
        "explanation_rows_requested": SHAP_EXPLAIN_ROWS,
        "kernel_shap_nsamples": SHAP_NSAMPLES,
        "interpretation": "Associational feature contributions; not causal effects.",
        "runs": {},
    }

    for k in PAPER_K_VALUES:
        labels = pd.read_csv(RESULTS_DIR / f"kmeans_labels_K{k}.csv")
        labels = labels.sort_values("source_row").reset_index(drop=True)
        row_ids = labels["source_row"].astype(int).to_numpy()
        y = labels["cluster"].astype(int).to_numpy()

        if len(row_ids) != len(X_all) or not np.array_equal(
            row_ids, np.arange(len(X_all))
        ):
            raise ValueError(
                f"K={k} label file does not align with Stage 01 feature rows. "
                "Expected source_row 0..N-1."
            )

        X = X_all.iloc[row_ids].reset_index(drop=True)
        groups = paper_groups()[row_ids]
        train_idx, test_idx = group_holdout_for_ros(groups, y)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe = make_paper_rf_ros_pipe()
        pipe.fit(X_train, y_train)

        # KernelExplainer receives raw input rows and calls the complete fitted
        # pipeline, ensuring the same imputation/scaling/model transformations.
        background_n = min(SHAP_BACKGROUND_ROWS, len(X_train))
        background = X_train.sample(
            n=background_n, random_state=RANDOM_STATE
        ).to_numpy(dtype=float)

        X_explain, y_explain = select_grouped_shap_sample(
            X_test, y_test, SHAP_EXPLAIN_ROWS, RANDOM_STATE + k
        )
        explain_array = X_explain.to_numpy(dtype=float)

        # SHAP's legacy KernelExplainer introspects bound methods and, with
        # recent scikit-learn Pipelines, attempts to assign the read-only
        # feature_names_in_ property. A plain callable wrapper avoids that
        # incompatibility while still routing every prediction through the
        # already-fitted preprocessing + ROS + Random-Forest pipeline.
        def predict_probability(values):
            values = np.asarray(values, dtype=float)
            if values.ndim == 1:
                values = values.reshape(1, -1)
            batch = pd.DataFrame(values, columns=X_train.columns)
            return pipe.predict_proba(batch)

        explainer = shap.KernelExplainer(predict_probability, background)
        raw_values = explainer.shap_values(
            explain_array, nsamples=SHAP_NSAMPLES
        )
        values = normalize_multiclass_shap(
            raw_values, len(explain_array), X.shape[1]
        )

        # Figure 12 aggregates absolute contributions over rows and classes.
        mean_abs = np.mean(np.abs(values), axis=(0, 2))
        importance = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        importance_tables[k] = importance
        importance.to_csv(
            RESULTS_DIR / f"shap_rf_ros_K{k}_feature_importance.csv",
            index=False,
        )

        # Figure 13 uses the most prevalent training class for a readable,
        # explicitly class-specific signed beeswarm distribution.
        model_classes = np.asarray(pipe.named_steps["model"].classes_)
        majority_class = pd.Series(y_train).value_counts().idxmax()
        class_matches = np.flatnonzero(model_classes == majority_class)
        class_idx = int(class_matches[0]) if len(class_matches) else 0
        if values.shape[2] == 1:
            class_idx = 0
        elif class_idx >= values.shape[2]:
            class_idx = 0

        beeswarm_payload[k] = {
            "values": values[:, :, class_idx],
            "features": explain_array,
            "class_label": int(model_classes[class_idx])
                if class_idx < len(model_classes) else int(majority_class),
            "n_explained": int(len(explain_array)),
        }

        audit["runs"][str(k)] = {
            "train_records": int(len(train_idx)),
            "test_records": int(len(test_idx)),
            "train_publications": int(len(np.unique(groups[train_idx]))),
            "test_publications": int(len(np.unique(groups[test_idx]))),
            "train_class_counts_before_smote": {
                str(key): int(value)
                for key, value in pd.Series(y_train).value_counts().sort_index().items()
            },
            "class_counts_after_ros": {
                str(key): int(value)
                for key, value in pd.Series(pipe.named_steps["sampler"].fit_resample(
                    pipe.named_steps["scaler"].transform(
                        pipe.named_steps["variance"].transform(
                            pipe.named_steps["imputer"].transform(X_train)
                        )
                    ), y_train
                )[1]).value_counts().sort_index().items()
            },
            "background_rows": int(background_n),
            "explained_rows": int(len(explain_array)),
            "beeswarm_class": int(beeswarm_payload[k]["class_label"]),
            "shap_value_shape": list(values.shape),
            "ros_strategy": "not majority",
        }
        print(
            f"SHAP RF+ROS K={k}: train={len(train_idx)}, "
            f"held-out={len(test_idx)}, explained={len(explain_array)}"
        )

    # Figure 12: two side-by-side global mean(|SHAP|) bar charts.
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, k in zip(axes, PAPER_K_VALUES):
        top = importance_tables[k].head(TOP_SHAP_FEATURES).sort_values("mean_abs_shap")
        ax.barh(top["feature"], top["mean_abs_shap"])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"K = {k}", fontweight="bold")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "Figure 12. Global SHAP feature importance of Random Forest under ROS",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, FIGURES_DIR / "shap" / "Figure_12_RF_ROS_SHAP_Bar.png")
    fig.savefig(
        FIGURES_DIR / "shap" / "Figure_12_RF_ROS_SHAP_Bar.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # Figure 13: signed SHAP distributions, coloured by the corresponding
    # original feature value (low-to-high), one panel for each K.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, k in zip(axes, PAPER_K_VALUES):
        payload = beeswarm_payload[k]
        make_beeswarm_panel(
            ax, payload["values"], payload["features"],
            feature_names, f"K = {k} (class {payload['class_label']})",
        )
    fig.suptitle(
        "Figure 13. SHAP beeswarm plots of Random Forest under ROS",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, FIGURES_DIR / "shap" / "Figure_13_RF_ROS_SHAP_Beeswarm.png")
    fig.savefig(
        FIGURES_DIR / "shap" / "Figure_13_RF_ROS_SHAP_Beeswarm.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    (RESULTS_DIR / "shap_rf_ros_K3_K5_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return audit

def main() -> None:
    """Fit the selected candidate and explain a bounded held-out sample."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    candidate = load_candidate()
    frame = read_processed()
    X_all, feature_names, _ = build_feature_matrix(frame)
    row_ids, y = candidate_labels(candidate)
    X = X_all.iloc[row_ids].reset_index(drop=True)
    groups = paper_groups()[row_ids]
    train_idx, test_idx = group_holdout(groups, y)
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

    # Additional paper-inspired analysis: Stage 04 winner, Random Forest + ROS,
    # evaluated with the K=3 and K=5 cluster labels.
    paper_audit = run_paper_rf_ros_shap(X_all, feature_names)
    stage_status("06_shap_analysis_paper_nn_smote", "completed", {
        "figures": [
            str(FIGURES_DIR / "shap" / "Figure_12_RF_ROS_SHAP_Bar.png"),
            str(FIGURES_DIR / "shap" / "Figure_13_RF_ROS_SHAP_Beeswarm.png"),
        ],
        "audit": str(RESULTS_DIR / "shap_rf_ros_K3_K5_audit.json"),
        "runs": paper_audit["runs"],
    })
    print("Generated Figure 12 and Figure 13 for the Stage 04 winner: RF + ROS, K=3 and K=5.")


if __name__ == "__main__":
    main()
