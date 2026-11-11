
"""
STAGE 05 — REFERENCE-STYLE LEARNING CURVES

4 x 3 layout:
    Rows: Decision Tree, SVM, Random Forest, Neural Network
    Columns: K=3, K=4, K=5

Red:   Training accuracy
Green: Publication-held-out validation accuracy

Also calculates macro-F1 for both training and validation.
All validation splits are grouped by publication.

Outputs:
    figures/learning_curves/HMF_Learning_Curves_Figure.png
    figures/learning_curves/HMF_Learning_Curves_Figure.pdf
    results/learning_curves_all_models.csv
    results/learning_curves_summary.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

from common import (
    RESULTS_DIR,
    FIGURES_DIR,
    build_feature_matrix,
    make_classifier,
    paper_groups,
    read_processed,
    stage_status,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

K_VALUES = [3, 4, 5]

MODEL_NAMES = [
    "DecisionTree",
    "SVM",
    "RandomForest",
    "NeuralNetwork",
]

MODEL_DISPLAY_NAMES = {
    "DecisionTree": "Decision Tree (DT)",
    "SVM": "Support Vector Machine (SVM)",
    "RandomForest": "Random Forest (RF)",
    "NeuralNetwork": "Neural Network (NN)",
}

TRAINING_FRACTIONS = [
    0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90, 1.00
]

N_REPEATS = 5
TEST_SIZE = 0.20
RANDOM_STATE = 42

OUTPUT_DIR = FIGURES_DIR / "learning_curves"


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    processed = read_processed()

    X, _, _ = build_feature_matrix(processed)

    groups = np.asarray(paper_groups())

    if len(X) != len(groups):
        raise ValueError(
            "Feature rows and publication groups do not match."
        )

    return X.reset_index(drop=True), groups


# ============================================================
# 3. LOAD K-MEANS LABELS
# ============================================================

def load_cluster_labels(k):

    path = RESULTS_DIR / f"kmeans_labels_K{k}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run Stage 02 first."
        )

    frame = pd.read_csv(path)

    if not {"source_row", "cluster"}.issubset(frame.columns):
        raise ValueError(
            f"Invalid cluster file for K={k}."
        )

    row_ids = pd.to_numeric(
        frame["source_row"], errors="coerce"
    )

    labels = pd.to_numeric(
        frame["cluster"], errors="coerce"
    )

    valid = row_ids.notna() & labels.notna()

    row_ids = row_ids[valid].astype(int).to_numpy()
    y = labels[valid].astype(int).to_numpy()

    return row_ids, y


# ============================================================
# 4. BUILD MODEL PIPELINE
# ============================================================

def make_learning_pipeline(model_name, seed):

    if model_name == "NeuralNetwork":

        # Disable internal early stopping because very small
        # classes make its internal validation split unreliable.
        model = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            alpha=0.02,
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=False,
            random_state=seed,
        )

    else:
        model = make_classifier(model_name, seed)

    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        ),
        (
            "variance",
            VarianceThreshold(threshold=0.0),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        ("model", model),
    ])


# ============================================================
# 5. SELECT A GOOD PUBLICATION-GROUPED OUTER SPLIT
# ============================================================

def make_outer_splits(X, y, groups, k):

    splitter = GroupShuffleSplit(
        n_splits=50,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE + k,
    )

    candidates = []

    all_classes = set(np.unique(y))

    for train_idx, test_idx in splitter.split(X, y, groups):

        train_classes = set(np.unique(y[train_idx]))
        test_classes = set(np.unique(y[test_idx]))

        if len(train_classes) < 2:
            continue

        # Prefer broad training-class coverage and a test set
        # containing as many observed classes as possible.
        train_coverage = len(train_classes) / max(k, 1)
        test_coverage = len(test_classes) / max(k, 1)

        size_score = 1.0 - abs(
            len(test_idx) / len(y) - TEST_SIZE
        )

        score = (
            5.0 * train_coverage
            + 2.0 * test_coverage
            + size_score
        )

        candidates.append(
            (score, train_idx, test_idx)
        )

    if not candidates:
        raise ValueError(
            f"K={k}: no valid publication-grouped splits."
        )

    # Use the best available split for this repeat.
    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return candidates[0][1], candidates[0][2]


# ============================================================
# 6. SELECT TRAINING PUBLICATIONS
# ============================================================

def select_training_subset(
    outer_train,
    groups,
    y,
    fraction,
    seed,
):
    """
    Select whole publications while trying to preserve class coverage.

    A publication is never divided between training and testing.
    """

    rng = np.random.default_rng(seed)

    available_groups = np.unique(groups[outer_train])

    target_records = max(
        2,
        int(round(len(outer_train) * fraction)),
    )

    group_records = {
        group: outer_train[groups[outer_train] == group]
        for group in available_groups
    }

    group_classes = {
        group: set(np.unique(y[group_records[group]]))
        for group in available_groups
    }

    selected_groups = []
    selected_records = []
    covered_classes = set()

    remaining_groups = list(available_groups)
    rng.shuffle(remaining_groups)

    # First try to include groups carrying currently missing classes.
    all_training_classes = set(np.unique(y[outer_train]))

    while remaining_groups:

        best_group = None
        best_score = -float("inf")

        for group in remaining_groups:

            new_classes = (
                group_classes[group] - covered_classes
            )

            n_records = len(group_records[group])

            score = (
                10.0 * len(new_classes)
                - 0.05 * n_records
            )

            if score > best_score:
                best_score = score
                best_group = group

        if best_group is None:
            break

        # Continue adding groups until the target sample size
        # is reached and at least two classes are represented.
        if (
            len(selected_records) >= target_records
            and len(covered_classes) >= 2
        ):
            break

        selected_groups.append(best_group)

        selected_records.extend(
            group_records[best_group].tolist()
        )

        covered_classes.update(
            group_classes[best_group]
        )

        remaining_groups.remove(best_group)

        if covered_classes == all_training_classes:
            if len(selected_records) >= target_records:
                break

    selected_records = np.asarray(
        selected_records,
        dtype=int,
    )

    return selected_records


# ============================================================
# 7. CALCULATE SCORES
# ============================================================

def calculate_scores(
    X,
    y,
    train_idx,
    test_idx,
    model_name,
    seed,
    k,
):

    pipeline = make_learning_pipeline(
        model_name,
        seed,
    )

    pipeline.fit(
        X.iloc[train_idx],
        y[train_idx],
    )

    train_predictions = pipeline.predict(
        X.iloc[train_idx]
    )

    test_predictions = pipeline.predict(
        X.iloc[test_idx]
    )

    # Fixed class labels prevent macro-F1 from silently
    # dropping absent classes in a particular validation fold.
    fixed_labels = np.arange(k)

    train_accuracy = accuracy_score(
        y[train_idx],
        train_predictions,
    )

    validation_accuracy = accuracy_score(
        y[test_idx],
        test_predictions,
    )

    train_macro_f1 = f1_score(
        y[train_idx],
        train_predictions,
        labels=fixed_labels,
        average="macro",
        zero_division=0,
    )

    validation_macro_f1 = f1_score(
        y[test_idx],
        test_predictions,
        labels=fixed_labels,
        average="macro",
        zero_division=0,
    )

    return (
        train_accuracy,
        validation_accuracy,
        train_macro_f1,
        validation_macro_f1,
    )


# ============================================================
# 8. RUN ONE K VALUE
# ============================================================

def run_one_k(X_all, groups_all, k):

    row_ids, y = load_cluster_labels(k)

    if row_ids.min() < 0 or row_ids.max() >= len(X_all):
        raise ValueError(
            f"K={k}: invalid source_row indices."
        )

    X = X_all.iloc[row_ids].reset_index(drop=True)
    groups = groups_all[row_ids]

    print(
        f"\nK={k}: {len(X)} records, "
        f"{len(np.unique(groups))} publications"
    )

    print(
        "Class distribution:",
        pd.Series(y).value_counts().sort_index().to_dict(),
    )

    all_rows = []

    for repeat in range(1, N_REPEATS + 1):

        seed = RANDOM_STATE + k * 100 + repeat

        outer_train, outer_test = make_outer_splits(
            X,
            y,
            groups,
            k,
        )

        for model_name in MODEL_NAMES:

            print(
                f"  K={k}, {MODEL_DISPLAY_NAMES[model_name]}, "
                f"repeat={repeat}"
            )

            for fraction in TRAINING_FRACTIONS:

                subset_seed = (
                    seed + int(fraction * 1000)
                )

                train_idx = select_training_subset(
                    outer_train,
                    groups,
                    y,
                    fraction,
                    subset_seed,
                )

                if len(train_idx) < 2:
                    continue

                if len(np.unique(y[train_idx])) < 2:
                    warnings.warn(
                        f"K={k}, {model_name}, "
                        f"fraction={fraction}: fewer than two "
                        "training classes; point skipped."
                    )
                    continue

                try:

                    scores = calculate_scores(
                        X=X,
                        y=y,
                        train_idx=train_idx,
                        test_idx=outer_test,
                        model_name=model_name,
                        seed=subset_seed,
                        k=k,
                    )

                    train_acc, test_acc, train_f1, test_f1 = scores

                    all_rows.append({
                        "K": k,
                        "model": model_name,
                        "repeat": repeat,
                        "training_fraction": fraction,
                        "training_records": len(train_idx),
                        "validation_records": len(outer_test),
                        "training_publications": len(
                            np.unique(groups[train_idx])
                        ),
                        "validation_publications": len(
                            np.unique(groups[outer_test])
                        ),
                        "training_accuracy": train_acc,
                        "validation_accuracy": test_acc,
                        "training_macro_f1": train_f1,
                        "validation_macro_f1": test_f1,
                        "status": "completed",
                    })

                except Exception as exc:

                    warnings.warn(
                        f"K={k}, {model_name}, "
                        f"repeat={repeat}, fraction={fraction}: "
                        f"{exc}"
                    )

                    all_rows.append({
                        "K": k,
                        "model": model_name,
                        "repeat": repeat,
                        "training_fraction": fraction,
                        "status": "failed",
                        "reason": str(exc)[:300],
                    })

    return pd.DataFrame(all_rows)


# ============================================================
# 9. SUMMARISE RESULTS
# ============================================================

def create_summary(results):

    valid = results[
        results["status"] == "completed"
    ].copy()

    if valid.empty:
        raise RuntimeError(
            "No valid learning-curve results were generated."
        )

    summary = (
        valid.groupby(
            ["K", "model", "training_fraction"],
            as_index=False,
        )
        .agg(
            mean_training_records=("training_records", "mean"),

            mean_training_accuracy=("training_accuracy", "mean"),
            sd_training_accuracy=("training_accuracy", "std"),

            mean_validation_accuracy=("validation_accuracy", "mean"),
            sd_validation_accuracy=("validation_accuracy", "std"),

            mean_training_macro_f1=("training_macro_f1", "mean"),
            mean_validation_macro_f1=("validation_macro_f1", "mean"),

            valid_repeats=("repeat", "nunique"),
        )
    )

    for column in [
        "sd_training_accuracy",
        "sd_validation_accuracy",
    ]:
        summary[column] = summary[column].fillna(0)

    return summary


# ============================================================
# 10. CREATE THE 4 x 3 REFERENCE-STYLE FIGURE
# ============================================================

def plot_learning_curves(summary):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(15, 17),
        sharey=True,
    )

    panel_labels = list("abcdefghijkl")
    panel = 0

    for row, model_name in enumerate(MODEL_NAMES):

        for col, k in enumerate(K_VALUES):

            ax = axes[row, col]

            subset = summary[
                (summary["model"] == model_name)
                & (summary["K"] == k)
            ].sort_values("training_fraction")

            if subset.empty:

                ax.text(
                    0.5,
                    0.5,
                    "No valid observations",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

                panel += 1
                continue

            x = subset[
                "mean_training_records"
            ].to_numpy()

            train_mean = subset[
                "mean_training_accuracy"
            ].to_numpy()

            train_std = subset[
                "sd_training_accuracy"
            ].to_numpy()

            validation_mean = subset[
                "mean_validation_accuracy"
            ].to_numpy()

            validation_std = subset[
                "sd_validation_accuracy"
            ].to_numpy()

            # Red: training score.
            ax.plot(
                x,
                train_mean,
                color="red",
                marker="o",
                markersize=4,
                linewidth=1.5,
                label="Training score",
            )

            ax.fill_between(
                x,
                np.clip(train_mean - train_std, 0, 1),
                np.clip(train_mean + train_std, 0, 1),
                color="red",
                alpha=0.10,
            )

            # Green: publication-held-out validation score.
            ax.plot(
                x,
                validation_mean,
                color="green",
                marker="o",
                markersize=4,
                linewidth=1.5,
                label="Cross-validation score",
            )

            ax.fill_between(
                x,
                np.clip(validation_mean - validation_std, 0, 1),
                np.clip(validation_mean + validation_std, 0, 1),
                color="green",
                alpha=0.10,
            )

            ax.set_title(
                f"{MODEL_DISPLAY_NAMES[model_name]}, K={k}",
                fontsize=10,
                fontweight="bold",
            )

            ax.set_xlabel(
                "Training examples",
                fontsize=9,
            )

            ax.set_ylabel(
                "Accuracy score",
                fontsize=9,
            )

            ax.set_ylim(0, 1.05)

            ax.grid(
                True,
                linestyle="--",
                linewidth=0.5,
                alpha=0.3,
            )

            ax.text(
                0.96,
                0.06,
                f"({panel_labels[panel]})",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

            if row == 0 and col == 0:
                ax.legend(
                    loc="lower right",
                    fontsize=7,
                    frameon=True,
                )

            panel += 1

    fig.suptitle(
        "Learning Curves for HMF Oxidation to FDCA",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    fig.tight_layout(
        rect=[0.02, 0.02, 1, 0.98],
        h_pad=2,
        w_pad=1.5,
    )

    png_path = (
        OUTPUT_DIR / "HMF_Learning_Curves_Figure.png"
    )

    pdf_path = (
        OUTPUT_DIR / "HMF_Learning_Curves_Figure.pdf"
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nFigure saved:")
    print(png_path)

    print("\nPDF saved:")
    print(pdf_path)




# ============================================================
# 12. OVERSAMPLED NEURAL-NETWORK LEARNING CURVES (FIGURE 11)
# ============================================================

OVERSAMPLING_K_VALUES = [3, 4, 5, 6]
OVERSAMPLING_METHODS = ["ROS", "ADASYN", "SMOTE"]


def fit_oversampled_nn_scores(X, y, train_idx, test_idx, seed, k, method):
    """Fit preprocessing and sampler on training data only; score original rows."""
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import StandardScaler

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Every transform is fitted exclusively on this training subset.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    selector = VarianceThreshold(threshold=0.0)
    X_train_sel = selector.fit_transform(X_train_imp)
    X_test_sel = selector.transform(X_test_imp)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    class_counts = pd.Series(y_train).value_counts()
    min_count = int(class_counts.min())

    if method == "ROS":
        sampler = RandomOverSampler(random_state=seed)
    elif method == "SMOTE":
        if min_count < 2:
            raise ValueError(
                f"SMOTE requires at least 2 training observations per class; "
                f"minimum class count is {min_count}."
            )
        sampler = SMOTE(
            k_neighbors=min(2, min_count - 1),
            random_state=seed,
        )
    elif method == "ADASYN":
        if min_count < 2:
            raise ValueError(
                f"ADASYN requires at least 2 training observations per class; "
                f"minimum class count is {min_count}."
            )
        sampler = ADASYN(
            n_neighbors=min(2, min_count - 1),
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown sampler: {method}")

    X_resampled, y_resampled = sampler.fit_resample(
        X_train_scaled, y_train
    )

    # Keep NN architecture and regularisation aligned with the original Stage 05.
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        alpha=0.02,
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(X_resampled, y_resampled)

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    fixed_labels = np.arange(k)

    return {
        "training_macro_f1": f1_score(
            y_train, train_pred, labels=fixed_labels,
            average="macro", zero_division=0
        ),
        "validation_macro_f1": f1_score(
            y_test, test_pred, labels=fixed_labels,
            average="macro", zero_division=0
        ),
        "training_accuracy": accuracy_score(y_train, train_pred),
        "validation_accuracy": accuracy_score(y_test, test_pred),
        "resampled_records": int(len(y_resampled)),
        "minimum_training_class_count": min_count,
    }


def run_oversampled_nn_learning_curves(X_all, groups_all):
    """Run the additional reference-inspired NN curves for K=3,4,5,6."""
    rows = []

    for k in OVERSAMPLING_K_VALUES:
        row_ids, y = load_cluster_labels(k)
        if len(row_ids) == 0:
            continue
        if row_ids.min() < 0 or row_ids.max() >= len(X_all):
            raise ValueError(f"K={k}: invalid source_row indices.")

        X = X_all.iloc[row_ids].reset_index(drop=True)
        groups = groups_all[row_ids]

        print(
            f"\nOversampled NN curves K={k}: {len(X)} rows, "
            f"{len(np.unique(groups))} publications; "
            f"class counts={pd.Series(y).value_counts().sort_index().to_dict()}"
        )

        for repeat in range(1, N_REPEATS + 1):
            seed = RANDOM_STATE + k * 100 + repeat
            outer_train, outer_test = make_outer_splits(X, y, groups, k)

            for fraction in TRAINING_FRACTIONS:
                subset_seed = seed + int(fraction * 1000)
                train_idx = select_training_subset(
                    outer_train, groups, y, fraction, subset_seed
                )
                if len(train_idx) < 2 or len(np.unique(y[train_idx])) < 2:
                    continue

                for method in OVERSAMPLING_METHODS:
                    record = {
                        "K": k,
                        "model": "NeuralNetwork",
                        "sampler": method,
                        "repeat": repeat,
                        "training_fraction": fraction,
                        "training_records": int(len(train_idx)),
                        "validation_records": int(len(outer_test)),
                        "training_publications": int(len(np.unique(groups[train_idx]))),
                        "validation_publications": int(len(np.unique(groups[outer_test]))),
                    }
                    try:
                        scores = fit_oversampled_nn_scores(
                            X, y, train_idx, outer_test,
                            subset_seed, k, method
                        )
                        record.update(scores)
                        record["status"] = "completed"
                    except Exception as exc:
                        record["status"] = "infeasible"
                        record["reason"] = str(exc)[:350]
                    rows.append(record)

        print(f"Completed oversampled NN learning-curve audit for K={k}")

    return pd.DataFrame(rows)


def summarise_oversampled_nn_curves(results):
    valid = results[results["status"] == "completed"].copy()
    if valid.empty:
        raise RuntimeError(
            "No valid oversampled NN learning-curve points were produced. "
            "Check the K-means label files and class counts."
        )
    summary = (
        valid.groupby(
            ["K", "sampler", "training_fraction"], as_index=False
        )
        .agg(
            mean_training_records=("training_records", "mean"),
            mean_training_macro_f1=("training_macro_f1", "mean"),
            sd_training_macro_f1=("training_macro_f1", "std"),
            mean_validation_macro_f1=("validation_macro_f1", "mean"),
            sd_validation_macro_f1=("validation_macro_f1", "std"),
            mean_resampled_records=("resampled_records", "mean"),
            valid_repeats=("repeat", "nunique"),
        )
    )
    for col in ["sd_training_macro_f1", "sd_validation_macro_f1"]:
        summary[col] = summary[col].fillna(0.0)
    return summary


def plot_oversampled_nn_figure11(summary):
    """Create Figure 11: rows K=3..6, columns ROS/ADASYN/SMOTE."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        4, 3, figsize=(15.5, 16.5), sharex=False, sharey=True
    )
    panel_labels = list("abcdefghijkl")
    panel = 0

    for row, k in enumerate(OVERSAMPLING_K_VALUES):
        for col, method in enumerate(OVERSAMPLING_METHODS):
            ax = axes[row, col]
            subset = summary[
                (summary["K"] == k) & (summary["sampler"] == method)
            ].sort_values("mean_training_records")

            if subset.empty:
                ax.text(
                    0.5, 0.5, "N/A\n(no feasible training points)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="dimgray"
                )
            else:
                x = subset["mean_training_records"].to_numpy()
                train = subset["mean_training_macro_f1"].to_numpy()
                valid = subset["mean_validation_macro_f1"].to_numpy()
                train_sd = subset["sd_training_macro_f1"].to_numpy()
                valid_sd = subset["sd_validation_macro_f1"].to_numpy()

                ax.plot(
                    x, train, color="red", marker="o", markersize=3.5,
                    linewidth=1.4, label="Training score"
                )
                ax.plot(
                    x, valid, color="green", marker="o", markersize=3.5,
                    linewidth=1.4, label="Cross-validation score"
                )
                ax.fill_between(
                    x, np.clip(train - train_sd, 0, 1),
                    np.clip(train + train_sd, 0, 1),
                    color="red", alpha=0.10
                )
                ax.fill_between(
                    x, np.clip(valid - valid_sd, 0, 1),
                    np.clip(valid + valid_sd, 0, 1),
                    color="green", alpha=0.10
                )

            ax.set_title(f"K={k} | {method}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Training examples", fontsize=8)
            ax.set_ylabel("Macro-F1 score", fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.3)
            ax.text(
                0.96, 0.06, f"({panel_labels[panel]})",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11, fontweight="bold"
            )
            if panel == 0:
                ax.legend(loc="upper right", fontsize=7, frameon=True)
            panel += 1

    fig.suptitle(
        "Figure 11. Learning curves of the oversampled neural network",
        fontsize=14, fontweight="bold", y=0.995
    )
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.975], h_pad=1.8, w_pad=1.3)

    png_path = OUTPUT_DIR / "Figure_11_Oversampled_NN_Learning_Curves_ROS_ADASYN_SMOTE.png"
    pdf_path = OUTPUT_DIR / "Figure_11_Oversampled_NN_Learning_Curves_ROS_ADASYN_SMOTE.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print("\nFigure 11 saved:")
    print(png_path)
    print(pdf_path)


def run_oversampled_nn_stage(X, groups):
    results = run_oversampled_nn_learning_curves(X, groups)
    raw_path = RESULTS_DIR / "learning_curves_oversampled_nn_fold_results.csv"
    results.to_csv(raw_path, index=False)

    summary = summarise_oversampled_nn_curves(results)
    summary_path = RESULTS_DIR / "learning_curves_oversampled_nn_summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_oversampled_nn_figure11(summary)

    audit = {
        "K_values": OVERSAMPLING_K_VALUES,
        "samplers": OVERSAMPLING_METHODS,
        "training_fractions": TRAINING_FRACTIONS,
        "repeats": N_REPEATS,
        "total_fold_fraction_sampler_rows": int(len(results)),
        "completed_points": int((results["status"] == "completed").sum()),
        "infeasible_points": int((results["status"] != "completed").sum()),
        "infeasibility_reasons": (
            results.loc[results["status"] != "completed", "reason"]
            .value_counts().to_dict()
            if "reason" in results else {}
        ),
        "methodology": (
            "Publication-grouped outer split; fold-local imputation, "
            "variance filtering, scaling and training-only ROS/ADASYN/SMOTE. "
            "Training and validation scores are calculated on original, "
            "non-resampled observations."
        ),
    }
    (RESULTS_DIR / "learning_curves_oversampled_nn_audit.json").write_text(
        __import__("json").dumps(audit, indent=2), encoding="utf-8"
    )
    print(
        f"Figure 11 audit: {audit['completed_points']} completed points; "
        f"{audit['infeasible_points']} infeasible points."
    )



# ============================================================
# 11. MAIN
# ============================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )

    print("=" * 75)
    print("STAGE 05 — REFERENCE-STYLE LEARNING CURVES")
    print("=" * 75)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X, groups = load_data()

    print(f"Total feature records: {len(X)}")
    print(
        f"Independent publications: "
        f"{len(np.unique(groups))}"
    )

    all_results = []

    for k in K_VALUES:

        result = run_one_k(
            X_all=X,
            groups_all=groups,
            k=k,
        )

        all_results.append(result)

    results = pd.concat(
        all_results,
        ignore_index=True,
    )

    raw_path = (
        RESULTS_DIR / "learning_curves_all_models.csv"
    )

    results.to_csv(
        raw_path,
        index=False,
    )

    summary = create_summary(results)

    summary_path = (
        RESULTS_DIR / "learning_curves_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    plot_learning_curves(summary)

    # Additional Figure 11: oversampled NN curves for ROS, ADASYN and SMOTE.
    run_oversampled_nn_stage(X, groups)

    valid = results[
        results["status"] == "completed"
    ]

    details = {
        "models": MODEL_NAMES,
        "k_values": K_VALUES,
        "repeats": N_REPEATS,
        "training_fractions": TRAINING_FRACTIONS,
        "total_feature_records": len(X),
        "independent_publications": len(np.unique(groups)),
        "valid_points": len(valid),
        "mean_validation_accuracy": (
            float(valid["validation_accuracy"].mean())
            if len(valid) else None
        ),
        "mean_validation_macro_f1": (
            float(valid["validation_macro_f1"].mean())
            if len(valid) else None
        ),
    }

    stage_status(
        "05_learning_curves",
        "completed",
        details,
    )

    print("\nSummary:")
    print(summary.to_string(index=False))

    print("\n" + "=" * 75)
    print("STAGE 05 COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
