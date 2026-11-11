"""Stage 05 — fast paper-grouped learning curves for the selected classifier.

The learning-curve stage consumes stage-04's selected K/model/sampler. It does
not launch a second full grid search. Training fractions are defined over
publication groups, not individual rows, so papers remain isolated between
training and validation. The output is a compact generalisation diagnostic.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from common import (
    RESULTS_DIR, FIGURES_DIR, build_feature_matrix, make_classifier,
    paper_groups, read_cleaned, read_processed, save_figure, stage_status,
)
from importlib.util import spec_from_file_location, module_from_spec

STAGE04 = Path(__file__).resolve().with_name("04_model_evaluation.py")
SPEC = spec_from_file_location("hmf_stage04", STAGE04)
STAGE04_MODULE = module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(STAGE04_MODULE)
FRACTIONS = (0.35, 0.50, 0.65, 0.80)
REPEATS = 2


def load_best_candidate() -> dict:
    """Read the exact classifier selected by stage 04."""
    path = RESULTS_DIR / "best_classification_candidate.json"
    if not path.exists():
        raise FileNotFoundError("Stage 04 candidate is missing; run model evaluation first.")
    candidate = json.loads(path.read_text(encoding="utf-8"))
    if candidate.get("status") != "screening_candidate":
        raise RuntimeError("Stage 04 did not identify a valid classifier candidate.")
    return candidate


def sampler_for(name: str, seed: int):
    """Keep the same sampling definition used by stage 04."""
    if name == "none":
        return None
    if name == "ROS":
        return RandomOverSampler(random_state=seed)
    if name == "SMOTE":
        return SMOTE(k_neighbors=2, random_state=seed)
    if name == "ADASYN":
        return ADASYN(n_neighbors=2, random_state=seed)
    raise ValueError(name)


def make_selected_pipeline(candidate: dict, seed: int, n_features: int = 15):
    """Construct only the already-selected model/sampler pair."""
    steps = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=min(15, n_features))),
    ]
    sampler = sampler_for(candidate["sampler"], seed)
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", make_classifier(candidate["model"], seed)))
    return ImbPipeline(steps)


def group_split(groups: np.ndarray, fraction: float, seed: int):
    """Choose whole papers for training and keep all other papers for testing."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    shuffled = rng.permutation(unique)
    n_train = max(4, min(len(unique) - 2, int(round(len(unique) * fraction))))
    train_groups = set(shuffled[:n_train])
    train_idx = np.flatnonzero(np.isin(groups, list(train_groups)))
    test_idx = np.flatnonzero(~np.isin(groups, list(train_groups)))
    return train_idx, test_idx


def candidate_labels(candidate: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load either the K-means class labels or the >=90% performance labels."""
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
    path = RESULTS_DIR / f"kmeans_labels_K{int(candidate['k'])}.csv"
    labels = pd.read_csv(path)
    return labels.source_row.astype(int).to_numpy(), labels.cluster.astype(int).to_numpy()


def run_curve(X_all: pd.DataFrame, groups_all: np.ndarray, candidate: dict):
    """Fit the chosen candidate over increasing numbers of training papers."""
    row_ids, y = candidate_labels(candidate)
    X = X_all.iloc[row_ids].reset_index(drop=True)
    groups = groups_all[row_ids]
    rows = []
    for repeat in range(REPEATS):
        for fraction in FRACTIONS:
            seed = 4200 + repeat * 100 + int(fraction * 100)
            train_idx, test_idx = group_split(groups, fraction, seed)
            if len(test_idx) < 4 or len(np.unique(y[train_idx])) < 2:
                continue
            try:
                pipe = make_selected_pipeline(candidate, seed, X.shape[1])
                pipe.fit(X.iloc[train_idx], y[train_idx])
                train_pred = pipe.predict(X.iloc[train_idx])
                test_pred = pipe.predict(X.iloc[test_idx])
                rows.append({
                    "repeat": repeat + 1, "fraction": fraction,
                    "train_records": len(train_idx), "test_records": len(test_idx),
                    "train_papers": len(np.unique(groups[train_idx])),
                    "test_papers": len(np.unique(groups[test_idx])),
                    "train_macro_f1": f1_score(y[train_idx], train_pred,
                                                average="macro", zero_division=0),
                    "test_macro_f1": f1_score(y[test_idx], test_pred,
                                               average="macro", zero_division=0),
                })
            except Exception as exc:
                rows.append({"repeat": repeat + 1, "fraction": fraction,
                             "status": "infeasible", "reason": str(exc)[:300]})
    return pd.DataFrame(rows)


def plot_curve(frame: pd.DataFrame, candidate: dict) -> None:
    """Plot training and unseen-paper macro-F1 against training fraction."""
    valid = frame.dropna(subset=["train_macro_f1", "test_macro_f1"])
    if valid.empty:
        return
    summary = valid.groupby("fraction")[
        ["train_macro_f1", "test_macro_f1"]
    ].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for column, label in (("train_macro_f1", "Training papers"),
                          ("test_macro_f1", "Unseen papers")):
        mean = summary[(column, "mean")]
        std = summary[(column, "std")].fillna(0)
        ax.errorbar(summary.index, mean, yerr=std, marker="o", capsize=3, label=label)
    ax.set_xlabel("Fraction of available papers used for training")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Learning curve — K={candidate['k']}, {candidate['model']} + {candidate['sampler']}"
    )
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, FIGURES_DIR / "learning_curves" / "selected_model_learning_curve.png")


def main() -> None:
    """Run one focused learning-curve analysis instead of repeating the grid."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    candidate = load_best_candidate()
    frame = read_processed()
    X, _, _ = build_feature_matrix(frame)
    groups = paper_groups()
    curve = run_curve(X, groups, candidate)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    curve.to_csv(RESULTS_DIR / "learning_curve_selected_model.csv", index=False)
    plot_curve(curve, candidate)
    valid = curve.dropna(subset=["test_macro_f1"])
    details = {
        "candidate": candidate, "valid_curve_points": int(len(valid)),
        "repeats": REPEATS, "fractions": list(FRACTIONS),
        "mean_unseen_paper_f1": float(valid.test_macro_f1.mean()) if len(valid) else None,
    }
    stage_status("05_learning_curves", "completed", details)
    print("Learning-curve results:")
    print(curve.to_string(index=False))


if __name__ == "__main__":
    main()
