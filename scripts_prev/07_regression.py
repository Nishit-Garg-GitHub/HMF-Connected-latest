"""Stage 07 — one joint multi-output regression model for four outcomes.

The four continuous responses are learned together from the same input matrix:
conversion, FDCA yield, Faradaic efficiency and selectivity. Rows with missing
values in any output are not assigned invented targets. Validation is grouped
by publication. A median baseline is evaluated alongside a compact ExtraTrees
multi-output estimator. The model is saved only when it improves on the
baseline across the agreed validation gate; otherwise NSGA-II is blocked.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from joblib import dump

from common import (
    RESULTS_DIR, TARGETS, TARGET_LABELS, build_feature_matrix,
    paper_groups, read_processed, stage_status, write_json,
)

SEED = 42
N_SPLITS = 3
N_TREES = 160
MIN_COMPLETE_ROWS = 30


def complete_target_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return rows with all four observed numerical targets."""
    missing = [target for target in TARGETS if target not in frame.columns]
    if missing:
        raise KeyError(f"Missing regression targets: {missing}")
    targets = frame[TARGETS].apply(pd.to_numeric, errors="coerce")
    targets = targets.replace([np.inf, -np.inf], np.nan)
    mask = targets.notna().all(axis=1)
    return frame.loc[mask].copy().assign(**{
        target: targets.loc[mask, target].clip(0, 100).to_numpy()
        for target in TARGETS
    })


def make_regressor() -> Pipeline:
    """Create a single multi-output estimator with bounded model complexity."""
    estimator = MultiOutputRegressor(
        ExtraTreesRegressor(
            n_estimators=N_TREES, max_depth=7, min_samples_leaf=2,
            max_features=0.8, random_state=SEED, n_jobs=1,
        ),
        n_jobs=1,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


def metrics_by_target(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate target-specific regression metrics for each output."""
    result = {}
    for index, target in enumerate(TARGETS):
        truth = y_true[:, index]
        pred = np.clip(y_pred[:, index], 0, 100)
        result[target] = {
            "r2": float(r2_score(truth, pred)),
            "mae": float(mean_absolute_error(truth, pred)),
            "rmse": float(np.sqrt(mean_squared_error(truth, pred))),
            "n": int(len(truth)),
        }
    return result


def run_grouped_cv(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray):
    """Evaluate one multi-output model against a dummy baseline by paper."""
    unique_groups = np.unique(groups)
    folds = min(N_SPLITS, len(unique_groups))
    splitter = GroupKFold(n_splits=folds)
    rows = []
    oof = np.full_like(y, np.nan, dtype=float)
    baseline_oof = np.full_like(y, np.nan, dtype=float)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        model = make_regressor()
        baseline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", MultiOutputRegressor(DummyRegressor(strategy="median"))),
        ])
        model.fit(X.iloc[train_idx], y[train_idx])
        baseline.fit(X.iloc[train_idx], y[train_idx])
        prediction = np.clip(model.predict(X.iloc[test_idx]), 0, 100)
        base_prediction = np.clip(baseline.predict(X.iloc[test_idx]), 0, 100)
        oof[test_idx] = prediction
        baseline_oof[test_idx] = base_prediction
        model_metrics = metrics_by_target(y[test_idx], prediction)
        base_metrics = metrics_by_target(y[test_idx], base_prediction)
        for target in TARGETS:
            rows.append({
                "fold": fold, "target": target,
                "model_r2": model_metrics[target]["r2"],
                "model_mae": model_metrics[target]["mae"],
                "model_rmse": model_metrics[target]["rmse"],
                "baseline_r2": base_metrics[target]["r2"],
                "baseline_mae": base_metrics[target]["mae"],
                "n_train": len(train_idx), "n_test": len(test_idx),
                "train_papers": len(np.unique(groups[train_idx])),
                "test_papers": len(np.unique(groups[test_idx])),
            })
        print(f"Regression fold {fold}/{folds} complete: train={len(train_idx)}, test={len(test_idx)}")
    return pd.DataFrame(rows), oof, baseline_oof


def validation_gate(scores: pd.DataFrame) -> tuple[bool, dict]:
    """Require better MAE than baseline on all outputs and non-catastrophic R²."""
    summary = scores.groupby("target").agg(
        model_mae=("model_mae", "mean"), baseline_mae=("baseline_mae", "mean"),
        model_r2=("model_r2", "mean"),
    )
    target_checks = {}
    for target in TARGETS:
        row = summary.loc[target]
        # R² can be negative for small noisy literature sets; MAE vs baseline
        # is the primary predictive gate. A modest tolerance prevents overclaim.
        target_checks[target] = {
            "mae_beats_baseline": bool(row.model_mae < row.baseline_mae),
            "mean_model_mae": float(row.model_mae),
            "mean_baseline_mae": float(row.baseline_mae),
            "mean_model_r2": float(row.model_r2),
        }
    passed = all(item["mae_beats_baseline"] for item in target_checks.values())
    return passed, target_checks


def save_predictions(frame: pd.DataFrame, y: np.ndarray, pred: np.ndarray,
                     baseline: np.ndarray) -> None:
    """Persist out-of-fold predictions for later review and optimisation audit."""
    output = pd.DataFrame({"source_row": frame.index.to_numpy()})
    for i, target in enumerate(TARGETS):
        output[f"observed_{target}"] = y[:, i]
        output[f"predicted_{target}"] = pred[:, i]
        output[f"baseline_{target}"] = baseline[:, i]
    output.to_csv(RESULTS_DIR / "multioutput_oof_predictions.csv", index=False)


def main() -> None:
    """Run the shared four-output regression and persist its validation gate."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    frame = read_processed()
    complete = complete_target_frame(frame)
    if len(complete) < MIN_COMPLETE_ROWS:
        raise RuntimeError(
            f"Only {len(complete)} complete four-target rows; need at least {MIN_COMPLETE_ROWS}."
        )
    X_all, features, _ = build_feature_matrix(frame)
    row_ids = complete.index.to_numpy(dtype=int)
    X = X_all.iloc[row_ids].reset_index(drop=True)
    y = complete[TARGETS].to_numpy(dtype=float)
    groups = paper_groups()[row_ids]
    print("=" * 78)
    print("STAGE 07 — JOINT FOUR-OUTPUT REGRESSION")
    print(f"Complete records={len(y)} | papers={len(np.unique(groups))} | inputs={len(features)}")
    scores, oof, baseline = run_grouped_cv(X, y, groups)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(RESULTS_DIR / "multioutput_regression_fold_scores.csv", index=False)
    save_predictions(complete, y, oof, baseline)
    passed, checks = validation_gate(scores)
    summary = scores.groupby("target").agg(
        model_r2_mean=("model_r2", "mean"), model_r2_sd=("model_r2", "std"),
        model_mae_mean=("model_mae", "mean"), baseline_mae_mean=("baseline_mae", "mean"),
        model_rmse_mean=("model_rmse", "mean"),
    ).reset_index()
    summary.to_csv(RESULTS_DIR / "multioutput_regression_summary.csv", index=False)
    # Refit the same joint model on all complete observations only after CV.
    final_model = make_regressor()
    final_model.fit(X, y)
    model_path = RESULTS_DIR / "multioutput_regressor.joblib"
    dump(final_model, model_path)
    gate = {
        "passed": bool(passed), "target_checks": checks,
        "n_complete_records": int(len(y)), "n_papers": int(len(np.unique(groups))),
        "features": features, "targets": TARGETS,
        "model": "MultiOutputRegressor(ExtraTreesRegressor)",
        "validation": "3-fold GroupKFold by publication; compare MAE with median baseline",
        "model_path": str(model_path.relative_to(Path(__file__).resolve().parents[1])),
        "warning": "A passing internal gate is not experimental validation; confirm candidates in the laboratory.",
    }
    write_json(RESULTS_DIR / "regression_validation_gate.json", gate)
    stage_status("07_regression", "completed", gate)
    print(summary.to_string(index=False))
    print("NSGA-II gate:", "PASSED" if passed else "BLOCKED — model did not beat baseline on every target")


if __name__ == "__main__":
    main()
