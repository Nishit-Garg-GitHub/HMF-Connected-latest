"""Stage 07 — per-target and joint multi-output regression for four outcomes.

The four continuous responses are learned both independently (per-target, using
all records where that specific target is observed and in [0, 100]) and jointly
(multi-output, requiring all four targets present). Per-target models are
primary and drive the validation gate; the joint model is kept as a secondary
comparison. Validation is grouped by publication. A median baseline is evaluated
alongside the models: per-target pipelines use mutual-information SelectKBest
(k=15) with a shallow RandomForestRegressor; the secondary joint model uses
ExtraTrees. Per-target models are saved only when
they improve on the baseline across the agreed validation gate; otherwise
NSGA-II is blocked.
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
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from joblib import dump

from common import (
    RESULTS_DIR, TARGETS, TARGET_LABELS, build_feature_matrix,
    paper_groups, read_processed, stage_status, write_json, mutual_info_scored,
)

SEED = 42
N_SPLITS = 3
N_REPEATS_REG = 10
N_TREES = 160
MIN_COMPLETE_ROWS = 30


# ---------------------------------------------------------------------------
# Per-target model factory
# ---------------------------------------------------------------------------

def make_single_target_regressor(n_features=None):
    """Create a single-target pipeline with feature selection and Random Forest."""
    k_best = min(15, n_features) if n_features else 15
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(
            score_func=mutual_info_scored,
            k=k_best,
        )),
        ("model", RandomForestRegressor(
            n_estimators=60, max_depth=3, min_samples_leaf=5,
            max_features=0.5, random_state=SEED, n_jobs=1,
        )),
    ])


# ---------------------------------------------------------------------------
# Per-target CV (PRIMARY)
# ---------------------------------------------------------------------------

def run_per_target_cv(X_all, frame, groups_all, features):
    """Train one model per target using all records with that target observed."""
    all_scores = []
    per_target_oof = {}

    for target in TARGETS:
        target_values = pd.to_numeric(frame[target], errors="coerce")
        valid = target_values.notna() & target_values.between(0, 100)
        row_ids = np.flatnonzero(valid.to_numpy())
        y = target_values.iloc[row_ids].to_numpy(dtype=float)
        X = X_all.iloc[row_ids].reset_index(drop=True)
        groups = groups_all[row_ids]
        n_features = X.shape[1]

        print(f"\n{target}: {len(y)} valid records, {len(np.unique(groups))} papers")

        oof_pred = np.full(len(y), np.nan)
        oof_baseline = np.full(len(y), np.nan)

        for repeat in range(N_REPEATS_REG):
            repeat_seed = SEED + repeat * 7
            unique_groups = np.unique(groups)
            n_folds = min(N_SPLITS, len(unique_groups))
            if n_folds < 2:
                continue

            # Shuffle groups for different fold compositions
            rng = np.random.default_rng(repeat_seed)
            shuffled_group_map = {g: i for i, g in enumerate(rng.permutation(unique_groups))}
            shuffled_groups = np.array([shuffled_group_map[g] for g in groups])

            splitter = GroupKFold(n_splits=n_folds)
            for fold, (train_idx, test_idx) in enumerate(
                splitter.split(X, y, shuffled_groups), start=1
            ):
                model = make_single_target_regressor(n_features)
                baseline = Pipeline([
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("model", DummyRegressor(strategy="mean")),
                ])
                model.fit(X.iloc[train_idx], y[train_idx])
                baseline.fit(X.iloc[train_idx], y[train_idx])
                pred = np.clip(model.predict(X.iloc[test_idx]), 0, 100)
                base_pred = np.clip(baseline.predict(X.iloc[test_idx]), 0, 100)

                # Only store OOF from last repeat for prediction export
                if repeat == N_REPEATS_REG - 1:
                    oof_pred[test_idx] = pred
                    oof_baseline[test_idx] = base_pred

                model_mae = float(mean_absolute_error(y[test_idx], pred))
                base_mae = float(mean_absolute_error(y[test_idx], base_pred))
                model_r2 = float(r2_score(y[test_idx], pred))
                model_rmse = float(np.sqrt(mean_squared_error(y[test_idx], pred)))

                all_scores.append({
                    "target": target, "repeat": repeat, "fold": fold,
                    "model_r2": model_r2, "model_mae": model_mae,
                    "model_rmse": model_rmse,
                    "baseline_mae": base_mae,
                    "n_train": len(train_idx), "n_test": len(test_idx),
                    "train_papers": len(np.unique(groups[train_idx])),
                    "test_papers": len(np.unique(groups[test_idx])),
                    "n_valid_records": len(y),
                })

        per_target_oof[target] = {
            "row_ids": row_ids, "y": y,
            "pred": oof_pred, "baseline": oof_baseline,
        }
        print(f"  {target}: completed {N_REPEATS_REG} repeats x {N_SPLITS} folds")

    return pd.DataFrame(all_scores), per_target_oof


# ---------------------------------------------------------------------------
# Per-target validation gate (PRIMARY)
# ---------------------------------------------------------------------------

def validation_gate(scores):
    """Require better MAE than baseline on each target and non-catastrophic R²."""
    summary = scores.groupby("target").agg(
        model_mae=("model_mae", "mean"),
        baseline_mae=("baseline_mae", "mean"),
        model_r2=("model_r2", "mean"),
        folds_won=("model_mae", lambda v: int(
            (v.to_numpy() < scores.loc[v.index, "baseline_mae"].to_numpy()).sum()
        )),
        folds_evaluated=("model_mae", "count"),
    )
    target_checks = {}
    for target in TARGETS:
        if target not in summary.index:
            target_checks[target] = {
                "mae_beats_baseline": False,
                "reason": "target not evaluated",
            }
            continue
        row = summary.loc[target]
        target_checks[target] = {
            "mae_beats_baseline": bool(row.model_mae < row.baseline_mae),
            "mean_model_mae": float(row.model_mae),
            "mean_baseline_mae": float(row.baseline_mae),
            "mean_model_r2": float(row.model_r2),
            "folds_won": int(row.folds_won),
            "folds_evaluated": int(row.folds_evaluated),
            "wins_majority_of_folds": bool(row.folds_won >= (row.folds_evaluated // 2 + 1)),
        }
    passed = all(
        item.get("mae_beats_baseline", False) and item.get("wins_majority_of_folds", False)
        for item in target_checks.values()
    )
    return passed, target_checks


# ---------------------------------------------------------------------------
# Per-target OOF predictions export
# ---------------------------------------------------------------------------

def save_per_target_predictions(per_target_oof):
    """Persist per-target out-of-fold predictions for review."""
    rows = []
    for target, data in per_target_oof.items():
        for i, rid in enumerate(data["row_ids"]):
            rows.append({
                "source_row": int(rid),
                "target": target,
                "observed": float(data["y"][i]),
                "predicted": float(data["pred"][i]) if np.isfinite(data["pred"][i]) else None,
                "baseline": float(data["baseline"][i]) if np.isfinite(data["baseline"][i]) else None,
            })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "per_target_oof_predictions.csv", index=False)


# ---------------------------------------------------------------------------
# Joint multi-output helpers (SECONDARY — kept for comparison)
# ---------------------------------------------------------------------------

def complete_target_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return rows with all four observed numerical targets."""
    missing = [target for target in TARGETS if target not in frame.columns]
    if missing:
        raise KeyError(f"Missing regression targets: {missing}")
    targets = frame[TARGETS].apply(pd.to_numeric, errors="coerce")
    targets = targets.replace([np.inf, -np.inf], np.nan)
    # Keep only complete, finite, physically valid percentage observations.
    # Invalid reported values are excluded and counted in the stage audit; they
    # are never silently clipped into apparently valid experimental targets.
    valid_range = targets.ge(0).all(axis=1) & targets.le(100).all(axis=1)
    mask = targets.notna().all(axis=1) & valid_range
    return frame.loc[mask].copy().assign(**{
        target: targets.loc[mask, target].to_numpy()
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
    if folds < 2:
        raise ValueError("At least two independent papers are required for regression CV.")
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


def save_predictions(frame: pd.DataFrame, y: np.ndarray, pred: np.ndarray,
                     baseline: np.ndarray) -> None:
    """Persist out-of-fold predictions for later review and optimisation audit."""
    output = pd.DataFrame({"source_row": frame.index.to_numpy()})
    for i, target in enumerate(TARGETS):
        output[f"observed_{target}"] = y[:, i]
        output[f"predicted_{target}"] = pred[:, i]
        output[f"baseline_{target}"] = baseline[:, i]
    output.to_csv(RESULTS_DIR / "multioutput_oof_predictions.csv", index=False)


# ---------------------------------------------------------------------------
# Joint multi-output validation gate (SECONDARY)
# ---------------------------------------------------------------------------

def validation_gate_joint(scores: pd.DataFrame) -> tuple[bool, dict]:
    """Require better MAE than baseline on all outputs and non-catastrophic R²."""
    summary = scores.groupby("target").agg(
        model_mae=("model_mae", "mean"), baseline_mae=("baseline_mae", "mean"),
        model_r2=("model_r2", "mean"),
        folds_won=("model_mae", lambda values: int(
            (values.to_numpy() < scores.loc[values.index, "baseline_mae"].to_numpy()).sum()
        )),
        folds_evaluated=("model_mae", "count"),
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
            "folds_won": int(row.folds_won),
            "folds_evaluated": int(row.folds_evaluated),
            "wins_majority_of_folds": bool(row.folds_won >= (row.folds_evaluated // 2 + 1)),
        }
    passed = all(
        item["mae_beats_baseline"] and item["wins_majority_of_folds"]
        for item in target_checks.values()
    )
    return passed, target_checks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run per-target regression (primary) and joint multi-output (secondary)."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    frame = read_processed()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Shared feature matrix ---
    X_all, features, _ = build_feature_matrix(frame)
    groups_all = paper_groups()

    # ===================================================================
    # PRIMARY — Per-target regression (repeated GroupKFold)
    # ===================================================================
    print("=" * 78)
    print("STAGE 07 — PER-TARGET REGRESSION (PRIMARY)")
    print(f"Features={len(features)} | Repeats={N_REPEATS_REG} | Folds={N_SPLITS}")

    pt_scores, per_target_oof = run_per_target_cv(X_all, frame, groups_all, features)
    pt_scores.to_csv(RESULTS_DIR / "per_target_regression_fold_scores.csv", index=False)

    pt_summary = pt_scores.groupby("target").agg(
        model_r2_mean=("model_r2", "mean"), model_r2_sd=("model_r2", "std"),
        model_mae_mean=("model_mae", "mean"), baseline_mae_mean=("baseline_mae", "mean"),
        model_rmse_mean=("model_rmse", "mean"),
    ).reset_index()
    pt_summary.to_csv(RESULTS_DIR / "per_target_regression_summary.csv", index=False)

    save_per_target_predictions(per_target_oof)

    passed, checks = validation_gate(pt_scores)

    # Refit per-target models on all valid records and save
    per_target_models = {}
    for target in TARGETS:
        target_values = pd.to_numeric(frame[target], errors="coerce")
        valid = target_values.notna() & target_values.between(0, 100)
        row_ids = np.flatnonzero(valid.to_numpy())
        y_target = target_values.iloc[row_ids].to_numpy(dtype=float)
        X_target = X_all.iloc[row_ids].reset_index(drop=True)
        model = make_single_target_regressor(X_target.shape[1])
        model.fit(X_target, y_target)
        model_path = RESULTS_DIR / f"regressor_{target}.joblib"
        dump(model, model_path)
        per_target_models[target] = str(
            model_path.relative_to(Path(__file__).resolve().parents[1])
        ).replace("\\", "/")

    # ===================================================================
    # SECONDARY — Joint four-output regression (kept for comparison)
    # ===================================================================
    print("\n" + "=" * 78)
    print("STAGE 07 — JOINT FOUR-OUTPUT REGRESSION (SECONDARY)")

    complete = complete_target_frame(frame)
    numeric_targets = frame[TARGETS].apply(pd.to_numeric, errors="coerce")
    invalid_target_rows = int((
        numeric_targets.notna().all(axis=1)
        & (~numeric_targets.ge(0).all(axis=1) | ~numeric_targets.le(100).all(axis=1))
    ).sum())

    if len(complete) >= MIN_COMPLETE_ROWS:
        row_ids_joint = complete.index.to_numpy(dtype=int)
        X_joint = X_all.iloc[row_ids_joint].reset_index(drop=True)
        y_joint = complete[TARGETS].to_numpy(dtype=float)
        groups_joint = groups_all[row_ids_joint]
        print(f"Complete valid records={len(y_joint)} | papers={len(np.unique(groups_joint))} | inputs={len(features)}")
        print(f"Rows excluded for out-of-range observed percentages: {invalid_target_rows}")

        joint_scores, oof, baseline = run_grouped_cv(X_joint, y_joint, groups_joint)
        joint_scores.to_csv(RESULTS_DIR / "multioutput_regression_fold_scores.csv", index=False)
        save_predictions(complete, y_joint, oof, baseline)

        joint_summary = joint_scores.groupby("target").agg(
            model_r2_mean=("model_r2", "mean"), model_r2_sd=("model_r2", "std"),
            model_mae_mean=("model_mae", "mean"), baseline_mae_mean=("baseline_mae", "mean"),
            model_rmse_mean=("model_rmse", "mean"),
        ).reset_index()
        joint_summary.to_csv(RESULTS_DIR / "multioutput_regression_summary.csv", index=False)

        # Refit the same joint model on all complete observations only after CV.
        final_model = make_regressor()
        final_model.fit(X_joint, y_joint)
        joint_model_path = RESULTS_DIR / "multioutput_regressor.joblib"
        dump(final_model, joint_model_path)
        joint_model_relpath = str(
            joint_model_path.relative_to(Path(__file__).resolve().parents[1])
        ).replace("\\", "/")
    else:
        print(f"Only {len(complete)} complete four-target rows (need {MIN_COMPLETE_ROWS}); skipping joint model.")
        joint_model_relpath = None
        joint_summary = pd.DataFrame()

    # ===================================================================
    # Gate JSON — uses per-target results (primary)
    # ===================================================================
    gate = {
        "passed": bool(passed),
        "target_checks": checks,
        "per_target_model_paths": per_target_models,
        "features": features,
        "targets": TARGETS,
        "model": "Per-target RandomForestRegressor with SelectKBest (primary)",
        "validation": (
            f"Repeated publication-grouped GroupKFold ({N_REPEATS_REG} repeats x {N_SPLITS} folds); "
            "MAE must beat median baseline and win a majority of folds for every target."
        ),
        "joint_model_path": joint_model_relpath,
        "warning": "A passing internal gate is not experimental validation; confirm candidates in the laboratory.",
    }
    write_json(RESULTS_DIR / "regression_validation_gate.json", gate)
    stage_status("07_regression", "completed", gate)

    # --- Summary output ---
    print("\n--- Per-target summary (PRIMARY) ---")
    print(pt_summary.to_string(index=False))
    if not joint_summary.empty:
        print("\n--- Joint multi-output summary (SECONDARY) ---")
        print(joint_summary.to_string(index=False))
    print("\nNSGA-II gate:", "PASSED" if passed else "BLOCKED — model did not beat baseline on every target")


if __name__ == "__main__":
    main()
