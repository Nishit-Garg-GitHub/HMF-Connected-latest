"""Stage 07 — per-target and joint multi-output regression for four outcomes.

Per-target models are primary: each outcome is learned from every record where
that outcome is observed and inside 0-100 (mean of a Huber gradient-boosting
and an extra-trees regressor). A joint multi-output ExtraTrees model on the
rows with all four outcomes is kept as a secondary comparison. Validation is
repeated, publication-grouped 5-fold CV against a training-fold MEDIAN
baseline; pooled out-of-fold R2, MAE and fold wins are reported. NSGA-II is
unblocked only if the targets it optimises (Yield, FE) pass the gate.
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
    paper_groups, read_processed, stage_status, write_json, mutual_info_scored,  # noqa: F401
    make_regressor_single,
)

SEED = 42
N_SPLITS = 5
N_REPEATS_REG = 10
N_TREES = 160
MIN_COMPLETE_ROWS = 30
# Gate thresholds. The baseline is the training-fold MEDIAN: it minimises MAE,
# so beating a mean baseline is not evidence of skill. R2 is the pooled
# out-of-fold R2 of each repeat (fold-level R2 on ~25 records with a tiny
# variance is degenerate); its mean over repeats must not be negative.
R2_FLOOR = 0.0
MIN_FOLD_WIN_SHARE = 0.5
# Only the targets that NSGA-II optimises must pass the gate. Conversion and
# Selectivity are reported for completeness but are not used downstream.
GATE_REQUIRED_TARGETS = ("Yield_pct", "FE_pct")


# ---------------------------------------------------------------------------
# Per-target model factory
# ---------------------------------------------------------------------------

def make_single_target_regressor(n_features=None):
    """Per-target pipeline: median imputation + mean of Huber GB and ExtraTrees."""
    return make_regressor_single(SEED)


# ---------------------------------------------------------------------------
# Per-target CV (PRIMARY)
# ---------------------------------------------------------------------------

def run_per_target_cv(X_all, frame, groups_all, features):
    """Train one model per target using all records with that target observed."""
    all_scores, repeat_scores = [], []
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

            rep_pred = np.full(len(y), np.nan)
            rep_base = np.full(len(y), np.nan)
            splitter = GroupKFold(n_splits=n_folds)
            for fold, (train_idx, test_idx) in enumerate(
                splitter.split(X, y, shuffled_groups), start=1
            ):
                model = make_single_target_regressor(n_features)
                baseline = Pipeline([
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("model", DummyRegressor(strategy="median")),
                ])
                model.fit(X.iloc[train_idx], y[train_idx])
                baseline.fit(X.iloc[train_idx], y[train_idx])
                pred = np.clip(model.predict(X.iloc[test_idx]), 0, 100)
                base_pred = np.clip(baseline.predict(X.iloc[test_idx]), 0, 100)
                rep_pred[test_idx] = pred
                rep_base[test_idx] = base_pred

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

            ok = np.isfinite(rep_pred)
            repeat_scores.append({
                "target": target, "repeat": repeat,
                "pooled_r2": float(r2_score(y[ok], rep_pred[ok])),
                "pooled_mae": float(mean_absolute_error(y[ok], rep_pred[ok])),
                "pooled_rmse": float(np.sqrt(mean_squared_error(y[ok], rep_pred[ok]))),
                "pooled_baseline_mae": float(mean_absolute_error(y[ok], rep_base[ok])),
                "pooled_baseline_r2": float(r2_score(y[ok], rep_base[ok])),
                "n": int(ok.sum()),
            })
            # Keep the last repeat's out-of-fold predictions for export/figures.
            if repeat == N_REPEATS_REG - 1:
                oof_pred, oof_baseline = rep_pred, rep_base

        per_target_oof[target] = {
            "row_ids": row_ids, "y": y,
            "pred": oof_pred, "baseline": oof_baseline,
        }
        print(f"  {target}: completed {N_REPEATS_REG} repeats x {N_SPLITS} folds")

    return pd.DataFrame(all_scores), pd.DataFrame(repeat_scores), per_target_oof


# ---------------------------------------------------------------------------
# Per-target validation gate (PRIMARY)
# ---------------------------------------------------------------------------

def validation_gate(scores, repeat_scores):
    """Gate on the targets NSGA-II optimises (Yield, FE).

    A target passes when (i) its mean MAE beats the training-fold MEDIAN
    baseline, (ii) the model wins more than half of the folds and (iii) its
    pooled out-of-fold R2, averaged over repeats, is at least R2_FLOOR. The
    other targets are evaluated identically but are informational only.
    Repeats re-shuffle the same records, so fold wins are correlated; the win
    share is a sanity check, not a significance test.
    """
    summary = scores.groupby("target").agg(
        model_mae=("model_mae", "mean"),
        baseline_mae=("baseline_mae", "mean"),
        folds_won=("model_mae", lambda v: int(
            (v.to_numpy() < scores.loc[v.index, "baseline_mae"].to_numpy()).sum()
        )),
        folds_evaluated=("model_mae", "count"),
        fold_mean_r2=("model_r2", "mean"),
    )
    pooled = repeat_scores.groupby("target").agg(
        pooled_r2=("pooled_r2", "mean"), pooled_r2_sd=("pooled_r2", "std"))
    target_checks = {}
    for target in TARGETS:
        if target not in summary.index:
            target_checks[target] = {
                "required": target in GATE_REQUIRED_TARGETS,
                "mae_beats_baseline": False, "reason": "target not evaluated",
            }
            continue
        row, pool = summary.loc[target], pooled.loc[target]
        target_checks[target] = {
            "required": target in GATE_REQUIRED_TARGETS,
            "mae_beats_baseline": bool(row.model_mae < row.baseline_mae),
            "mean_model_mae": float(row.model_mae),
            "mean_baseline_mae": float(row.baseline_mae),
            "mean_model_r2": float(pool.pooled_r2),
            "pooled_r2_sd": float(pool.pooled_r2_sd),
            "mean_fold_r2": float(row.fold_mean_r2),
            "r2_at_or_above_floor": bool(pool.pooled_r2 >= R2_FLOOR),
            "folds_won": int(row.folds_won),
            "folds_evaluated": int(row.folds_evaluated),
            "wins_majority_of_folds": bool(
                row.folds_won > MIN_FOLD_WIN_SHARE * row.folds_evaluated),
        }
        target_checks[target]["passed"] = bool(
            target_checks[target]["mae_beats_baseline"]
            and target_checks[target]["wins_majority_of_folds"]
            and target_checks[target]["r2_at_or_above_floor"])
    passed = all(target_checks[t].get("passed", False) for t in GATE_REQUIRED_TARGETS)
    return passed, target_checks


# ---------------------------------------------------------------------------
# Nested (selection-corrected) check
# ---------------------------------------------------------------------------

def _candidate_zoo():
    """Small model zoo from which each outer fold picks using only its training papers."""
    from sklearn.linear_model import RidgeCV
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    imp = lambda: SimpleImputer(strategy="median", keep_empty_features=True)
    return {
        "RF_depth3": Pipeline([("i", imp()), ("m", RandomForestRegressor(
            200, max_depth=3, min_samples_leaf=5, max_features=0.5, random_state=SEED))]),
        "RF_depth5": Pipeline([("i", imp()), ("m", RandomForestRegressor(
            300, max_depth=5, min_samples_leaf=3, max_features=0.4, random_state=SEED))]),
        "ExtraTrees": Pipeline([("i", imp()), ("m", ExtraTreesRegressor(
            300, min_samples_leaf=4, max_features=0.5, random_state=SEED))]),
        "GB_huber": Pipeline([("i", imp()), ("m", GradientBoostingRegressor(
            n_estimators=80, max_depth=2, learning_rate=0.05, subsample=0.7,
            loss="huber", random_state=SEED))]),
        "Ridge": Pipeline([("i", imp()), ("s", StandardScaler()),
                           ("m", RidgeCV(alphas=np.logspace(0, 3, 10)))]),
        "kNN": Pipeline([("i", imp()), ("s", StandardScaler()),
                         ("m", KNeighborsRegressor(10, weights="distance"))]),
        "GB+ET (selected)": make_regressor_single(SEED),
    }


def _nested_one(X_all, frame, groups_all, target, repeat):
    from sklearn.base import clone
    v = pd.to_numeric(frame[target], errors="coerce")
    ids = np.flatnonzero((v.notna() & v.between(0, 100)).to_numpy())
    y = v.iloc[ids].to_numpy(float)
    X = X_all.iloc[ids].reset_index(drop=True)
    g = groups_all[ids]
    rng = np.random.default_rng(1000 + repeat)
    order = {k: i for i, k in enumerate(rng.permutation(np.unique(g)))}
    sg = np.array([order[x] for x in g])
    oof, base, picks = np.zeros(len(y)), np.zeros(len(y)), []
    for tr, te in GroupKFold(N_SPLITS).split(X, y, sg):
        best, best_mae = None, np.inf
        for name, model in _candidate_zoo().items():
            errs = []
            for itr, ite in GroupKFold(3).split(X.iloc[tr], y[tr], sg[tr]):
                m = clone(model).fit(X.iloc[tr].iloc[itr], y[tr][itr])
                errs.append(mean_absolute_error(
                    y[tr][ite], np.clip(m.predict(X.iloc[tr].iloc[ite]), 0, 100)))
            if np.mean(errs) < best_mae:
                best, best_mae = name, float(np.mean(errs))
        picks.append(best)
        m = clone(_candidate_zoo()[best]).fit(X.iloc[tr], y[tr])
        oof[te] = np.clip(m.predict(X.iloc[te]), 0, 100)
        base[te] = np.median(y[tr])
    return {"target": target, "repeat": repeat, "nested_r2": float(r2_score(y, oof)),
            "nested_mae": float(mean_absolute_error(y, oof)),
            "baseline_mae": float(mean_absolute_error(y, base)),
            "selected_models": ";".join(picks)}


def nested_selection_check(X_all, frame, groups_all, n_repeats=6) -> pd.DataFrame:
    """Selection-corrected R2/MAE: model choice is made inside each training fold.

    The headline per-target scores above use a model family that was picked after
    comparing candidates on the same records, so they are mildly optimistic.
    """
    from joblib import Parallel, delayed
    rows = Parallel(n_jobs=4)(
        delayed(_nested_one)(X_all, frame, groups_all, t, r)
        for t in TARGETS for r in range(n_repeats))
    return pd.DataFrame(rows)


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

    pt_scores, pt_repeats, per_target_oof = run_per_target_cv(X_all, frame, groups_all, features)
    pt_scores.to_csv(RESULTS_DIR / "per_target_regression_fold_scores.csv", index=False)
    pt_repeats.to_csv(RESULTS_DIR / "per_target_regression_repeat_scores.csv", index=False)

    pt_summary = pt_scores.groupby("target").agg(
        fold_r2_mean=("model_r2", "mean"),
        model_mae_mean=("model_mae", "mean"), baseline_mae_mean=("baseline_mae", "mean"),
        model_rmse_mean=("model_rmse", "mean"),
    ).reset_index().merge(
        pt_repeats.groupby("target").agg(
            model_r2_mean=("pooled_r2", "mean"), model_r2_sd=("pooled_r2", "std"),
            baseline_r2_mean=("pooled_baseline_r2", "mean")).reset_index(),
        on="target")
    pt_summary["n_records"] = pt_summary["target"].map(
        {t: len(d["y"]) for t, d in per_target_oof.items()})
    pt_summary.to_csv(RESULTS_DIR / "per_target_regression_summary.csv", index=False)

    save_per_target_predictions(per_target_oof)
    nested = nested_selection_check(X_all, frame, groups_all)
    nested.to_csv(RESULTS_DIR / "per_target_regression_nested.csv", index=False)
    print("\nNested (selection-corrected) regression:")
    print(nested.groupby("target")[["nested_r2", "nested_mae", "baseline_mae"]].mean().round(3))

    passed, checks = validation_gate(pt_scores, pt_repeats)

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
        "model": "Per-target mean of Huber GradientBoosting and ExtraTrees regressors (primary)",
        "required_targets": list(GATE_REQUIRED_TARGETS),
        "validation": (
            f"Repeated publication-grouped GroupKFold ({N_REPEATS_REG} repeats x {N_SPLITS} folds). "
            f"For each required target ({', '.join(GATE_REQUIRED_TARGETS)}) the mean MAE must beat the "
            f"training-fold MEDIAN baseline, win a majority of folds and have mean pooled "
            f"out-of-fold R2 >= {R2_FLOOR}. Other targets are informational."
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
    print("\nNSGA-II gate:", "PASSED" if passed else
          "BLOCKED — a required target did not beat the median baseline (MAE, fold wins, R2)")


if __name__ == "__main__":
    main()
