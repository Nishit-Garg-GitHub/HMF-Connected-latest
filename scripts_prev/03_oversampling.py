"""Stage 03 — inspect fold-local class balancing with ROS, SMOTE and ADASYN.

This stage does not create synthetic records for the full dataset. It audits
sampler feasibility on training partitions only. Stage 04 repeats the same
samplers inside every model-validation fold, so validation records are never
resampled and no synthetic sample can leak into a held-out publication.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

from common import (
    RESULTS_DIR, build_feature_matrix, grouped_splits, load_cluster_labels,
    paper_groups, read_processed, stage_status, write_json,
)

SEED = 42
K_VALUES = (3, 4, 5, 6)
SAMPLERS = ("ROS", "SMOTE", "ADASYN")


def sampler_factory(name: str, seed: int = SEED):
    """Return an imbalanced-learn sampler with conservative neighbours."""
    if name == "ROS":
        return RandomOverSampler(random_state=seed)
    if name == "SMOTE":
        return SMOTE(k_neighbors=2, random_state=seed)
    if name == "ADASYN":
        return ADASYN(n_neighbors=2, random_state=seed)
    raise ValueError(f"Unsupported sampler: {name}")


def count_map(y) -> dict[str, int]:
    """Return class counts as a JSON-friendly mapping."""
    values, counts = np.unique(np.asarray(y), return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def prepare_training_fold(X: pd.DataFrame, train_idx: np.ndarray):
    """Fit imputation/scaling on a training fold and transform only that fold."""
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    train = imputer.fit_transform(X.iloc[train_idx])
    train = scaler.fit_transform(train)
    return train


def audit_one_k(k: int, X: pd.DataFrame, groups: np.ndarray) -> list[dict]:
    """Audit sampler output counts for the first valid paper-grouped fold."""
    labels = load_cluster_labels(k)
    idx = labels["source_row"].astype(int).to_numpy()
    y = labels["cluster"].astype(int).to_numpy()
    Xk = X.iloc[idx].reset_index(drop=True)
    gk = groups[idx]
    rows: list[dict] = []
    try:
        folds = grouped_splits(y, gk, SEED, n_splits=3)
    except ValueError as exc:
        return [{"k": k, "sampler": s, "status": "infeasible", "reason": str(exc)}
                for s in SAMPLERS]
    train_idx, test_idx = folds[0]
    X_train = prepare_training_fold(Xk, train_idx)
    y_train = y[train_idx]
    for name in SAMPLERS:
        before = count_map(y_train)
        try:
            sampler = sampler_factory(name)
            X_res, y_res = sampler.fit_resample(X_train, y_train)
            rows.append({
                "k": k, "sampler": name, "status": "ok",
                "train_before": before, "train_after": count_map(y_res),
                "n_train_before": int(len(y_train)),
                "n_train_after": int(len(y_res)),
                "n_validation_untouched": int(len(test_idx)),
                "note": "Synthetic/resampled data restricted to training fold.",
            })
        except Exception as exc:
            rows.append({
                "k": k, "sampler": name, "status": "infeasible",
                "train_before": before, "reason": str(exc)[:500],
                "n_validation_untouched": int(len(test_idx)),
            })
    return rows


def write_audit(rows: list[dict]) -> None:
    """Save both machine-readable and human-readable balancing diagnostics."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS_DIR / "oversampling_audit.json", rows)
    flat = []
    for row in rows:
        flat.append({
            "k": row.get("k"), "sampler": row.get("sampler"),
            "status": row.get("status"),
            "before": json.dumps(row.get("train_before", {})),
            "after": json.dumps(row.get("train_after", {})),
            "n_before": row.get("n_train_before"),
            "n_after": row.get("n_train_after"),
            "validation_rows": row.get("n_validation_untouched"),
            "reason": row.get("reason", ""),
        })
    pd.DataFrame(flat).to_csv(RESULTS_DIR / "oversampling_audit.csv", index=False)


def print_summary(rows: list[dict]) -> None:
    """Print a compact audit that makes infeasible samplers visible."""
    print("\nFOLD-LOCAL OVERSAMPLING AUDIT")
    print("K | sampler | status | before -> after")
    for row in rows:
        print(
            f"{row.get('k')} | {row.get('sampler')} | {row.get('status')} | "
            f"{row.get('train_before', {})} -> {row.get('train_after', {})}"
        )


def main() -> None:
    """Run balancing diagnostics for K=3,4,5,6 using stage-02 labels."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    frame = read_processed()
    X, features, _ = build_feature_matrix(frame)
    groups = paper_groups()
    rows: list[dict] = []
    print("=" * 72)
    print("STAGE 03 — ROS / SMOTE / ADASYN FEASIBILITY")
    print(f"Records={len(frame)} | engineered descriptors={len(features)}")
    for k in K_VALUES:
        result = audit_one_k(k, X, groups)
        rows.extend(result)
        print(f"Completed sampler audit for K={k}")
    write_audit(rows)
    print_summary(rows)
    stage_status("03_oversampling", "completed", {
        "k_values": list(K_VALUES), "samplers": list(SAMPLERS),
        "feasible": sum(r["status"] == "ok" for r in rows),
        "infeasible": sum(r["status"] != "ok" for r in rows),
    })


if __name__ == "__main__":
    main()
