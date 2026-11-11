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
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
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
    """Audit sampler feasibility independently in every paper-grouped fold."""
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
    for fold_no, (train_idx, test_idx) in enumerate(folds, start=1):
        X_train = prepare_training_fold(Xk, train_idx)
        y_train = y[train_idx]
        for name in SAMPLERS:
            before = count_map(y_train)
            try:
                sampler = sampler_factory(name)
                X_res, y_res = sampler.fit_resample(X_train, y_train)
                rows.append({
                    "k": k, "fold": fold_no, "sampler": name, "status": "ok",
                    "train_before": before, "train_after": count_map(y_res),
                    "n_train_before": int(len(y_train)),
                    "n_train_after": int(len(y_res)),
                    "n_validation_untouched": int(len(test_idx)),
                    "note": "Resampling restricted to this training fold.",
                })
            except Exception as exc:
                rows.append({
                    "k": k, "fold": fold_no, "sampler": name, "status": "infeasible",
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
            "k": row.get("k"), "fold": row.get("fold"), "sampler": row.get("sampler"),
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
    print("K | fold | sampler | status | before -> after")
    for row in rows:
        print(
            f"{row.get('k')} | {row.get('fold', '—')} | {row.get('sampler')} | {row.get('status')} | "
            f"{row.get('train_before', {})} -> {row.get('train_after', {})}"
        )



def _make_2d_sampler(name: str, y: np.ndarray, seed: int = SEED):
    """Choose a feasible neighbour count for the 2D performance-space plot."""
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError(
            "At least two measured observations are required in every "
            "class for synthetic interpolation."
        )
    neighbours = min(2, int(counts.min()) - 1)
    if name == "SMOTE":
        return SMOTE(k_neighbors=neighbours, random_state=seed)
    if name == "ADASYN":
        return ADASYN(n_neighbors=neighbours, random_state=seed)
    raise ValueError(name)


def _plot_cluster_mesh(ax, points: np.ndarray, labels: np.ndarray) -> None:
    """Draw within-class triangulation edges as a visual guide, when possible."""
    for class_id in np.unique(labels):
        subset = points[labels == class_id]
        if len(subset) < 3 or np.unique(subset, axis=0).shape[0] < 3:
            continue
        try:
            triangulation = mtri.Triangulation(subset[:, 0], subset[:, 1])
            ax.triplot(
                triangulation, color=plt.get_cmap("tab10")(int(class_id) % 10),
                linewidth=0.35, alpha=0.45, zorder=1,
            )
        except (RuntimeError, ValueError):
            # Degenerate/collinear subsets are still shown as points.
            continue


def generate_kmeans_oversampling_figures(frame: pd.DataFrame) -> None:
    """Create the 8-panel K=3..6 ADASYN/SMOTE performance-space figure.

    The two plotted dimensions are the measured HMF conversion and FDCA yield.
    K-means labels are loaded from Stage 02. Imputed outcome values are excluded
    from this plot so the axes contain only experimentally reported outcomes.

    For this figure only, SMOTE/ADASYN interpolate the two-dimensional
    performance coordinates within the existing K-means classes. This is an
    exploratory visualization of how synthetic balancing changes the plotted
    performance-space distribution; it is not an independent predictive model.
    Stage 04 continues to perform its own feature-space sampling inside each
    training fold, leaving validation publications untouched.
    """
    output_dir = Path(__file__).resolve().parents[1] / "figures" / "oversampling_kmeans"
    output_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    required = {"Conversion_pct", "Yield_pct"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(
            "Cannot create K-means oversampling plots; missing measured "
            f"outcome columns: {missing}"
        )

    conversion = pd.to_numeric(frame["Conversion_pct"], errors="coerce")
    yield_values = pd.to_numeric(frame["Yield_pct"], errors="coerce")
    measured_mask = conversion.notna() & yield_values.notna()
    all_rows_for_export = []

    fig, axes = plt.subplots(4, 2, figsize=(12, 17), sharex=True, sharey=True)
    methods = ("ADASYN", "SMOTE")

    for row_no, k in enumerate(K_VALUES):
        labels_frame = load_cluster_labels(k)
        source_rows = labels_frame["source_row"].astype(int).to_numpy()
        cluster_labels = labels_frame["cluster"].astype(int).to_numpy()

        # Align the stage-02 cluster labels with the source-row indices.
        valid_positions = measured_mask.iloc[source_rows].to_numpy()
        kept_rows = source_rows[valid_positions]
        y = cluster_labels[valid_positions]
        if len(kept_rows) == 0:
            for ax in axes[row_no]:
                ax.text(0.5, 0.5, "No measured conversion/yield pairs",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"K={k}")
            continue

        raw_xy = np.column_stack([
            conversion.iloc[kept_rows].to_numpy(dtype=float),
            yield_values.iloc[kept_rows].to_numpy(dtype=float),
        ])
        # Scale dimensions for neighbour calculations, but inverse-transform
        # synthetic coordinates before plotting in percentage units.
        scaler = StandardScaler()
        scaled_xy = scaler.fit_transform(raw_xy)

        for col_no, method in enumerate(methods):
            ax = axes[row_no, col_no]
            try:
                sampler = _make_2d_sampler(method, y, SEED)
                scaled_resampled, y_resampled = sampler.fit_resample(scaled_xy, y)
                synthetic_scaled = scaled_resampled[len(y):]
                synthetic_xy = scaler.inverse_transform(synthetic_scaled)
                y_synthetic = np.asarray(y_resampled[len(y):], dtype=int)

                combined_xy = np.vstack([raw_xy, synthetic_xy])
                combined_y = np.concatenate([y, y_synthetic])
                _plot_cluster_mesh(ax, combined_xy, combined_y)

                # Original measured records
                for class_id in np.unique(y):
                    mask = y == class_id
                    ax.scatter(
                        raw_xy[mask, 0], raw_xy[mask, 1],
                        s=20, color=plt.get_cmap("tab10")(int(class_id) % 10),
                        edgecolors="black", linewidths=0.25,
                        label=f"Class {int(class_id)}" if row_no == 0 and col_no == 0 else None,
                        zorder=3,
                    )

                # Synthetic records are marked separately using crosses.
                for class_id in np.unique(y_synthetic):
                    mask = y_synthetic == class_id
                    ax.scatter(
                        synthetic_xy[mask, 0], synthetic_xy[mask, 1],
                        s=28, marker="x",
                        color=plt.get_cmap("tab10")(int(class_id) % 10),
                        linewidths=1.0, zorder=4,
                    )

                for point, class_id in zip(synthetic_xy, y_synthetic):
                    all_rows_for_export.append({
                        "k": k, "sampler": method, "point_type": "synthetic",
                        "cluster": int(class_id),
                        "conversion_pct": float(point[0]),
                        "fdca_yield_pct": float(point[1]),
                    })

                ax.text(
                    0.02, 0.98,
                    f"Measured: {len(raw_xy)} | Synthetic: {len(synthetic_xy)}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=7,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )
            except Exception as exc:
                # Keep the panel and report why this sampler cannot be applied.
                for class_id in np.unique(y):
                    mask = y == class_id
                    ax.scatter(
                        raw_xy[mask, 0], raw_xy[mask, 1], s=20,
                        color=plt.get_cmap("tab10")(int(class_id) % 10),
                        edgecolors="black", linewidths=0.25,
                    )
                ax.text(
                    0.5, 0.5, f"{method} infeasible\n{str(exc)[:110]}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=7, wrap=True,
                    bbox={"facecolor": "white", "alpha": 0.85},
                )

            ax.set_title(f"({'abcdefgh'[row_no * 2 + col_no]}) K={k} · {method}")
            ax.set_xlim(-5, 105)
            ax.set_ylim(-5, 105)
            ax.set_xlabel("HMF conversion (%)")
            ax.set_ylabel("FDCA yield (%)")
            ax.grid(alpha=0.15, linewidth=0.4)

    fig.suptitle(
        "K-means performance-space distribution after oversampling\n"
        "Circles: measured records | Crosses: synthetic records",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"KMeans_After_Oversampling_K3_to_K6.{extension}",
            dpi=300, bbox_inches="tight",
        )
    plt.close(fig)

    synthetic_df = pd.DataFrame(all_rows_for_export)
    synthetic_df.to_csv(
        RESULTS_DIR / "kmeans_oversampling_synthetic_points.csv", index=False
    )
    metadata = {
        "figure": "KMeans_After_Oversampling_K3_to_K6",
        "k_values": list(K_VALUES),
        "samplers": list(methods),
        "measured_records_only": True,
        "n_records_with_measured_conversion_and_yield": int(measured_mask.sum()),
        "synthetic_points_generated": int(len(synthetic_df)),
        "coordinate_space": ["Conversion_pct", "Yield_pct"],
        "scaling": (
            "StandardScaler used only for neighbour-distance calculations; "
            "synthetic coordinates inverse-transformed for plotting."
        ),
        "interpretation": (
            "Exploratory visualization. K-means classes are defined from the "
            "same conversion/yield performance space; this is not an independent "
            "prediction or validation result. Stage 04 remains fold-local."
        ),
    }
    write_json(RESULTS_DIR / "kmeans_oversampling_figure_audit.json", metadata)
    print(
        "Generated K-means after-oversampling figure for K=3,4,5,6: "
        f"{len(synthetic_df)} synthetic points saved."
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
    generate_kmeans_oversampling_figures(frame)
    stage_status("03_oversampling", "completed", {
        "k_values": list(K_VALUES), "samplers": list(SAMPLERS),
        "feasible": sum(r["status"] == "ok" for r in rows),
        "infeasible": sum(r["status"] != "ok" for r in rows),
    })


if __name__ == "__main__":
    main()
