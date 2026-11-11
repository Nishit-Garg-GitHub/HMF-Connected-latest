
"""
STAGE 02 — K-MEANS CLUSTERING USING ALL 154 RECORDS

Exploratory sensitivity experiment:
- Uses Conversion_pct and Yield_pct.
- Median-imputes missing values so all 154 records can be clustered.
- Marks every record with an imputed outcome.
- Preserves the existing downstream CSV contract:
      source_row
      cluster

Stages 03, 04 and 05 can consume these files without modification.

IMPORTANT:
Clusters containing imputed outcomes are exploratory pseudo-labels,
not experimentally observed performance categories.
"""

from pathlib import Path
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from common import (
    PREPROCESSED_FILE,
    RESULTS_DIR,
    FIGURES_DIR,
    stage_status,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
K_VALUES = [3, 4, 5, 6]

OUTPUT_DIR = FIGURES_DIR / "KMeans_All154"
BACKUP_DIR = RESULTS_DIR / "observed_only_backup"

REQUIRED_COLUMNS = [
    "Conversion_pct",
    "Yield_pct",
]


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    if not PREPROCESSED_FILE.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found: {PREPROCESSED_FILE}"
        )

    df = pd.read_excel(PREPROCESSED_FILE)

    if len(df) != 154:
        warnings.warn(
            f"Expected 154 records, found {len(df)}."
        )

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing required columns: {missing}"
        )

    X = df[REQUIRED_COLUMNS].copy()

    for col in REQUIRED_COLUMNS:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce",
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return df, X


# ============================================================
# 3. BACK UP EXISTING 102-RECORD LABEL FILES
# ============================================================

def backup_existing_labels():

    BACKUP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for k in K_VALUES:

        source = RESULTS_DIR / f"kmeans_labels_K{k}.csv"

        if source.exists():

            destination = (
                BACKUP_DIR / f"kmeans_labels_K{k}_original.csv"
            )

            if not destination.exists():
                shutil.copy2(
                    source,
                    destination,
                )

                print(
                    f"Backed up previous labels: {destination}"
                )


# ============================================================
# 4. IMPUTE MISSING OUTCOMES
# ============================================================

def prepare_all_records(X):

    missing_mask = X.isna()

    print("\nMISSING OUTCOME AUDIT")

    for col in REQUIRED_COLUMNS:

        print(
            f"{col}: {int(missing_mask[col].sum())} missing"
        )

    # Median imputation is used only for this exploratory
    # all-record clustering experiment.
    imputer = SimpleImputer(
        strategy="median",
    )

    X_imputed = imputer.fit_transform(X)

    X_imputed = pd.DataFrame(
        X_imputed,
        columns=REQUIRED_COLUMNS,
        index=X.index,
    )

    if X_imputed.isna().any().any():
        raise ValueError(
            "Outcome imputation left missing values."
        )

    # Retain which records had at least one imputed outcome.
    imputed_rows = missing_mask.any(axis=1).to_numpy()

    return (
        X_imputed,
        missing_mask,
        imputed_rows,
        imputer.statistics_,
    )


# ============================================================
# 5. PLOT CLUSTERS
# ============================================================

def plot_clusters(
    k,
    X_original,
    X_imputed,
    labels,
    centers,
    imputed_rows,
):

    fig, ax = plt.subplots(
        figsize=(9, 7),
    )

    cmap = plt.get_cmap("tab10")

    for cluster in range(k):

        cluster_mask = labels == cluster

        observed_mask = (
            cluster_mask & ~imputed_rows
        )

        imputed_mask = (
            cluster_mask & imputed_rows
        )

        color = cmap(cluster % 10)

        # Records with both measured outcomes.
        if observed_mask.any():

            ax.scatter(
                X_original.loc[
                    observed_mask, "Conversion_pct"
                ],
                X_original.loc[
                    observed_mask, "Yield_pct"
                ],
                color=color,
                s=35,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.4,
                label=f"Cluster {cluster} - observed",
            )

        # Records with at least one imputed outcome.
        if imputed_mask.any():

            ax.scatter(
                X_imputed.loc[
                    imputed_mask, "Conversion_pct"
                ],
                X_imputed.loc[
                    imputed_mask, "Yield_pct"
                ],
                color=color,
                marker="x",
                s=65,
                linewidths=1.8,
                label=f"Cluster {cluster} - imputed",
            )

    # Centroids are intentionally not drawn: the figure shows cluster
    # membership only. Centre coordinates remain available from the model.

    ax.set_xlabel("Conversion (%)")
    ax.set_ylabel("FDCA Yield (%)")

    ax.set_title(
        f"K-Means clustering using all records (K={k})"
    )

    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)

    ax.grid(
        alpha=0.2,
        linestyle="--",
    )

    ax.legend(
        fontsize=7,
        loc="best",
    )

    fig.tight_layout()

    output = (
        OUTPUT_DIR / f"HMF_KMeans_All154_K{k}.png"
    )

    fig.savefig(
        output,
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Figure saved: {output}")


# ============================================================
# 6. RUN K-MEANS FOR EACH K
# ============================================================

def run_kmeans(
    df,
    X_original,
    X_imputed,
    missing_mask,
    imputed_rows,
    k,
):

    print("\n" + "=" * 70)
    print(f"RUNNING K-MEANS WITH K={k}")
    print("=" * 70)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_imputed
    )

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=100,
    )

    labels = model.fit_predict(
        X_scaled
    )

    centers = scaler.inverse_transform(
        model.cluster_centers_
    )

    # --------------------------------------------------------
    # SAVE DOWNSTREAM-COMPATIBLE LABELS
    # --------------------------------------------------------

    label_frame = pd.DataFrame({
        "source_row": np.arange(len(df), dtype=int),
        "cluster": labels.astype(int),

        # Audit columns; downstream stages ignore these.
        "conversion_imputed": (
            missing_mask["Conversion_pct"].to_numpy()
        ),

        "yield_imputed": (
            missing_mask["Yield_pct"].to_numpy()
        ),

        "any_outcome_imputed": imputed_rows,

        "conversion_used": (
            X_imputed["Conversion_pct"].to_numpy()
        ),

        "yield_used": (
            X_imputed["Yield_pct"].to_numpy()
        ),
    })

    label_path = (
        RESULTS_DIR / f"kmeans_labels_K{k}.csv"
    )

    label_frame.to_csv(
        label_path,
        index=False,
    )

    # Save cluster-assigned full dataset.
    result = df.copy()

    result["Cluster"] = labels

    result["Any_Outcome_Imputed_For_Clustering"] = (
        imputed_rows
    )

    result["Conversion_Used_For_Clustering"] = (
        X_imputed["Conversion_pct"].to_numpy()
    )

    result["Yield_Used_For_Clustering"] = (
        X_imputed["Yield_pct"].to_numpy()
    )

    result_path = (
        RESULTS_DIR / f"kmeans_all154_results_K{k}.csv"
    )

    result.to_csv(
        result_path,
        index=False,
    )

    # --------------------------------------------------------
    # PRINT CLUSTER AUDIT
    # --------------------------------------------------------

    print("\nCluster distribution:")

    counts = pd.Series(labels).value_counts().sort_index()

    for cluster, count in counts.items():

        selected = labels == cluster

        imputed_count = int(
            imputed_rows[selected].sum()
        )

        observed_count = int(
            count - imputed_count
        )

        print(
            f"Cluster {cluster}: {count} records | "
            f"{observed_count} fully observed | "
            f"{imputed_count} with imputed outcomes"
        )

    print(
        f"\nSaved labels: {label_path}"
    )

    print(
        f"Saved full result: {result_path}"
    )

    plot_clusters(
        k,
        X_original,
        X_imputed,
        labels,
        centers,
        imputed_rows,
    )

    return label_frame


# ============================================================
# 6b. PAPER-STYLE FIGURE 6 (CONVEX HULLS + SPOKES, 2x2 PANELS)
# ============================================================

FIGURE6_COLORS = ["#f0545c", "#5fd35f", "#3fa0f0", "#e76fe0", "#f5a05a", "#4f7fc4"]


def _draw_cluster_patch(ax, points, color):
    """Filled convex hull plus spokes from the cluster mean to each point."""
    from scipy.spatial import ConvexHull, QhullError
    from matplotlib.patches import Polygon

    centre = points.mean(axis=0)
    for x, y in points:
        ax.plot([centre[0], x], [centre[1], y], color=color,
                linewidth=0.5, alpha=0.55, zorder=2)
    unique = np.unique(points, axis=0)
    if len(unique) >= 3:
        try:
            hull = ConvexHull(unique)
            ax.add_patch(Polygon(unique[hull.vertices], closed=True,
                                 facecolor=color, edgecolor=color,
                                 alpha=0.22, linewidth=0.8, zorder=1))
        except QhullError:  # collinear points: spokes are enough
            pass
    ax.scatter(points[:, 0], points[:, 1], s=7, color=color,
               edgecolors="none", zorder=3)


def plot_figure6_panels(X_imputed, labels_by_k):
    """Figure 6 layout: K-means clusters for K=3-6 in a 2x2 grid.

    Points are the (Conversion, Yield) values used for clustering, so
    records with a median-imputed outcome appear at the imputed value.
    Cluster centres are not marked; spokes converge on each cluster mean.
    """
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.6))
    values = X_imputed[["Conversion_pct", "Yield_pct"]].to_numpy(dtype=float)
    for ax, k, letter in zip(axes.ravel(), K_VALUES, "abcd"):
        labels = labels_by_k[k]
        handles = []
        for cluster in range(k):
            color = FIGURE6_COLORS[cluster % len(FIGURE6_COLORS)]
            members = values[labels == cluster]
            if len(members):
                _draw_cluster_patch(ax, members, color)
            handles.append(Line2D([0], [0], marker="o", linestyle="", markersize=5,
                                  markerfacecolor=color, markeredgecolor=color,
                                  label=str(cluster)))
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                  ncol=min(k, 4 if k <= 4 else 3), frameon=False, fontsize=9,
                  handletextpad=0.2, columnspacing=1.2)
        ax.text(0.04, 0.95, f"({letter})", transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", family="serif")
        ax.set_xlim(-8, 108)
        ax.set_ylim(-8, 112)
        ax.set_xticks(range(0, 101, 20))
        ax.set_yticks(range(0, 101, 20))
        ax.set_xlabel("Conversion(%)", fontweight="bold", family="serif")
        ax.set_ylabel("Yield(%)", fontweight="bold", family="serif")
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
    fig.suptitle("Without sampling K-means clustering results: "
                 "(a) K = 3, (b) K = 4, (c) K = 5, and (d) K = 6",
                 fontsize=11, family="serif", y=0.005, va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_dir = FIGURES_DIR / "KMeans_Figure6"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "HMF_KMeans_Figure6_Style.png", dpi=400, bbox_inches="tight")
    fig.savefig(out_dir / "HMF_KMeans_Figure6_Style.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 6-style panels saved: {out_dir / 'HMF_KMeans_Figure6_Style.png'}")


# ============================================================
# 7. MAIN
# ============================================================

def main():

    print("=" * 75)
    print("HMF ELECTROOXIDATION — ALL 154 RECORDS")
    print("EXPLORATORY MEDIAN-IMPUTED K-MEANS")
    print("=" * 75)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    backup_existing_labels()

    df, X_original = load_data()

    (
        X_imputed,
        missing_mask,
        imputed_rows,
        medians,
    ) = prepare_all_records(X_original)

    print(f"\nOriginal records: {len(df)}")

    print(
        "Records with both outcomes observed:",
        int((~imputed_rows).sum()),
    )

    print(
        "Records with at least one imputed outcome:",
        int(imputed_rows.sum()),
    )

    print(
        "\nMedian values used for imputation:"
    )

    for col, median in zip(REQUIRED_COLUMNS, medians):

        print(
            f"{col}: {median:.4f}"
        )

    all_labels = {}

    for k in K_VALUES:

        all_labels[k] = run_kmeans(
            df=df,
            X_original=X_original,
            X_imputed=X_imputed,
            missing_mask=missing_mask,
            imputed_rows=imputed_rows,
            k=k,
        )

    plot_figure6_panels(
        X_imputed,
        {k: frame["cluster"].to_numpy() for k, frame in all_labels.items()},
    )

    # --------------------------------------------------------
    # SAVE RUN MANIFEST
    # --------------------------------------------------------

    audit = {
        "experiment": "all_154_median_imputed_kmeans",
        "total_records": int(len(df)),
        "fully_observed_records": int((~imputed_rows).sum()),
        "records_with_imputed_outcomes": int(imputed_rows.sum()),
        "imputation_method": "median",
        "imputation_medians": {
            col: float(value)
            for col, value in zip(REQUIRED_COLUMNS, medians)
        },
        "k_values": K_VALUES,
        "label_files": [
            f"kmeans_labels_K{k}.csv"
            for k in K_VALUES
        ],
        "warning": (
            "Clusters for records with imputed outcomes are exploratory "
            "pseudo-labels, not measured performance classes."
        ),
    }

    import json

    audit_path = (
        RESULTS_DIR / "kmeans_all154_audit.json"
    )

    audit_path.write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    stage_status(
        "02_kmeans",
        "completed",
        audit,
    )

    print("\n" + "=" * 75)
    print("ALL-154 K-MEANS COMPLETED")
    print("=" * 75)

    print(
        "\nAll four downstream label files contain "
        f"{len(df)} rows."
    )


if __name__ == "__main__":
    main()
