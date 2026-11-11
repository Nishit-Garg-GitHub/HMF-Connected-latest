
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

    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="X",
        color="black",
        s=130,
        label="Centroids",
        zorder=10,
    )

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
