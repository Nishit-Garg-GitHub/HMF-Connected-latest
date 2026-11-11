# ============================================================
# HMF ELECTROOXIDATION
# FIGURE 6-STYLE K-MEANS CLUSTERING
#
# X-AXIS : Conversion (%)
# Y-AXIS : FDCA Yield (%)
#
# K-MEANS IS PERFORMED DIRECTLY ON:
#       Conversion
#       FDCA Yield
#
# OUTPUT:
#       ONE publication-style 2 x 2 figure
#
#       (a) K = 3
#       (b) K = 4
#       (c) K = 5
#       (d) K = 6
#
# No PCA
# No elbow plot
# No silhouette plot
# No additional figures
# ============================================================


import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from scipy.spatial import ConvexHull


# ============================================================
# 1. SETTINGS
# ============================================================

ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(ROOT, "HMF_ML_Preprocessed.xlsx")

OUTPUT_DIR = os.path.join(ROOT, "outputs", "KMeans_Figure6")

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

RANDOM_STATE = 42

K_VALUES = [3, 4, 5, 6]


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 75)
print("HMF ELECTROOXIDATION")
print("FIGURE 6-STYLE K-MEANS CLUSTERING")
print("=" * 75)

df = pd.read_excel(
    DATA_FILE
)

print(
    "\nOriginal dataset:",
    df.shape
)


# ============================================================
# 3. SELECT ONLY CONVERSION AND YIELD
# ============================================================

X = df[
    [
        "Conversion_pct",
        "Yield_pct"
    ]
].copy()


# ============================================================
# 4. REMOVE ROWS WITH MISSING VALUES
# ============================================================

X = X.dropna(
    subset=[
        "Conversion_pct",
        "Yield_pct"
    ]
)


# Keep corresponding original indices

df_plot = df.loc[
    X.index
].copy()


X_values = X[
    [
        "Conversion_pct",
        "Yield_pct"
    ]
].values


print(
    "\nExperiments used for clustering:",
    len(X_values)
)


# ============================================================
# 5. STANDARDIZE X AND Y
#
# This is important because K-Means is distance based.
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X_values
)


# ============================================================
# 6. CREATE 2 x 2 FIGURE
# ============================================================

fig, axes = plt.subplots(

    2,
    2,

    figsize=(12, 9)

)


axes = axes.flatten()


# ============================================================
# 7. RUN K-MEANS FOR K = 3,4,5,6
# ============================================================

for panel_number, k in enumerate(
    K_VALUES
):


    print(
        f"\nRunning K-Means: K = {k}"
    )


    # ========================================================
    # K-MEANS
    # ========================================================

    model = KMeans(

        n_clusters=k,

        random_state=RANDOM_STATE,

        n_init=50

    )


    labels = model.fit_predict(
        X_scaled
    )


    # ========================================================
    # GET CENTROIDS
    # ========================================================

    centers_scaled = (
        model.cluster_centers_
    )


    centers = scaler.inverse_transform(
        centers_scaled
    )


    # ========================================================
    # CREATE RESULT COPY
    # ========================================================

    result = df_plot.copy()

    result[
        "Cluster"
    ] = labels


    # ========================================================
    # CURRENT AXIS
    # ========================================================

    ax = axes[
        panel_number
    ]


    # ========================================================
    # COLOR MAP
    # ========================================================

    cmap = plt.get_cmap(
        "tab10"
    )


    # ========================================================
    # PLOT EACH CLUSTER
    # ========================================================

    for cluster in range(k):


        mask = (
            labels
            == cluster
        )


        cluster_points = X_values[
            mask
        ]


        if len(cluster_points) == 0:

            continue


        color = cmap(
            cluster
            % 10
        )


        # ====================================================
        # SCATTER POINTS
        # ====================================================

        ax.scatter(

            cluster_points[:, 0],

            cluster_points[:, 1],

            s=24,

            color=color,

            alpha=0.85,

            edgecolors="white",

            linewidths=0.25,

            zorder=3,

            label=str(cluster)

        )


        # ====================================================
        # CENTROID
        # ====================================================

        center_x = centers[
            cluster,
            0
        ]

        center_y = centers[
            cluster,
            1
        ]


        # ====================================================
        # CONNECT POINTS TO CENTROID
        # ====================================================

        for point in cluster_points:

            ax.plot(

                [
                    center_x,
                    point[0]
                ],

                [
                    center_y,
                    point[1]
                ],

                color=color,

                linewidth=0.45,

                alpha=0.20,

                zorder=1

            )


        # ====================================================
        # CONVEX HULL
        # ====================================================

        if len(cluster_points) >= 3:

            try:

                hull = ConvexHull(
                    cluster_points
                )


                hull_points = (
                    cluster_points[
                        hull.vertices
                    ]
                )


                # ------------------------------------------------
                # Filled cluster region
                # ------------------------------------------------

                ax.fill(

                    hull_points[:, 0],

                    hull_points[:, 1],

                    color=color,

                    alpha=0.10,

                    zorder=0

                )


                # ------------------------------------------------
                # Cluster boundary
                # ------------------------------------------------

                closed_hull = np.vstack(

                    [
                        hull_points,

                        hull_points[0]

                    ]

                )


                ax.plot(

                    closed_hull[:, 0],

                    closed_hull[:, 1],

                    color=color,

                    linewidth=1.0,

                    alpha=0.65,

                    zorder=2

                )


            except Exception:

                pass


        # ====================================================
        # CENTROID MARKER
        # ====================================================

        ax.scatter(

            center_x,

            center_y,

            marker="x",

            s=120,

            color=color,

            linewidths=2.5,

            zorder=10

        )


    # ========================================================
    # PANEL LABEL
    # ========================================================

    panel_labels = [
        "(a)",
        "(b)",
        "(c)",
        "(d)"
    ]


    ax.text(

        0.03,

        0.95,

        f"{panel_labels[panel_number]} K = {k}",

        transform=ax.transAxes,

        fontsize=12,

        fontweight="bold",

        va="top"

    )


    # ========================================================
    # AXIS LABELS
    # ========================================================

    ax.set_xlabel(

        "Conversion (%)",

        fontsize=11

    )


    ax.set_ylabel(

        "FDCA Yield (%)",

        fontsize=11

    )


    # ========================================================
    # AXIS LIMITS
    # ========================================================

    ax.set_xlim(
        0,
        105
    )


    ax.set_ylim(
        -5,
        105
    )


    # ========================================================
    # TICKS
    # ========================================================

    ax.set_xticks(

        [
            0,
            20,
            40,
            60,
            80,
            100
        ]

    )


    ax.set_yticks(

        [
            0,
            20,
            40,
            60,
            80,
            100
        ]

    )


    # ========================================================
    # GRID
    # ========================================================

    ax.grid(

        True,

        alpha=0.15,

        linewidth=0.6

    )


    # ========================================================
    # LEGEND
    # ========================================================

    ax.legend(

        loc="upper right",

        frameon=False,

        fontsize=8,

        title=None

    )


    # ========================================================
    # AXIS SPINES
    # ========================================================

    for spine in ax.spines.values():

        spine.set_linewidth(
            0.8
        )


# ============================================================
# 8. OVERALL TITLE
# ============================================================

fig.suptitle(

    "K-Means Clustering of HMF Electrooxidation Experiments",

    fontsize=16,

    fontweight="bold",

    y=0.995

)


# ============================================================
# 9. LAYOUT
# ============================================================

plt.subplots_adjust(

    left=0.08,

    right=0.98,

    bottom=0.08,

    top=0.92,

    wspace=0.22,

    hspace=0.28

)


# ============================================================
# 10. SAVE ONLY ONE FIGURE
# ============================================================

output_file = os.path.join(

    OUTPUT_DIR,

    "HMF_KMeans_Figure6_Style.png"

)


plt.savefig(

    output_file,

    dpi=600,

    bbox_inches="tight"

)


plt.show()

plt.close()


# ============================================================
# 11. FINISHED
# ============================================================

print("\n")
print("=" * 75)
print("FIGURE GENERATED SUCCESSFULLY")
print("=" * 75)

print(
    "\nSaved at:"
)

print(
    output_file
)