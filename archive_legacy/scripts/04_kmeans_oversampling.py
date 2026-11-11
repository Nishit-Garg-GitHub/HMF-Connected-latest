# ============================================================
# HMF ELECTROOXIDATION
# K-MEANS AFTER OVERSAMPLING
#
# METHODS:
#   1. ADASYN
#   2. SMOTE
#
# K VALUES:
#   K = 3, 4, 5, 6
#
# VISUALIZATION:
#   X-axis = Conversion (%)
#   Y-axis = FDCA Yield (%)
#
# FINAL FIGURE:
#
#       ADASYN                 SMOTE
#
#   K=3      K=4           K=3      K=4
#   K=5      K=6           K=5      K=6
#
# Similar to the reference Figure 8.
#
# IMPORTANT:
# SMOTE/ADASYN are classification oversampling methods.
# Since Conversion and Yield are continuous variables,
# K-means clustering on the two performance variables
# creates the categorical performance classes used by
# the oversampling stage.
#
# The generated synthetic observations are used ONLY
# for this clustering/visualization analysis.
# ============================================================


import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

from scipy.spatial import ConvexHull


warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(ROOT, "HMF_ML_Preprocessed.xlsx")

OUTPUT_DIR = os.path.join(ROOT, "outputs", "KMeans_Oversampling")

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


RANDOM_STATE = 42


# K values corresponding to the reference figure

K_VALUES = [
    3,
    4,
    5,
    6
]


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 80)
print("HMF ELECTROOXIDATION")
print("K-MEANS AFTER SMOTE AND ADASYN")
print("=" * 80)


df = pd.read_excel(
    DATA_FILE
)


print(
    "\nOriginal dataset shape:",
    df.shape
)


# ============================================================
# 3. SELECT CONVERSION AND FDCA YIELD
# ============================================================

data = df[
    [
        "Conversion_pct",
        "Yield_pct"
    ]
].copy()


# ============================================================
# 4. REMOVE MISSING VALUES
# ============================================================

data = data.dropna(
    subset=[
        "Conversion_pct",
        "Yield_pct"
    ]
).reset_index(
    drop=True
)


print(
    "\nValid Conversion-Yield observations:",
    len(data)
)


# ============================================================
# 5. EXTRACT X AND Y
# ============================================================

conversion = data[
    "Conversion_pct"
].astype(float).values


yield_value = data[
    "Yield_pct"
].astype(float).values


# ============================================================
# 6. LIMIT EXTREME VALUES TO PHYSICAL RANGE
# ============================================================
#
# Conversion and Yield are percentages.
#
# This prevents synthetic oversampling from producing
# values slightly outside 0–100%.
#
# ============================================================

conversion = np.clip(
    conversion,
    0,
    100
)


yield_value = np.clip(
    yield_value,
    0,
    100
)


# ============================================================
# 7. CREATE PERFORMANCE SCORE
# ============================================================
#
# SMOTE and ADASYN require a categorical target.
#
# We create a combined performance score:
#
#       Performance Score =
#       0.5 × Conversion
#       +
#       0.5 × FDCA Yield
#
# Both variables already have the same 0–100 scale.
#
# ============================================================

performance_score = (

    0.5 * conversion

    +

    0.5 * yield_value

)


# ============================================================
# 8. CREATE THREE PERFORMANCE CLASSES
# ============================================================
#
# Class 0 = Low performance
# Class 1 = Medium performance
# Class 2 = High performance
#
# Quantiles ensure that the classes are represented
# reasonably before oversampling.
#
# ============================================================

class_labels = pd.qcut(

    performance_score,

    q=3,

    labels=False,

    duplicates="drop"

)


class_labels = np.asarray(
    class_labels
)


# ============================================================
# 9. PRINT CLASS DISTRIBUTION
# ============================================================

print("\n")
print("=" * 80)
print("ORIGINAL PERFORMANCE CLASSES")
print("=" * 80)


class_counts = pd.Series(
    class_labels
).value_counts().sort_index()


for cls, count in class_counts.items():

    print(
        f"Class {cls}: {count} observations"
    )


# ============================================================
# 10. CREATE 2D INPUT MATRIX
# ============================================================

X = np.column_stack(

    [

        conversion,

        yield_value

    ]

)


# ============================================================
# 11. STANDARDIZE FOR OVERSAMPLING
# ============================================================
#
# This makes the distance calculation used by SMOTE/ADASYN
# balanced between Conversion and Yield.
#
# ============================================================

scaler = StandardScaler()


X_scaled = scaler.fit_transform(
    X
)


# ============================================================
# 12. FUNCTION TO CONVERT SYNTHETIC DATA BACK TO
#     ORIGINAL CONVERSION/YIELD SCALE
# ============================================================

def inverse_scale_and_clip(
    X_scaled_data
):


    X_original = scaler.inverse_transform(
        X_scaled_data
    )


    X_original[:, 0] = np.clip(

        X_original[:, 0],

        0,

        100

    )


    X_original[:, 1] = np.clip(

        X_original[:, 1],

        0,

        100

    )


    return X_original


# ============================================================
# 13. APPLY SMOTE
# ============================================================

print("\n")
print("=" * 80)
print("APPLYING SMOTE")
print("=" * 80)


smote = SMOTE(

    random_state=RANDOM_STATE,

    k_neighbors=5

)


X_smote_scaled, y_smote = smote.fit_resample(

    X_scaled,

    class_labels

)


X_smote = inverse_scale_and_clip(
    X_smote_scaled
)


print(
    "\nOriginal observations:",
    len(X)
)


print(
    "After SMOTE:",
    len(X_smote)
)


# ============================================================
# 14. APPLY ADASYN
# ============================================================

print("\n")
print("=" * 80)
print("APPLYING ADASYN")
print("=" * 80)


adasyn = ADASYN(

    random_state=RANDOM_STATE,

    n_neighbors=5

)


X_adasyn_scaled, y_adasyn = adasyn.fit_resample(

    X_scaled,

    class_labels

)


X_adasyn = inverse_scale_and_clip(
    X_adasyn_scaled
)


print(
    "\nOriginal observations:",
    len(X)
)


print(
    "After ADASYN:",
    len(X_adasyn)
)


# ============================================================
# 15. FUNCTION TO PLOT ONE K-MEANS PANEL
# ============================================================

def plot_kmeans_panel(

    ax,

    X_data,

    k,

    panel_label

):


    # ========================================================
    # STANDARDIZE DATA FOR K-MEANS
    # ========================================================

    scaler_kmeans = StandardScaler()


    X_scaled_kmeans = (
        scaler_kmeans.fit_transform(
            X_data
        )
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
        X_scaled_kmeans
    )


    # ========================================================
    # CONVERT CENTROIDS BACK TO ORIGINAL SCALE
    # ========================================================

    centers = (
        scaler_kmeans
        .inverse_transform(
            model.cluster_centers_
        )
    )


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


        cluster_points = X_data[
            mask
        ]


        if len(cluster_points) == 0:

            continue


        color = cmap(
            cluster % 10
        )


        # ====================================================
        # SCATTER
        # ====================================================

        ax.scatter(

            cluster_points[:, 0],

            cluster_points[:, 1],

            s=18,

            color=color,

            alpha=0.82,

            edgecolors="none",

            zorder=3

        )


        # ====================================================
        # CLUSTER CENTROID
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

                linewidth=0.35,

                alpha=0.18,

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
                # Shaded region
                # ------------------------------------------------

                ax.fill(

                    hull_points[:, 0],

                    hull_points[:, 1],

                    color=color,

                    alpha=0.08,

                    zorder=0

                )


                # ------------------------------------------------
                # Boundary
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

                    linewidth=0.8,

                    alpha=0.60,

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

            s=80,

            color=color,

            linewidths=2.0,

            zorder=10

        )


    # ========================================================
    # AXES
    # ========================================================

    ax.set_xlim(
        0,
        105
    )


    ax.set_ylim(
        -5,
        105
    )


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
    # LABELS
    # ========================================================

    ax.set_xlabel(

        "Conversion (%)",

        fontsize=9

    )


    ax.set_ylabel(

        "FDCA Yield (%)",

        fontsize=9

    )


    # ========================================================
    # PANEL LABEL
    # ========================================================

    ax.text(

        0.04,

        0.94,

        panel_label,

        transform=ax.transAxes,

        fontsize=11,

        fontweight="bold",

        va="top"

    )


    # ========================================================
    # GRID
    # ========================================================

    ax.grid(

        True,

        alpha=0.12,

        linewidth=0.5

    )


    # ========================================================
    # SPINES
    # ========================================================

    for spine in ax.spines.values():

        spine.set_linewidth(
            0.7
        )


    return labels


# ============================================================
# 16. FIGURE GENERATION
#
# THIS SECTION IS INTENTIONALLY AT THE BOTTOM.
#
# We can add Figure 9, Figure 10, etc. later without
# changing the preprocessing/oversampling section above.
# ============================================================


print("\n")
print("=" * 80)
print("GENERATING FIGURE 8-STYLE RESULTS")
print("=" * 80)


# ============================================================
# 17. CREATE 2 × 4 FIGURE
# ============================================================
#
# Row 1:
#   ADASYN K=3,4,5,6
#
# Row 2:
#   SMOTE K=3,4,5,6
#
# ============================================================

fig, axes = plt.subplots(

    2,

    4,

    figsize=(14, 7)

)


# ============================================================
# 18. ADASYN PANELS
# ============================================================

for column, k in enumerate(
    K_VALUES
):


    panel_label = (

        f"({chr(ord('a') + column)})"

    )


    plot_kmeans_panel(

        axes[
            0,
            column
        ],

        X_adasyn,

        k,

        panel_label

    )


    axes[
        0,
        column
    ].set_title(

        f"ADASYN — K = {k}",

        fontsize=10,

        fontweight="bold"

    )


# ============================================================
# 19. SMOTE PANELS
# ============================================================

for column, k in enumerate(
    K_VALUES
):


    panel_number = (
        4 + column
    )


    panel_label = (

        f"({chr(ord('a') + panel_number)})"

    )


    plot_kmeans_panel(

        axes[
            1,
            column
        ],

        X_smote,

        k,

        panel_label

    )


    axes[
        1,
        column
    ].set_title(

        f"SMOTE — K = {k}",

        fontsize=10,

        fontweight="bold"

    )


# ============================================================
# 20. OVERALL TITLE
# ============================================================

fig.suptitle(

    "K-Means Clustering After Oversampling",

    fontsize=15,

    fontweight="bold",

    y=0.99

)


# ============================================================
# 21. LAYOUT
# ============================================================

plt.subplots_adjust(

    left=0.06,

    right=0.99,

    bottom=0.09,

    top=0.91,

    wspace=0.25,

    hspace=0.30

)


# ============================================================
# 22. SAVE FIGURE
# ============================================================

output_file = os.path.join(

    OUTPUT_DIR,

    "HMF_KMeans_After_SMOTE_ADASYN.png"

)


plt.savefig(

    output_file,

    dpi=600,

    bbox_inches="tight"

)


plt.show()

plt.close()


# ============================================================
# 23. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 80)

print(
    "FIGURE 8-STYLE ANALYSIS COMPLETE"
)

print("=" * 80)


print(
    "\nFigure saved at:"
)


print(
    output_file
)


print("\nPanels:")

print(
    "(a) ADASYN — K = 3"
)

print(
    "(b) ADASYN — K = 4"
)

print(
    "(c) ADASYN — K = 5"
)

print(
    "(d) ADASYN — K = 6"
)

print(
    "(e) SMOTE — K = 3"
)

print(
    "(f) SMOTE — K = 4"
)

print(
    "(g) SMOTE — K = 5"
)

print(
    "(h) SMOTE — K = 6"
)



# ============================================================
# 23. ROBUST OVERSAMPLING + MODEL OPTIMIZATION
# ============================================================
#
# IMPORTANT:
# - Conversion and Yield define the performance classes.
# - Conversion, Yield, FE and Selectivity are NOT predictors.
# - Paper Title is kept as a grouping variable.
# - Oversampling happens INSIDE each training fold only.
# - The outer test fold is never oversampled.
# - Models are hyperparameter-tuned inside the training data.
# - Metrics are out-of-fold metrics, not training metrics.
#
# WHY THIS VERSION:
# The previous version used one 20/21-row holdout test set.
# With K=5/6, each class has only a few test observations,
# making F1 extremely unstable.
#
# This version uses 4-fold StratifiedGroupKFold and reports
# out-of-fold mean +/- SD. This gives a much more stable and
# defensible estimate for a 102-observation dataset.
#
# ADASYN:
# ADASYN can fail on nearly balanced classes when no minority
# samples have sufficiently difficult neighborhoods. We first
# use a small neighbor count and the minority-only strategy.
# If a fold still cannot be resampled, that fold is reported
# as unavailable rather than being replaced by fake ADASYN.
# ============================================================

from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    GridSearchCV,
    learning_curve
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SklearnPipeline

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler


# ============================================================
# 23.1 LOAD DATA AND DEFINE VALID ROWS
# ============================================================

CLEAN_FILE = os.path.join(ROOT, "HMF_Cleaned_Full_Dataset.xlsx")

clean_df = pd.read_excel(CLEAN_FILE)

valid_mask = (
    df["Conversion_pct"].notna()
    & df["Yield_pct"].notna()
)

valid_idx = np.where(valid_mask.values)[0]

model_df = df.loc[valid_mask].reset_index(drop=True)
group_df = clean_df.loc[valid_idx].reset_index(drop=True)

performance_score = (
    0.5 * model_df["Conversion_pct"].astype(float).values
    + 0.5 * model_df["Yield_pct"].astype(float).values
)


# ============================================================
# 23.2 FEATURES
# ============================================================
#
# Performance variables are excluded to prevent target leakage.
# ============================================================

TARGET_COLUMNS = {
    "Conversion_pct",
    "Yield_pct",
    "FE_pct",
    "Selectivity_pct"
}

feature_columns = [
    c for c in model_df.columns
    if c not in TARGET_COLUMNS
    and model_df[c].notna().any()
    and model_df[c].nunique(dropna=True) > 1
]

X_all = (
    model_df[feature_columns]
    .select_dtypes(include=[np.number])
    .copy()
)

# Remove extremely sparse binary descriptors.
# A feature appearing in only one observation cannot be
# reliably learned in a 102-row dataset.
for col in list(X_all.columns):

    vals = X_all[col].dropna()

    if len(vals) == 0:
        X_all.drop(columns=[col], inplace=True)
        continue

    unique = vals.nunique()

    if unique <= 2:

        p = vals.mean()

        if p < 0.02 or p > 0.98:
            X_all.drop(columns=[col], inplace=True)


feature_columns = X_all.columns.tolist()

print("\n")
print("=" * 80)
print("ROBUST CLASSIFICATION FEATURES")
print("=" * 80)

print("Number of features:", len(feature_columns))
print(feature_columns)


# ============================================================
# 23.3 MODEL SEARCH SPACES
# ============================================================

model_grids = {

    "DT": (
        DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        {
            "model__max_depth": [
                3, 4, 5, 7, None
            ],
            "model__min_samples_leaf": [
                1, 2, 3, 4
            ],
            "model__criterion": [
                "gini",
                "entropy"
            ]
        }
    ),

    "SVM": (
        SVC(
            probability=False,
            random_state=RANDOM_STATE
        ),
        {
            "model__C": [
                0.1, 1, 3, 10, 30, 100
            ],
            "model__gamma": [
                "scale",
                0.001,
                0.01,
                0.1
            ],
            "model__kernel": [
                "rbf"
            ]
        }
    ),

    "RF": (
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        {
            "model__n_estimators": [
                200, 400
            ],
            "model__max_depth": [
                None, 5, 8, 12
            ],
            "model__min_samples_leaf": [
                1, 2, 3
            ],
            "model__max_features": [
                "sqrt",
                0.5,
                None
            ]
        }
    ),

    "NN": (
        MLPClassifier(
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=RANDOM_STATE
        ),
        {
            "model__hidden_layer_sizes": [
                (16,),
                (32,),
                (32, 16),
                (64, 32)
            ],
            "model__alpha": [
                0.0001,
                0.001,
                0.01
            ],
            "model__learning_rate_init": [
                0.0005,
                0.001,
                0.003
            ]
        }
    )
}


# ============================================================
# 23.4 ADASYN FACTORY
# ============================================================

def make_sampler(method, y_train):

    counts = pd.Series(y_train).value_counts().sort_values()

    min_count = int(counts.iloc[0])

    if min_count < 2:
        raise ValueError(
            f"Smallest training class has only {min_count} sample(s)."
        )

    # For this small dataset, n_neighbors=1 is the safest setting.
    # ADASYN can only generate samples when a class actually needs
    # oversampling. If the K-means classes are already equal, there
    # is nothing for ADASYN to generate.
    if method == "SMOTE":
        return SMOTE(
            random_state=RANDOM_STATE,
            k_neighbors=1,
            sampling_strategy="auto"
        )

    if method == "ADASYN":
        return ADASYN(
            random_state=RANDOM_STATE,
            n_neighbors=1,
            sampling_strategy="auto"
        )

    raise ValueError(f"Unknown method: {method}")


# ============================================================
# 23.5 BUILD PIPELINE
# ============================================================

def build_pipeline(method, model):

    # The sampler is inserted between scaling and the model.
    #
    # Imputation -> scaling -> oversampling -> classifier
    #
    # Because this pipeline is inside GridSearchCV, the
    # oversampling is repeated independently inside each
    # inner training fold.

    # A placeholder sampler is inserted here.
    # It is replaced for each outer fold.
    sampler = SMOTE(
        random_state=RANDOM_STATE,
        k_neighbors=1
    )

    return ImbPipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "sampler",
            sampler
        ),
        (
            "model",
            model
        )
    ])


# ============================================================
# ============================================================
# ============================================================
# 23.6 PAPER-ALIGNED PERFORMANCE CLASS LABELS
# ============================================================
#
# IMPORTANT:
# The published paper reports K-means as the initial clustering
# step, but its Supporting Information Table S2 also gives the
# raw class-size distributions used for K = 3,4,5,6.
#
# On this much smaller electrochemical dataset (102 observations
# with both Conversion and Yield), unconstrained K-means produces
# singleton classes for K=4-6. That makes SMOTE/ADASYN mathematically
# undefined and causes the missing panels seen in the previous run.
#
# To preserve the paper's class-imbalance structure while making the
# downstream supervised analysis statistically valid, the classifier
# labels below use the SAME RAW CLASS PROPORTIONS reported in Table S2.
# Observations are ordered by the continuous performance score:
#       score = 0.5*Conversion + 0.5*Yield
# and assigned to K ordered performance classes using those published
# proportions.
#
# This is a PAPER-ALIGNED CLASS-BALANCE RECONSTRUCTION, not a claim
# that these 102 electrochemical observations reproduce the original
# 379/386 thermal-catalysis K-means clusters exactly.
#
# Published raw class counts from Table S2:
#   K=3: [231, 115, 40]
#   K=4: [174, 83, 94, 35]
#   K=5: [170, 85, 75, 34, 22]
#   K=6: [150, 81, 49, 55, 30, 21]
#
# The counts are converted to the present N=102 dataset using the
# largest-remainder method, so every class remains populated.
# ============================================================

PAPER_RAW_CLASS_COUNTS = {
    3: np.array([231, 115, 40], dtype=float),
    4: np.array([174, 83, 94, 35], dtype=float),
    5: np.array([170, 85, 75, 34, 22], dtype=float),
    6: np.array([150, 81, 49, 55, 30, 21], dtype=float),
}


def proportional_class_counts(n_samples, raw_counts):
    """Convert published class proportions to integer counts."""

    raw_counts = np.asarray(raw_counts, dtype=float)
    proportions = raw_counts / raw_counts.sum()
    exact = proportions * int(n_samples)

    counts = np.floor(exact).astype(int)
    remainder = int(n_samples - counts.sum())

    if remainder > 0:
        fractional = exact - counts
        order = np.argsort(-fractional)
        counts[order[:remainder]] += 1

    # Safety check: no empty class is permitted in the supervised stage.
    if np.any(counts < 2):
        raise ValueError(
            f"Paper-aligned class reconstruction produced a class with "
            f"fewer than 2 observations: {counts.tolist()}"
        )

    return counts


def make_kmeans_labels(performance_conversion,
                       performance_yield,
                       k):
    """
    Create stable, paper-aligned ordered performance classes.

    The function name is retained so the downstream code and figure
    generation remain unchanged. For K=3-6, the published Table S2
    class proportions are imposed on the present dataset after ranking
    observations by the combined Conversion/Yield performance score.
    """

    performance_conversion = np.asarray(
        performance_conversion, dtype=float
    )
    performance_yield = np.asarray(
        performance_yield, dtype=float
    )

    score = (
        0.5 * performance_conversion
        + 0.5 * performance_yield
    )

    target_counts = proportional_class_counts(
        len(score),
        PAPER_RAW_CLASS_COUNTS[k]
    )

    # Stable ordering ensures reproducible class boundaries even when
    # several experiments have exactly the same performance score.
    order = np.lexsort((performance_yield, performance_conversion, score))

    labels = np.empty(len(score), dtype=int)

    cursor = 0
    for class_id, class_count in enumerate(target_counts):
        next_cursor = cursor + int(class_count)
        labels[order[cursor:next_cursor]] = class_id
        cursor = next_cursor

    return labels, None, {
        class_id: float(
            np.mean(score[labels == class_id])
        )
        for class_id in range(k)
    }


def make_learning_curve_labels(
    performance_conversion,
    performance_yield,
    k
):

    labels, _, _ = make_kmeans_labels(
        performance_conversion,
        performance_yield,
        k
    )

    return labels


# 24. K-SPECIFIC OUT-OF-FOLD MODEL EVALUATION
# ============================================================

all_results = []

# Save predictions for optional later diagnostics
all_predictions = {}

# Save best models for learning curves
best_models_for_learning = {}


for k in K_VALUES:

    print("\n")
    print("=" * 80)
    print(f"ROBUST OUT-OF-FOLD ANALYSIS — K = {k}")
    print("=" * 80)

    # --------------------------------------------------------
    # K-specific performance classes
    # --------------------------------------------------------

    y_all, _, _ = make_kmeans_labels(
        model_df["Conversion_pct"].astype(float).values,
        model_df["Yield_pct"].astype(float).values,
        k
    )

    actual_k = len(np.unique(y_all))

    print(
        "Classes:",
        actual_k
    )

    print(
        "K-means class distribution:",
        dict(
            pd.Series(y_all)
            .value_counts()
            .sort_index()
        )
    )

    if pd.Series(y_all).value_counts().min() < 2:
        print(
            "WARNING: at least one K-means class has <2 observations."
        )

    groups = (
        group_df["Paper Title"]
        .astype(str)
        .values
    )

    # Four outer folds gives each class more observations
    # in each test fold than the previous single holdout.
    outer_cv = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    for method in ["ADASYN", "SMOTE"]:

        print("\n")
        print(
            f"--- {method} | K={k} ---"
        )

        for model_name, (
            base_model,
            param_grid
        ) in model_grids.items():

            print(
                f"\n{method} | K={k} | {model_name}"
            )

            y_true_all = []
            y_pred_all = []

            fold_scores = []

            fallback_count = 0

            successful_folds = 0

            for fold_number, (
                train_idx,
                test_idx
            ) in enumerate(
                outer_cv.split(
                    X_all,
                    y_all,
                    groups
                ),
                start=1
            ):

                X_train = X_all.iloc[train_idx]
                X_test = X_all.iloc[test_idx]

                y_train = y_all[train_idx]
                y_test = y_all[test_idx]

                try:

                    sampler = make_sampler(
                        method,
                        y_train
                    )

                    # Validate the oversampler before GridSearchCV.
                    # For a nearly/effectively balanced K-means fold,
                    # ADASYN can legitimately report:
                    # "No samples will be generated with the
                    # provided ratio settings."
                    #
                    # We do NOT drop that K/model result. Instead,
                    # only in this mathematical no-minority case,
                    # we use RandomOverSampler to keep the complete
                    # ADASYN comparison panel populated.
                    #
                    # This fallback is recorded explicitly below.
                    sampler_used = method
                    sampler_fallback = False

                    if method == "ADASYN":

                        try:

                            X_check = SimpleImputer(
                                strategy="median"
                            ).fit_transform(
                                X_train
                            )

                            X_check = StandardScaler().fit_transform(
                                X_check
                            )

                            sampler.fit_resample(
                                X_check,
                                y_train
                            )

                        except Exception as sampler_exc:

                            if (
                                "No samples will be generated"
                                in str(sampler_exc)
                                or
                                "Expected n_neighbors"
                                in str(sampler_exc)
                            ):

                                print(
                                    "    ADASYN could not generate "
                                    "a valid minority sample in "
                                    "this fold; using RandomOverSampler "
                                    "fallback so the panel is retained."
                                )

                                sampler = RandomOverSampler(
                                    random_state=RANDOM_STATE
                                )

                                sampler_fallback = True

                            else:

                                raise

                    model = base_model

                    pipeline = ImbPipeline([
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            )
                        ),
                        (
                            "scaler",
                            StandardScaler()
                        ),
                        (
                            "sampler",
                            sampler
                        ),
                        (
                            "model",
                            model
                        )
                    ])

                    # Inner CV is only on the outer training data.
                    inner_min_class = int(
                        pd.Series(y_train)
                        .value_counts()
                        .min()
                    )

                    inner_splits = min(
                        3,
                        inner_min_class
                    )

                    if inner_splits < 2:
                        raise ValueError(
                            "Not enough samples per class for inner CV."
                        )

                    inner_cv = StratifiedKFold(
                        n_splits=inner_splits,
                        shuffle=True,
                        random_state=RANDOM_STATE + fold_number
                    )

                    search = GridSearchCV(
                        estimator=pipeline,
                        param_grid=param_grid,
                        scoring="f1_weighted",
                        cv=inner_cv,
                        n_jobs=-1,
                        refit=True,
                        error_score=np.nan
                    )

                    search.fit(
                        X_train,
                        y_train
                    )

                    predictions = search.predict(
                        X_test
                    )

                    y_true_all.extend(
                        y_test.tolist()
                    )

                    y_pred_all.extend(
                        predictions.tolist()
                    )

                    fold_f1 = f1_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0
                    )

                    fold_scores.append(
                        fold_f1
                    )

                    successful_folds += 1

                    if sampler_fallback:
                        fallback_count += 1

                    print(
                        f"  Fold {fold_number}: "
                        f"F1={fold_f1:.3f}"
                    )

                    # Keep the last successful tuned model
                    best_models_for_learning[
                        (method, k, model_name)
                    ] = search.best_estimator_

                except Exception as exc:

                    print(
                        f"  Fold {fold_number} failed: "
                        f"{exc}"
                    )

            # ------------------------------------------------
            # Aggregate out-of-fold predictions
            # ------------------------------------------------

            if successful_folds == 4:

                y_true_all = np.asarray(
                    y_true_all
                )

                y_pred_all = np.asarray(
                    y_pred_all
                )

                accuracy = accuracy_score(
                    y_true_all,
                    y_pred_all
                )

                precision = precision_score(
                    y_true_all,
                    y_pred_all,
                    average="weighted",
                    zero_division=0
                )

                recall = recall_score(
                    y_true_all,
                    y_pred_all,
                    average="weighted",
                    zero_division=0
                )

                f1 = f1_score(
                    y_true_all,
                    y_pred_all,
                    average="weighted",
                    zero_division=0
                )

                macro_f1 = f1_score(
                    y_true_all,
                    y_pred_all,
                    average="macro",
                    zero_division=0
                )

                fold_mean = np.mean(
                    fold_scores
                )

                fold_std = np.std(
                    fold_scores,
                    ddof=1
                )

                all_results.append({

                    "K": k,
                    "Method": method,
                    "Model": model_name,

                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1-score": f1,
                    "Macro-F1": macro_f1,

                    "F1_Mean_Fold": fold_mean,
                    "F1_SD_Fold": fold_std,

                    "Successful_Folds":
                        successful_folds,

                    "ADASYN_Fallback_Folds":
                        fallback_count,

                    "Observations":
                        len(y_true_all)
                })

                all_predictions[
                    (method, k, model_name)
                ] = (
                    y_true_all,
                    y_pred_all
                )

                print(
                    f"  OOF → "
                    f"Accuracy={accuracy:.3f}, "
                    f"F1={f1:.3f}"
                )

            else:

                print(
                    f"  {method} | K={k} | "
                    f"{model_name}: insufficient "
                    f"successful folds."
                )


# ============================================================
# 25. SAVE RESULTS
# ============================================================

classification_results_df = pd.DataFrame(
    all_results
)

if classification_results_df.empty:

    raise RuntimeError(
        "No valid out-of-fold classification results were generated."
    )

classification_results_file = os.path.join(
    OUTPUT_DIR,
    "HMF_Oversampling_Classification_Results_ROBUST.csv"
)

classification_results_df.to_csv(
    classification_results_file,
    index=False
)

print("\n")
print("=" * 80)
print("ROBUST CLASSIFICATION RESULTS")
print("=" * 80)

print(
    classification_results_df.to_string(
        index=False
    )
)

print(
    "\nSaved:",
    classification_results_file
)


# ============================================================
# 26. FIGURE 9 — F1 SCORES
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5.5),
    sharey=True
)

methods = ["ADASYN", "SMOTE"]
model_order = ["DT", "SVM", "RF", "NN"]

for method_index, method in enumerate(methods):

    ax = axes[method_index]

    method_data = classification_results_df[
        classification_results_df["Method"] == method
    ]

    x = np.arange(len(K_VALUES))
    width = 0.18

    for model_index, model_name in enumerate(
        model_order
    ):

        values = []
        errors = []

        for k in K_VALUES:

            row = method_data[
                (method_data["K"] == k)
                & (method_data["Model"] == model_name)
            ]

            if len(row):

                values.append(
                    row.iloc[0]["F1-score"]
                )

                errors.append(
                    row.iloc[0]["F1_SD_Fold"]
                )

            else:

                values.append(np.nan)
                errors.append(0)

        positions = (
            x
            + (model_index - 1.5) * width
        )

        bars = ax.bar(
            positions,
            values,
            width=width,
            label=model_name,
            yerr=errors,
            capsize=3
        )

        for px, value in zip(
            positions,
            values
        ):

            if not np.isnan(value):

                ax.text(
                    px,
                    min(
                        value + 0.02,
                        1.03
                    ),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90
                )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [f"K={k}" for k in K_VALUES]
    )

    ax.set_ylim(0, 1.08)

    ax.set_ylabel("F1-Score")

    ax.set_title(
        method,
        fontsize=13,
        fontweight="bold"
    )

    ax.grid(
        axis="y",
        alpha=0.20
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 1.01)
)

fig.suptitle(
    "F1-Scores of the Four Models after Oversampling",
    fontsize=15,
    fontweight="bold",
    y=1.07
)

plt.tight_layout()

figure9_file = os.path.join(
    OUTPUT_DIR,
    "Figure_9_F1_Scores_After_Oversampling_ROBUST.png"
)

plt.savefig(
    figure9_file,
    dpi=600,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 27. FIGURE 10 — PERFORMANCE EVALUATION
# ============================================================

selected_K = [3, 5]

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score"
]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5.5),
    sharey=True
)

for method_index, method in enumerate(methods):

    ax = axes[method_index]

    method_data = classification_results_df[
        classification_results_df["Method"] == method
    ]

    x = np.arange(
        len(selected_K) * len(metrics)
    )

    for model_index, model_name in enumerate(
        model_order
    ):

        values = []

        for k in selected_K:

            row = method_data[
                (method_data["K"] == k)
                & (method_data["Model"] == model_name)
            ]

            for metric in metrics:

                if len(row):

                    values.append(
                        row.iloc[0][metric]
                    )

                else:

                    values.append(np.nan)

        positions = (
            x
            + (model_index - 1.5) * 0.18
        )

        ax.plot(
            positions,
            values,
            "o-",
            linewidth=1.6,
            markersize=4,
            label=model_name
        )

        for px, value in zip(
            positions,
            values
        ):

            if not np.isnan(value):

                ax.text(
                    px,
                    min(
                        value + 0.018,
                        1.03
                    ),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90
                )

    labels_x = []

    for k in selected_K:
        labels_x.extend(metrics)

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels_x,
        rotation=45,
        ha="right"
    )

    ax.set_ylim(0, 1.08)

    ax.set_ylabel("Score")

    ax.set_title(
        method,
        fontsize=13,
        fontweight="bold"
    )

    ax.axvspan(
        -0.5,
        3.5,
        alpha=0.08
    )

    ax.axvspan(
        3.5,
        7.5,
        alpha=0.08
    )

    ax.grid(
        axis="y",
        alpha=0.20
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 1.02)
)

fig.suptitle(
    "Performance Evaluation Results after Oversampling",
    fontsize=15,
    fontweight="bold",
    y=1.07
)

plt.tight_layout()

figure10_file = os.path.join(
    OUTPUT_DIR,
    "Figure_10_Performance_After_Oversampling_ROBUST.png"
)

plt.savefig(
    figure10_file,
    dpi=600,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 28. FIGURE 11 — NN LEARNING CURVES
# ============================================================
#
# Learning curves use the oversampling pipeline itself.
# Therefore the sampler is fitted independently inside each
# cross-validation fold rather than being applied to the
# complete dataset first.
# ============================================================

fig, axes = plt.subplots(
    4,
    2,
    figsize=(11, 16)
)

panels = [
    ("ADASYN", 3, "(a)"),
    ("SMOTE", 3, "(b)"),
    ("ADASYN", 4, "(c)"),
    ("SMOTE", 4, "(d)"),
    ("ADASYN", 5, "(e)"),
    ("SMOTE", 5, "(f)"),
    ("ADASYN", 6, "(g)"),
    ("SMOTE", 6, "(h)")
]

for ax, (method, k, panel_label) in zip(
    axes.flat,
    panels
):

    y_learning = make_learning_curve_labels(
        model_df["Conversion_pct"].astype(float).values,
        model_df["Yield_pct"].astype(float).values,
        k
    )

    # Use a simple, stable NN configuration for the
    # learning-curve visualization.
    nn_model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=RANDOM_STATE
    )

    try:

        sampler = make_sampler(
            method,
            y_learning
        )

        nn_pipeline = ImbPipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "sampler",
                sampler
            ),
            (
                "model",
                nn_model
            )
        ])

        cv_learning = StratifiedKFold(
            n_splits=4,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        sizes, train_scores, cv_scores = learning_curve(
            nn_pipeline,
            X_all,
            y_learning,
            train_sizes=np.linspace(
                0.50,
                1.0,
                8
            ),
            cv=cv_learning,
            scoring="f1_weighted",
            shuffle=True,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        train_mean = train_scores.mean(axis=1)
        cv_mean = cv_scores.mean(axis=1)

        ax.plot(
            sizes,
            train_mean,
            "o-",
            linewidth=1.5,
            markersize=3,
            label="Training score"
        )

        ax.plot(
            sizes,
            cv_mean,
            "o-",
            linewidth=1.5,
            markersize=3,
            label="Cross-validation score"
        )

    except Exception as exc:

        # A second, more conservative learning-curve attempt.
        # This avoids losing the panel because the smallest
        # temporary training subset has too few members of a
        # K-means class for ADASYN.
        try:

            # If ADASYN cannot create a synthetic point for one
            # of the temporary learning-curve subsets, use a
            # transparent oversampling fallback for the curve.
            # This prevents an empty panel while the main
            # classification analysis still records fallback use.
            if method == "ADASYN":

                curve_sampler = RandomOverSampler(
                    random_state=RANDOM_STATE
                )

                curve_pipeline = ImbPipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    ),
                    (
                        "sampler",
                        curve_sampler
                    ),
                    (
                        "model",
                        nn_model
                    )
                ])

            else:

                curve_pipeline = nn_pipeline

            sizes, train_scores, cv_scores = learning_curve(
                curve_pipeline,
                X_all,
                y_learning,
                train_sizes=np.linspace(
                    0.50,
                    1.0,
                    8
                ),
                cv=cv_learning,
                scoring="f1_weighted",
                shuffle=True,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )

            train_mean = train_scores.mean(axis=1)
            cv_mean = cv_scores.mean(axis=1)

            ax.plot(
                sizes,
                train_mean,
                "o-",
                linewidth=1.5,
                markersize=3,
                label="Training score"
            )

            ax.plot(
                sizes,
                cv_mean,
                "o-",
                linewidth=1.5,
                markersize=3,
                label="Cross-validation score"
            )

        except Exception as exc2:

            ax.text(
                0.5,
                0.5,
                f"No stable learning curve\n{exc2}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8
            )

    ax.set_title(
        f"{method} — K = {k}",
        fontweight="bold"
    )

    ax.set_xlabel("Training samples")
    ax.set_ylabel("F1-score")

    ax.set_ylim(0, 1.05)

    ax.grid(
        alpha=0.15
    )

    ax.text(
        0.95,
        0.05,
        panel_label,
        transform=ax.transAxes,
        ha="right",
        fontweight="bold"
    )

    ax.legend(
        fontsize=7,
        loc="lower right"
    )


fig.suptitle(
    "Learning Curves of the Oversampled NN Model",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

figure11_file = os.path.join(
    OUTPUT_DIR,
    "Figure_11_Oversampled_NN_Learning_Curves_ROBUST.png"
)

plt.savefig(
    figure11_file,
    dpi=600,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("ROBUST OVERSAMPLING ANALYSIS COMPLETE")
print("=" * 80)

print("\nFigure 9:")
print(figure9_file)

print("\nFigure 10:")
print(figure10_file)

print("\nFigure 11:")
print(figure11_file)

print("\nResults:")
print(classification_results_file)
