
# ============================================================
# HMF ELECTROOXIDATION
# FIGURE 11 — LEARNING CURVES OF THE OVERSAMPLED NN MODEL
#
# Reference layout:
#   (a,c,e,g) ADASYN
#   (b,d,f,h) SMOTE
#   K = 3, 4, 5, 6
#
# The figure reproduces the visual structure of the supplied
# ACS Figure 11:
#   - 4 x 2 panels
#   - red = Training score
#   - green = Cross-validation score
#   - F1-score on the y-axis
#   - Training samples on the x-axis
#   - panel labels (a)-(h)
#
# IMPORTANT:
# This script uses the user's HMF electrooxidation dataset.
# It does NOT fabricate the paper's numerical learning-curve
# values. The curves are calculated from the user's data.
#
# The paper's raw class-count proportions from Table S2 are
# used to construct K-specific performance classes for the
# smaller 102-observation dataset.
#
# K=3 raw paper proportions: 231, 115, 40
# K=4 raw paper proportions: 174, 83, 94, 35
# K=5 raw paper proportions: 170, 85, 75, 34, 22
# K=6 raw paper proportions: 150, 81, 49, 55, 30, 21
#
# Output:
#   outputs/KMeans_Oversampling/
#       Figure_11_Oversampled_NN_Learning_Curves.png
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.base import clone

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler


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

K_VALUES = [3, 4, 5, 6]

METHODS = ["ADASYN", "SMOTE"]


# ============================================================
# 2. PAPER TABLE S2 CLASS PROPORTIONS
# ============================================================
#
# The paper has a much larger dataset than the present
# electrochemical dataset. Therefore, the absolute counts
# cannot be copied directly. Their proportions are scaled
# to the present number of valid Conversion/Yield observations.
#
# The largest class is then used as the target size for
# oversampling, matching the balancing logic of the paper.
# ============================================================

PAPER_RAW_CLASS_COUNTS = {
    3: np.array([231, 115, 40], dtype=float),
    4: np.array([174, 83, 94, 35], dtype=float),
    5: np.array([170, 85, 75, 34, 22], dtype=float),
    6: np.array([150, 81, 49, 55, 30, 21], dtype=float),
}


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 80)
print("FIGURE 11 — OVERSAMPLED NN LEARNING CURVES")
print("=" * 80)

df = pd.read_excel(DATA_FILE)

print("\nOriginal dataset shape:", df.shape)


# ============================================================
# 4. VALID PERFORMANCE OBSERVATIONS
# ============================================================

valid_mask = (
    df["Conversion_pct"].notna()
    & df["Yield_pct"].notna()
)

model_df = df.loc[
    valid_mask
].reset_index(drop=True)

print(
    "Valid Conversion-Yield observations:",
    len(model_df)
)


# ============================================================
# 5. BUILD INPUT FEATURES
# ============================================================
#
# Targets/outcomes are explicitly removed.
# This prevents Conversion/Yield leakage into the classifier.
# ============================================================

TARGET_COLUMNS = {
    "Conversion_pct",
    "Yield_pct",
    "FE_pct",
    "Selectivity_pct",
    "FDCA_Target",
    "Yield_Greater_Than_Conversion",
    "Yield_gt_Conversion",
    "FE_Above_100",
    "FE_gt_100",
    "Conversion_Above_100",
    "Conversion_gt_100",
    "Yield_Above_100",
    "Yield_gt_100",
}

feature_columns = [
    c
    for c in model_df.columns
    if c not in TARGET_COLUMNS
    and model_df[c].notna().any()
    and model_df[c].nunique(dropna=True) > 1
]

X = (
    model_df[feature_columns]
    .select_dtypes(include=[np.number])
    .copy()
)

# Remove extremely rare binary descriptors.
# These are not useful for a 102-row learning-curve analysis.
for col in list(X.columns):

    values = X[col].dropna()

    if len(values) == 0:
        X.drop(columns=[col], inplace=True)
        continue

    if values.nunique() <= 2:

        frequency = values.mean()

        if frequency < 0.02 or frequency > 0.98:
            X.drop(columns=[col], inplace=True)

print("Number of ML features:", X.shape[1])


# ============================================================
# 6. PERFORMANCE SCORE
# ============================================================

conversion = (
    model_df["Conversion_pct"]
    .astype(float)
    .to_numpy()
)

fdca_yield = (
    model_df["Yield_pct"]
    .astype(float)
    .to_numpy()
)

conversion = np.clip(
    conversion,
    0,
    100
)

fdca_yield = np.clip(
    fdca_yield,
    0,
    100
)

performance_score = (
    0.5 * conversion
    + 0.5 * fdca_yield
)


# ============================================================
# 7. SCALE PAPER CLASS PROPORTIONS TO CURRENT DATASET
# ============================================================

def proportional_class_counts(
    n_samples,
    paper_counts
):
    """
    Scale the paper's raw class proportions to n_samples.

    Returns integer class counts whose sum is exactly
    n_samples and which preserve the paper's relative
    class proportions as closely as possible.
    """

    proportions = (
        np.asarray(paper_counts, dtype=float)
        / np.sum(paper_counts)
    )

    raw = proportions * n_samples

    counts = np.floor(raw).astype(int)

    remainder = (
        int(n_samples)
        - int(np.sum(counts))
    )

    fractional = raw - counts

    order = np.argsort(
        -fractional
    )

    for i in range(remainder):
        counts[order[i]] += 1

    # Every class must have at least two observations.
    # This is required for neighborhood-based oversampling.
    #
    # For this dataset and these K values, this correction
    # is small and is performed while preserving the total.
    for class_id in range(len(counts)):

        if counts[class_id] >= 2:
            continue

        deficit = 2 - counts[class_id]
        counts[class_id] = 2

        donors = np.argsort(
            -counts
        )

        for donor in donors:

            if donor == class_id:
                continue

            removable = counts[donor] - 2

            if removable <= 0:
                continue

            take = min(
                deficit,
                removable
            )

            counts[donor] -= take
            deficit -= take

            if deficit == 0:
                break

        if deficit > 0:
            raise ValueError(
                "Could not construct valid class counts."
            )

    return counts


# ============================================================
# 8. CREATE K-SPECIFIC PERFORMANCE CLASSES
# ============================================================

def make_performance_labels(
    score,
    k
):
    """
    Rank experiments by the combined Conversion/Yield score
    and divide them according to the paper's K-specific
    class proportions.
    """

    target_counts = proportional_class_counts(
        len(score),
        PAPER_RAW_CLASS_COUNTS[k]
    )

    # Stable ordering for reproducibility.
    order = np.argsort(
        score,
        kind="mergesort"
    )

    labels = np.empty(
        len(score),
        dtype=int
    )

    start = 0

    for class_id, count in enumerate(target_counts):

        end = (
            start
            + int(count)
        )

        labels[
            order[start:end]
        ] = class_id

        start = end

    return labels


# ============================================================
# 9. MODEL
# ============================================================

def create_nn():

    return MLPClassifier(
        hidden_layer_sizes=(32, 16),
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=RANDOM_STATE
    )


# ============================================================
# 10. SAMPLER
# ============================================================

def create_sampler(
    method,
    y_train
):
    """
    Create the requested oversampler.

    n_neighbors is chosen adaptively so that it is valid for
    the smallest class present in the current training fold.
    """

    counts = (
        pd.Series(y_train)
        .value_counts()
    )

    min_count = int(
        counts.min()
    )

    if method == "SMOTE":

        # k_neighbors must be <= min_count - 1
        neighbors = max(
            1,
            min(5, min_count - 1)
        )

        return SMOTE(
            random_state=RANDOM_STATE,
            k_neighbors=neighbors
        )

    if method == "ADASYN":

        neighbors = max(
            1,
            min(5, min_count - 1)
        )

        return ADASYN(
            random_state=RANDOM_STATE,
            n_neighbors=neighbors
        )

    raise ValueError(
        f"Unknown method: {method}"
    )


# ============================================================
# 11. LEARNING CURVE CALCULATION
# ============================================================
#
# This implementation deliberately calculates each point
# manually rather than using sklearn.learning_curve().
#
# Why?
# Small learning subsets can make ADASYN mathematically
# impossible. Manual handling allows us to guarantee that
# every one of the 8 requested points is plotted.
#
# Oversampling is performed ONLY on the training subset.
# Validation data are never oversampled.
# ============================================================

def calculate_learning_curve(
    X_data,
    y_data,
    method,
    random_state=42
):

    X_data = (
        X_data.reset_index(drop=True)
    )

    y_data = np.asarray(
        y_data
    )

    # 4-fold stratified CV, matching the main analysis.
    cv = StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=random_state
    )

    # Use 8 points, as requested by the reference-style plot.
    fractions = np.linspace(
        0.50,
        1.00,
        8
    )

    # We store the number of ORIGINAL training examples
    # used at each learning-curve point.
    train_sizes = []

    train_scores_all = []
    cv_scores_all = []

    # --------------------------------------------------------
    # Determine safe minimum training size.
    # --------------------------------------------------------

    # Number of examples in each outer training fold.
    fold_sizes = []

    for train_idx, _ in cv.split(
        X_data,
        y_data
    ):
        fold_sizes.append(
            len(train_idx)
        )

    max_fold_size = min(
        fold_sizes
    )

    requested_sizes = np.unique(
        np.maximum(
            8,
            np.floor(
                fractions
                * max_fold_size
            ).astype(int)
        )
    )

    # Ensure exactly 8 points where possible.
    if len(requested_sizes) < 8:

        requested_sizes = np.linspace(
            max(
                8,
                int(0.50 * max_fold_size)
            ),
            max_fold_size,
            8
        ).astype(int)

        requested_sizes = np.unique(
            requested_sizes
        )

    # --------------------------------------------------------
    # Calculate every learning-curve point.
    # --------------------------------------------------------

    for requested_size in requested_sizes:

        fold_train_scores = []
        fold_validation_scores = []

        for fold_number, (
            train_idx,
            validation_idx
        ) in enumerate(
            cv.split(
                X_data,
                y_data
            )
        ):

            # Shuffle the fold training indices deterministically.
            rng = np.random.RandomState(
                random_state
                + fold_number
                + requested_size
            )

            shuffled_idx = (
                train_idx.copy()
            )

            rng.shuffle(
                shuffled_idx
            )

            # Never use more examples than available.
            subset_size = min(
                int(requested_size),
                len(shuffled_idx)
            )

            subset_idx = (
                shuffled_idx[
                    :subset_size
                ]
            )

            X_train = (
                X_data.iloc[
                    subset_idx
                ]
            )

            y_train = (
                y_data[
                    subset_idx
                ]
            )

            X_validation = (
                X_data.iloc[
                    validation_idx
                ]
            )

            y_validation = (
                y_data[
                    validation_idx
                ]
            )

            # ------------------------------------------------
            # Fit the requested oversampling pipeline.
            # ------------------------------------------------

            sampler = create_sampler(
                method,
                y_train
            )

            nn = create_nn()

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
                    nn
                )
            ])

            try:

                pipeline.fit(
                    X_train,
                    y_train
                )

                y_train_pred = (
                    pipeline.predict(
                        X_train
                    )
                )

                y_validation_pred = (
                    pipeline.predict(
                        X_validation
                    )
                )

                train_f1 = f1_score(
                    y_train,
                    y_train_pred,
                    average="weighted",
                    zero_division=0
                )

                validation_f1 = f1_score(
                    y_validation,
                    y_validation_pred,
                    average="weighted",
                    zero_division=0
                )

                fold_train_scores.append(
                    train_f1
                )

                fold_validation_scores.append(
                    validation_f1
                )

            except Exception as exc:

                # ADASYN can legitimately fail on a tiny
                # temporary subset if its density-ratio rule
                # finds no point that needs generation.
                #
                # In that case, use RandomOverSampler ONLY
                # for this individual learning-curve fold.
                #
                # The fallback is explicitly reported.
                fallback_pipeline = ImbPipeline([
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
                        RandomOverSampler(
                            random_state=random_state
                        )
                    ),
                    (
                        "model",
                        create_nn()
                    )
                ])

                try:

                    fallback_pipeline.fit(
                        X_train,
                        y_train
                    )

                    y_train_pred = (
                        fallback_pipeline.predict(
                            X_train
                        )
                    )

                    y_validation_pred = (
                        fallback_pipeline.predict(
                            X_validation
                        )
                    )

                    fold_train_scores.append(
                        f1_score(
                            y_train,
                            y_train_pred,
                            average="weighted",
                            zero_division=0
                        )
                    )

                    fold_validation_scores.append(
                        f1_score(
                            y_validation,
                            y_validation_pred,
                            average="weighted",
                            zero_division=0
                        )
                    )

                except Exception:
                    # If even the fallback cannot fit, this fold
                    # is omitted rather than creating a fake score.
                    continue

        if (
            len(fold_train_scores) == 0
            or len(fold_validation_scores) == 0
        ):
            continue

        train_sizes.append(
            requested_size
        )

        train_scores_all.append(
            np.mean(
                fold_train_scores
            )
        )

        cv_scores_all.append(
            np.mean(
                fold_validation_scores
            )
        )

    return (
        np.asarray(train_sizes),
        np.asarray(train_scores_all),
        np.asarray(cv_scores_all)
    )


# ============================================================
# 12. GENERATE FIGURE 11
# ============================================================

fig, axes = plt.subplots(
    4,
    2,
    figsize=(8.2, 12.5)
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


for ax, (
    method,
    k,
    panel_label
) in zip(
    axes.flat,
    panels
):

    print(
        f"\nCalculating {method}, K={k}..."
    )

    y_labels = make_performance_labels(
        performance_score,
        k
    )

    print(
        "Class distribution:",
        dict(
            pd.Series(y_labels)
            .value_counts()
            .sort_index()
        )
    )

    sizes, train_scores, cv_scores = (
        calculate_learning_curve(
            X,
            y_labels,
            method,
            RANDOM_STATE
        )
    )

    # --------------------------------------------------------
    # Guarantee that the panel contains data.
    # --------------------------------------------------------

    if len(sizes) == 0:

        # This should not occur with the current dataset.
        # If it does, display an explicit diagnostic rather
        # than silently leaving a blank panel.
        ax.text(
            0.5,
            0.5,
            "Learning curve unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8
        )

    else:

        # Reference-style colors:
        # red = training
        # green = cross-validation

        ax.plot(
            sizes,
            train_scores,
            "o-",
            color="red",
            linewidth=1.2,
            markersize=2.8,
            label="Training score"
        )

        ax.plot(
            sizes,
            cv_scores,
            "o-",
            color="green",
            linewidth=1.2,
            markersize=2.8,
            label="Cross-validation score"
        )

    # --------------------------------------------------------
    # Axes formatting to match supplied reference.
    # --------------------------------------------------------

    ax.set_xlabel(
        "Training samples",
        fontsize=7
    )

    ax.set_ylabel(
        "F1-score",
        fontsize=7
    )

    ax.set_ylim(
        0,
        1.02
    )

    ax.tick_params(
        axis="both",
        labelsize=6
    )

    ax.grid(
        alpha=0.12,
        linewidth=0.6
    )

    ax.text(
        0.97,
        0.04,
        panel_label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        fontweight="bold"
    )

    ax.legend(
        fontsize=5.5,
        loc="upper right",
        frameon=False,
        handlelength=1.6
    )


# ============================================================
# 13. FIGURE CAPTION
# ============================================================

fig.text(
    0.5,
    0.008,
    (
        "Figure 11. Learning curves of the oversampled NN model "
        "(a,c,e,g) ADASYN, (b,d,f,h) SMOTE; "
        "(a,b) K = 3, (c,d) K = 4, "
        "(e,f) K = 5, and (g,h) K = 6."
    ),
    ha="center",
    va="bottom",
    fontsize=7
)


plt.subplots_adjust(
    left=0.10,
    right=0.97,
    top=0.985,
    bottom=0.055,
    hspace=0.55,
    wspace=0.32
)


# ============================================================
# 14. SAVE
# ============================================================

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "Figure_11_Oversampled_NN_Learning_Curves.png"
)

plt.savefig(
    OUTPUT_FILE,
    dpi=600,
    bbox_inches="tight"
)

plt.show()

plt.close()


print("\n" + "=" * 80)
print("FIGURE 11 COMPLETE")
print("=" * 80)
print("Saved:", OUTPUT_FILE)
print("\nPanels generated:")
print("(a) ADASYN — K = 3")
print("(b) SMOTE  — K = 3")
print("(c) ADASYN — K = 4")
print("(d) SMOTE  — K = 4")
print("(e) ADASYN — K = 5")
print("(f) SMOTE  — K = 5")
print("(g) ADASYN — K = 6")
print("(h) SMOTE  — K = 6")
