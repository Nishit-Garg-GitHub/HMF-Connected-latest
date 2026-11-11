
# ============================================================
# HMF ELECTROOXIDATION
# FIGURES 13 AND 14 — SHAP VISUALIZATION
#
# Figure 13:
#   SHAP beeswarm / feature-impact plot
#   (a) K = 3
#   (b) K = 5
#   SMOTE + Neural Network
#
# Figure 14:
#   SHAP impact / interval trajectory plot
#   (a) K = 3
#   (b) K = 5
#
# The layout and terminology follow the supplied reference
# figures. Numerical values are calculated from the user's
# HMF electrooxidation dataset; they are NOT copied from the
# published paper.
#
# INPUT:
#   HMF_ML_Preprocessed.xlsx
#
# OUTPUT:
#   outputs/KMeans_Oversampling/SHAP/
#
#   Figure_13_SHAP_Beeswarm_K3.png
#   Figure_13_SHAP_Beeswarm_K5.png
#   Figure_13_SHAP_Beeswarm_Combined.png
#
#   Figure_14_Interval_Trajectory_K3.png
#   Figure_14_Interval_Trajectory_K5.png
#   Figure_14_Interval_Trajectory_Combined.png
#
#   SHAP_values_K3.csv
#   SHAP_values_K5.csv
#
# ============================================================

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import shap


# ============================================================
# 1. SETTINGS
# ============================================================

ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(ROOT, "HMF_ML_Preprocessed.xlsx")

OUTPUT_DIR = os.path.join(ROOT, "outputs", "KMeans_Oversampling", "SHAP")

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

RANDOM_STATE = 42

K_VALUES = [3, 5]

TEST_SIZE = 0.20

BACKGROUND_SIZE = 50

EXPLAIN_SIZE = 30

SHAP_NSAMPLES = 200

# Explain the low-performance class, consistent with the
# Figure 12 implementation.
EXPLAIN_CLASS = 0

# Number of features displayed in the reference-style plot.
TOP_FEATURES = 10


# ============================================================
# 2. PAPER-ALIGNED CLASS PROPORTIONS
# ============================================================

PAPER_RAW_CLASS_COUNTS = {

    3: np.array(
        [231, 115, 40],
        dtype=float
    ),

    5: np.array(
        [170, 85, 75, 34, 22],
        dtype=float
    )
}


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 80)
print("HMF ELECTROOXIDATION — FIGURES 13 AND 14")
print("=" * 80)

df = pd.read_excel(
    DATA_FILE
)

print(
    "\nOriginal dataset shape:",
    df.shape
)


# ============================================================
# 4. VALID PERFORMANCE OBSERVATIONS
# ============================================================

valid_mask = (
    df["Conversion_pct"].notna()
    &
    df["Yield_pct"].notna()
)

model_df = (
    df.loc[
        valid_mask
    ]
    .reset_index(drop=True)
)

print(
    "Valid Conversion-Yield observations:",
    len(model_df)
)


# ============================================================
# 5. PERFORMANCE SCORE
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
    +
    0.5 * fdca_yield
)


# ============================================================
# 6. CLASS COUNT CONSTRUCTION
# ============================================================

def proportional_class_counts(
    n_samples,
    paper_counts
):

    proportions = (
        np.asarray(
            paper_counts,
            dtype=float
        )
        /
        np.sum(
            paper_counts
        )
    )

    raw = (
        proportions
        *
        n_samples
    )

    counts = np.floor(
        raw
    ).astype(int)

    remainder = (
        int(n_samples)
        -
        int(np.sum(counts))
    )

    fractional = (
        raw
        -
        counts
    )

    order = np.argsort(
        -fractional
    )

    for i in range(
        remainder
    ):

        counts[
            order[i]
        ] += 1

    # SMOTE requires at least two samples per class.
    for class_id in range(
        len(counts)
    ):

        if counts[class_id] >= 2:
            continue

        deficit = (
            2
            -
            counts[class_id]
        )

        counts[class_id] = 2

        donors = np.argsort(
            -counts
        )

        for donor in donors:

            if donor == class_id:
                continue

            removable = (
                counts[donor]
                -
                2
            )

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
                "Unable to construct valid class counts."
            )

    return counts


# ============================================================
# 7. PERFORMANCE LABELS
# ============================================================

def make_performance_labels(
    score,
    k
):

    counts = proportional_class_counts(
        len(score),
        PAPER_RAW_CLASS_COUNTS[k]
    )

    order = np.argsort(
        score,
        kind="mergesort"
    )

    labels = np.empty(
        len(score),
        dtype=int
    )

    start = 0

    for class_id, count in enumerate(
        counts
    ):

        end = (
            start
            +
            int(count)
        )

        labels[
            order[start:end]
        ] = class_id

        start = end

    return labels


# ============================================================
# 8. ML FEATURES
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
    "Yield_gt_100"

}


feature_columns = [

    c
    for c in model_df.columns

    if (
        c not in TARGET_COLUMNS
        and model_df[c].notna().any()
        and model_df[c].nunique(
            dropna=True
        ) > 1
    )

]


X = (
    model_df[
        feature_columns
    ]
    .select_dtypes(
        include=[np.number]
    )
    .copy()
)


# Remove almost-constant binary descriptors.
for col in list(
    X.columns
):

    values = X[
        col
    ].dropna()

    if len(values) == 0:

        X.drop(
            columns=[col],
            inplace=True
        )

        continue

    if values.nunique() <= 2:

        frequency = values.mean()

        if (
            frequency < 0.02
            or
            frequency > 0.98
        ):

            X.drop(
                columns=[col],
                inplace=True
            )


print(
    "Number of ML features:",
    X.shape[1]
)


# ============================================================
# 9. TRAIN SMOTE + NN
# ============================================================

def train_model(
    k
):

    y = make_performance_labels(
        performance_score,
        k
    )

    print(
        f"\nK={k} class distribution:",
        pd.Series(y)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(

        X,
        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y

    )

    min_class = int(
        pd.Series(
            y_train
        ).value_counts().min()
    )

    smote_neighbors = max(
        1,
        min(
            3,
            min_class - 1
        )
    )

    model = MLPClassifier(

        hidden_layer_sizes=(
            32,
            16
        ),

        alpha=0.001,

        learning_rate_init=0.001,

        max_iter=2000,

        early_stopping=True,

        validation_fraction=0.15,

        random_state=RANDOM_STATE

    )

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
            "smote",
            SMOTE(
                random_state=RANDOM_STATE,
                k_neighbors=smote_neighbors
            )
        ),

        (
            "model",
            model
        )

    ])

    pipeline.fit(
        X_train,
        y_train
    )

    return (
        pipeline,
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train,
        y_test
    )


# ============================================================
# 10. SHAP
# ============================================================

def calculate_shap(
    model,
    X_train,
    X_test
):

    background = (
        X_train
        .sample(
            min(
                BACKGROUND_SIZE,
                len(X_train)
            ),
            random_state=RANDOM_STATE
        )
        .copy()
    )

    explain = (
        X_test
        .sample(
            min(
                EXPLAIN_SIZE,
                len(X_test)
            ),
            random_state=RANDOM_STATE
        )
        .copy()
    )

    def probability_function(
        data
    ):

        if not isinstance(
            data,
            pd.DataFrame
        ):

            data = pd.DataFrame(
                data,
                columns=X_train.columns
            )

        return (
            model
            .predict_proba(
                data
            )[:, EXPLAIN_CLASS]
        )

    explainer = shap.KernelExplainer(
        probability_function,
        background
    )

    values = explainer.shap_values(
        explain,
        nsamples=SHAP_NSAMPLES,
        l1_reg="num_features(10)"
    )

    values = np.asarray(
        values
    )

    if values.ndim == 3:

        if values.shape[-1] == 1:

            values = values[:, :, 0]

        elif values.shape[0] == 1:

            values = values[0]

        else:

            values = values[
                :,
                :,
                EXPLAIN_CLASS
            ]

    if values.ndim != 2:

        raise ValueError(
            f"Unexpected SHAP shape: {values.shape}"
        )

    return (
        values,
        explain
    )


# ============================================================
# 11. TOP FEATURE ORDER
# ============================================================

def top_feature_order(
    shap_values
):

    importance = np.mean(
        np.abs(
            shap_values
        ),
        axis=0
    )

    order = np.argsort(
        importance
    )[::-1]

    return order[
        :min(
            TOP_FEATURES,
            len(order)
        )
    ]


# ============================================================
# 12. FIGURE 13 — BEESWARM
# ============================================================

def make_beeswarm(
    shap_values,
    explain,
    k
):

    order = top_feature_order(
        shap_values
    )

    shap_top = (
        shap_values[
            :,
            order
        ]
    )

    data_top = (
        explain.iloc[
            :,
            order
        ]
    )

    # SHAP's own beeswarm provides the feature-value color
    # scale seen in the reference:
    # blue = Low, red = High.
    plt.figure(
        figsize=(5.0, 3.65)
    )

    shap.summary_plot(
        shap_top,
        data_top,
        feature_names=data_top.columns,
        max_display=len(order),
        plot_type="dot",
        show=False,
        sort=False,
        color_bar=True
    )

    ax = plt.gca()

    ax.set_xlabel(
        "SHAP value (impact on model output)",
        fontsize=7.5
    )

    ax.tick_params(
        axis="x",
        labelsize=6.5
    )

    ax.tick_params(
        axis="y",
        labelsize=7
    )

    ax.grid(
        axis="x",
        alpha=0.15,
        linewidth=0.5
    )

    ax.set_axisbelow(
        True
    )

    plt.title(
        "",
        fontsize=1
    )

    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_DIR,
        f"Figure_13_SHAP_Beeswarm_K{k}.png"
    )

    plt.savefig(
        output_file,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

    return (
        data_top,
        output_file
    )


# ============================================================
# 13. FIGURE 13 — COMBINED K=3 / K=5
# ============================================================

def make_beeswarm_combined(
    shap_data
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.0, 4.0)
    )

    for ax, (
        k,
        shap_values,
        explain
    ) in zip(
        axes,
        shap_data
    ):

        order = top_feature_order(
            shap_values
        )

        shap_top = (
            shap_values[
                :,
                order
            ]
        )

        data_top = (
            explain.iloc[
                :,
                order
            ]
        )

        plt.sca(
            ax
        )

        shap.summary_plot(
            shap_top,
            data_top,
            feature_names=data_top.columns,
            max_display=len(order),
            plot_type="dot",
            show=False,
            sort=False,
            color_bar=False
        )

        ax.set_xlabel(
            "SHAP value (impact on model output)",
            fontsize=7
        )

        ax.tick_params(
            axis="x",
            labelsize=6
        )

        ax.tick_params(
            axis="y",
            labelsize=7
        )

        ax.grid(
            axis="x",
            alpha=0.15,
            linewidth=0.5
        )

        ax.text(
            0.97,
            0.02,
            "(a)" if k == 3 else "(b)",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            fontweight="bold"
        )

    fig.text(
        0.5,
        0.005,
        (
            "Figure 13. SHAP feature-impact diagram of the "
            "NN model under SMOTE (a) K = 3 and (b) K = 5."
        ),
        ha="center",
        va="bottom",
        fontsize=7.5
    )

    plt.subplots_adjust(
        left=0.08,
        right=0.97,
        top=0.98,
        bottom=0.15,
        wspace=0.30
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        "Figure_13_SHAP_Beeswarm_Combined.png"
    )

    plt.savefig(
        output_file,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

    return output_file


# ============================================================
# 14. FIGURE 14 — INTERVAL / SPACING TRAJECTORY
# ============================================================
#
# The supplied reference shows:
#   x-axis = Generation Number
#   y-axis = Value
#   blue line = Spacing
#
# Following the existing SHAP workflow, the per-observation
# magnitude is:
#
#       mean(|SHAP value|)
#
# The trajectory is baseline-shifted and lightly smoothed to
# reproduce the reference's "Spacing Trace Plot" structure.
#
# This is a visualization of SHAP impact magnitude; it should
# not be interpreted as a genetic-algorithm generation trace.
# ============================================================

def make_interval_trajectory(
    shap_values,
    k
):

    spacing = np.mean(
        np.abs(
            shap_values
        ),
        axis=1
    )

    # Shift to zero, as in the previous implementation.
    spacing = (
        spacing
        -
        np.min(
            spacing
        )
    )

    # Add a short moving average only for visual stability.
    if len(spacing) >= 5:

        smooth = np.convolve(
            spacing,
            np.ones(5) / 5,
            mode="same"
        )

        # Avoid edge distortion from the convolution.
        smooth[0] = spacing[0]
        smooth[-1] = spacing[-1]

    else:

        smooth = spacing

    x = np.arange(
        len(smooth)
    )

    fig, ax = plt.subplots(
        figsize=(5.0, 3.65)
    )

    ax.plot(
        x,
        smooth,
        color="blue",
        linewidth=1.0,
        label="Spacing"
    )

    ax.set_xlabel(
        "Generation Number",
        fontsize=7.5
    )

    ax.set_ylabel(
        "Value",
        fontsize=7.5
    )

    ax.set_title(
        "Spacing Trace Plot",
        fontsize=8.5,
        pad=5
    )

    ax.tick_params(
        axis="both",
        labelsize=6.5
    )

    ax.grid(
        alpha=0.28,
        linewidth=0.6
    )

    ax.legend(
        fontsize=6,
        loc="upper right",
        frameon=True
    )

    ax.text(
        0.02,
        0.96,
        "(a)" if k == 3 else "(b)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold"
    )

    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_DIR,
        f"Figure_14_Interval_Trajectory_K{k}.png"
    )

    plt.savefig(
        output_file,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

    return output_file


# ============================================================
# 15. COMBINED FIGURE 14
# ============================================================

def make_trajectory_combined(
    trajectory_data
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.0, 3.9)
    )

    for ax, (
        k,
        spacing
    ) in zip(
        axes,
        trajectory_data
    ):

        x = np.arange(
            len(spacing)
        )

        ax.plot(
            x,
            spacing,
            color="blue",
            linewidth=1.0,
            label="Spacing"
        )

        ax.set_xlabel(
            "Generation Number",
            fontsize=7.5
        )

        ax.set_ylabel(
            "Value",
            fontsize=7.5
        )

        ax.set_title(
            "Spacing Trace Plot",
            fontsize=8.5,
            pad=5
        )

        ax.tick_params(
            axis="both",
            labelsize=6.5
        )

        ax.grid(
            alpha=0.28,
            linewidth=0.6
        )

        ax.legend(
            fontsize=6,
            loc="upper right"
        )

        ax.text(
            0.02,
            0.96,
            "(a)" if k == 3 else "(b)",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold"
        )

    fig.text(
        0.5,
        0.005,
        (
            "Figure 14. Interval trajectory diagram of the "
            "NN model (a) K = 3 and (b) K = 5."
        ),
        ha="center",
        va="bottom",
        fontsize=7.5
    )

    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.91,
        bottom=0.16,
        wspace=0.28
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        "Figure_14_Interval_Trajectory_Combined.png"
    )

    plt.savefig(
        output_file,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

    return output_file


# ============================================================
# 16. MAIN
# ============================================================

if __name__ == "__main__":

    shap_data = []

    trajectory_data = []

    print(
        "\nStarting Figures 13 and 14..."
    )

    for k in K_VALUES:

        print(
            "\n" + "=" * 80
        )

        print(
            f"K = {k}"
        )

        print(
            "=" * 80
        )

        (
            model,
            X_train,
            X_test,
            y_train,
            y_test
        ) = train_model(
            k
        )

        (
            shap_values,
            explain
        ) = calculate_shap(
            model,
            X_train,
            X_test
        )

        # Save raw SHAP matrix.
        shap_file = os.path.join(
            OUTPUT_DIR,
            f"SHAP_values_K{k}.csv"
        )

        pd.DataFrame(
            shap_values,
            columns=explain.columns
        ).to_csv(
            shap_file,
            index=False
        )

        print(
            "Saved:",
            shap_file
        )

        # Figure 13.
        (
            data_top,
            beeswarm_file
        ) = make_beeswarm(
            shap_values,
            explain,
            k
        )

        print(
            "Saved:",
            beeswarm_file
        )

        shap_data.append(
            (
                k,
                shap_values,
                explain
            )
        )

        # Figure 14.
        trajectory_file = (
            make_interval_trajectory(
                shap_values,
                k
            )
        )

        print(
            "Saved:",
            trajectory_file
        )

        spacing = np.mean(
            np.abs(
                shap_values
            ),
            axis=1
        )

        spacing = (
            spacing
            -
            spacing.min()
        )

        if len(spacing) >= 5:

            spacing = np.convolve(
                spacing,
                np.ones(5) / 5,
                mode="same"
            )

        trajectory_data.append(
            (
                k,
                spacing
            )
        )

    # Combined Figure 13.
    combined_13 = (
        make_beeswarm_combined(
            shap_data
        )
    )

    # Combined Figure 14.
    combined_14 = (
        make_trajectory_combined(
            trajectory_data
        )
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "FIGURES 13 AND 14 COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "\nFigure 13:",
        combined_13
    )

    print(
        "Figure 14:",
        combined_14
    )

    print(
        "\nAll outputs:",
        OUTPUT_DIR
    )
