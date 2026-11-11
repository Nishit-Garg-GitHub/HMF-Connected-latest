# ============================================================
# HMF ELECTROOXIDATION
# LEARNING CURVES FOR FOUR PERFORMANCE TARGETS
#
# TARGETS:
#   1. Conversion (%)
#   2. FDCA Yield (%)
#   3. Faradaic Efficiency (%)
#   4. Selectivity (%)
#
# MODELS:
#   1. Decision Tree
#   2. Support Vector Regression
#   3. Random Forest
#   4. Neural Network
#
# FIGURE:
#   4 x 4 = 16 panels
#
# Each panel:
#   Red   = Training R²
#   Green = Cross-validation R²
#
# X-axis = Training examples
# Y-axis = R² score
#
# IMPORTANT:
#   The four targets are never used as input features.
#   Each target is predicted separately using the same
#   experimental/process/catalyst input variables.
# ============================================================


import os
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import r2_score


warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(ROOT, "HMF_ML_Preprocessed.xlsx")

OUTPUT_DIR = os.path.join(ROOT, "outputs", "Learning_Curves")

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


RANDOM_STATE = 42


# ============================================================
# 2. TARGET VARIABLES
# ============================================================

TARGETS = {

    "Conversion_pct":
        "Conversion (%)",

    "Yield_pct":
        "FDCA Yield (%)",

    "FE_pct":
        "Faradaic Efficiency (%)",

    "Selectivity_pct":
        "Selectivity (%)"

}


# ============================================================
# 3. VARIABLES THAT MUST NOT BE INPUT FEATURES
# ============================================================

EXCLUDE_COLUMNS = [

    # --------------------------------------------------------
    # All four targets
    # --------------------------------------------------------

    "Conversion_pct",

    "Yield_pct",

    "FE_pct",

    "Selectivity_pct",


    # --------------------------------------------------------
    # Outcome-derived variable
    # --------------------------------------------------------

    "FDCA_Target",


    # --------------------------------------------------------
    # Data-quality flags
    # --------------------------------------------------------

    "Yield_gt_Conversion",

    "Conversion_gt_100",

    "Yield_gt_100",

    "FE_gt_100",


    # --------------------------------------------------------
    # Potential quality flag
    # --------------------------------------------------------

    "Potential_vs_RHE"

]


# ============================================================
# 4. LOAD DATA
# ============================================================

print("=" * 80)

print(
    "HMF ELECTROOXIDATION"
)

print(
    "LEARNING CURVE ANALYSIS"
)

print("=" * 80)


df = pd.read_excel(
    DATA_FILE
)


print(
    "\nDataset shape:",
    df.shape
)


# ============================================================
# 5. SELECT INPUT FEATURES
# ============================================================

numeric_columns = (

    df
    .select_dtypes(
        include=[np.number]
    )
    .columns
    .tolist()

)


INPUT_COLUMNS = [

    col

    for col in numeric_columns

    if col not in EXCLUDE_COLUMNS

]


# Remove completely empty columns

INPUT_COLUMNS = [

    col

    for col in INPUT_COLUMNS

    if df[col].notna().sum() > 0

]


# Remove constant columns

constant_columns = [

    col

    for col in INPUT_COLUMNS

    if df[col].nunique(
        dropna=True
    ) <= 1

]


INPUT_COLUMNS = [

    col

    for col in INPUT_COLUMNS

    if col not in constant_columns

]


print("\n")
print("=" * 80)

print(
    "INPUT FEATURES"
)

print("=" * 80)


print(
    "\nNumber of input features:",
    len(INPUT_COLUMNS)
)


for i, col in enumerate(
    INPUT_COLUMNS,
    1
):

    print(
        f"{i:2d}. {col}"
    )


print("\nTargets:")

for target_name in TARGETS.values():

    print(
        f"- {target_name}"
    )


# ============================================================
# 6. CREATE INPUT DATASET
# ============================================================

X_all = df[
    INPUT_COLUMNS
].copy()


X_all = X_all.apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# 7. DEFINE MODELS
# ============================================================

models = {

    # --------------------------------------------------------
    # DECISION TREE
    # --------------------------------------------------------

    "DT": Pipeline([

        (
            "imputer",

            SimpleImputer(
                strategy="median"
            )

        ),

        (
            "model",

            DecisionTreeRegressor(

                random_state=RANDOM_STATE,

                max_depth=None,

                min_samples_leaf=2

            )

        )

    ]),


    # --------------------------------------------------------
    # SUPPORT VECTOR REGRESSION
    # --------------------------------------------------------

    "SVR": Pipeline([

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
            "model",

            SVR(

                kernel="rbf",

                C=100,

                epsilon=0.1,

                gamma="scale"

            )

        )

    ]),


    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    "RF": Pipeline([

        (
            "imputer",

            SimpleImputer(
                strategy="median"
            )

        ),

        (
            "model",

            RandomForestRegressor(

                n_estimators=300,

                random_state=RANDOM_STATE,

                n_jobs=-1,

                min_samples_leaf=2

            )

        )

    ]),


    # --------------------------------------------------------
    # NEURAL NETWORK
    # --------------------------------------------------------

    "NN": Pipeline([

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
            "model",

            MLPRegressor(

                hidden_layer_sizes=(
                    64,
                    32
                ),

                activation="relu",

                solver="adam",

                alpha=0.001,

                learning_rate_init=0.001,

                max_iter=1500,

                early_stopping=True,

                validation_fraction=0.15,

                random_state=RANDOM_STATE

            )

        )

    ])

}


# ============================================================
# 8. TRAINING-SAMPLE SIZES
# ============================================================

# We use fractions of the available dataset so that
# targets with different numbers of observations can
# still be analyzed.

TRAINING_FRACTIONS = np.array([

    0.15,

    0.25,

    0.35,

    0.45,

    0.55,

    0.65,

    0.75,

    0.85,

    1.00

])


# ============================================================
# 9. CROSS-VALIDATION SETTINGS
# ============================================================

N_SPLITS = 5


# ============================================================
# 10. FUNCTION FOR LEARNING CURVE
# ============================================================

def calculate_learning_curve(

    model,

    X,

    y,

    training_fractions

):


    n_samples = len(y)


    # --------------------------------------------------------
    # Determine training sizes
    # --------------------------------------------------------

    training_sizes = (

        training_fractions
        * n_samples

    ).astype(int)


    training_sizes = np.unique(
        training_sizes
    )


    # Need enough samples for 5-fold CV

    training_sizes = training_sizes[
        training_sizes >= N_SPLITS
    ]


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    train_scores = []

    cv_scores = []

    actual_sizes = []


    # --------------------------------------------------------
    # Fixed shuffled CV
    # --------------------------------------------------------

    kfold = KFold(

        n_splits=N_SPLITS,

        shuffle=True,

        random_state=RANDOM_STATE

    )


    # ========================================================
    # LOOP THROUGH TRAINING SIZES
    # ========================================================

    for train_size in training_sizes:


        fold_train_scores = []

        fold_cv_scores = []


        # ----------------------------------------------------
        # Perform 5-fold CV
        # ----------------------------------------------------

        for train_idx, validation_idx in kfold.split(X):


            # ------------------------------------------------
            # If the fold training set is smaller than
            # requested size, use the requested subset.
            # ------------------------------------------------

            if train_size > len(train_idx):

                continue


            # ------------------------------------------------
            # Select a reproducible subset
            # ------------------------------------------------

            rng = np.random.RandomState(

                RANDOM_STATE
                + train_size

            )


            selected_idx = rng.choice(

                train_idx,

                size=train_size,

                replace=False

            )


            X_train = X.iloc[
                selected_idx
            ]

            y_train = y.iloc[
                selected_idx
            ]


            X_valid = X.iloc[
                validation_idx
            ]

            y_valid = y.iloc[
                validation_idx
            ]


            # ------------------------------------------------
            # Clone model by rebuilding using sklearn clone
            # ------------------------------------------------

            from sklearn.base import clone


            current_model = clone(
                model
            )


            # ------------------------------------------------
            # Train
            # ------------------------------------------------

            current_model.fit(

                X_train,

                y_train

            )


            # ------------------------------------------------
            # Training prediction
            # ------------------------------------------------

            train_prediction = (

                current_model
                .predict(
                    X_train
                )

            )


            # ------------------------------------------------
            # Validation prediction
            # ------------------------------------------------

            validation_prediction = (

                current_model
                .predict(
                    X_valid
                )

            )


            # ------------------------------------------------
            # R²
            # ------------------------------------------------

            train_r2 = r2_score(

                y_train,

                train_prediction

            )


            validation_r2 = r2_score(

                y_valid,

                validation_prediction

            )


            fold_train_scores.append(
                train_r2
            )


            fold_cv_scores.append(
                validation_r2
            )


        # ----------------------------------------------------
        # Store average score
        # ----------------------------------------------------

        if len(fold_cv_scores) > 0:

            actual_sizes.append(
                train_size
            )

            train_scores.append(
                np.mean(
                    fold_train_scores
                )
            )

            cv_scores.append(
                np.mean(
                    fold_cv_scores
                )
            )


    return (

        np.array(
            actual_sizes
        ),

        np.array(
            train_scores
        ),

        np.array(
            cv_scores
        )

    )


# ============================================================
# 11. STORE ALL RESULTS
# ============================================================

all_results = []


# ============================================================
# 12. CREATE 4 × 4 FIGURE
#
# ROWS:
#   Conversion
#   Yield
#   FE
#   Selectivity
#
# COLUMNS:
#   DT
#   SVR
#   RF
#   NN
# ============================================================

fig, axes = plt.subplots(

    4,

    4,

    figsize=(15, 13)

)


# ============================================================
# 13. LOOP THROUGH TARGETS
# ============================================================

for row, (

    target_column,

    target_label

) in enumerate(
    TARGETS.items()
):


    print("\n")
    print("=" * 80)

    print(
        target_label
    )

    print("=" * 80)


    # --------------------------------------------------------
    # Select rows having target value
    # --------------------------------------------------------

    target_mask = (

        df[target_column]
        .notna()

    )


    X = X_all.loc[
        target_mask
    ].copy()


    y = (

        df.loc[
            target_mask,
            target_column
        ]

        .astype(float)

    )


    print(
        f"Available experiments: {len(y)}"
    )


    # ========================================================
    # LOOP THROUGH MODELS
    # ========================================================

    for col, (

        model_name,

        model

    ) in enumerate(
        models.items()
    ):


        print(
            f"\n{target_label} — {model_name}"
        )


        # ----------------------------------------------------
        # Calculate learning curve
        # ----------------------------------------------------

        (

            training_sizes,

            train_scores,

            cv_scores

        ) = calculate_learning_curve(

            model,

            X,

            y,

            TRAINING_FRACTIONS

        )


        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        for size, train_score, cv_score in zip(

            training_sizes,

            train_scores,

            cv_scores

        ):

            all_results.append(

                {

                    "Target":
                        target_label,

                    "Target_Column":
                        target_column,

                    "Model":
                        model_name,

                    "Training_Examples":
                        size,

                    "Training_R2":
                        train_score,

                    "Cross_Validation_R2":
                        cv_score

                }

            )


        # ====================================================
        # CURRENT AXIS
        # ====================================================

        ax = axes[
            row,
            col
        ]


        # ====================================================
        # TRAINING CURVE
        # ====================================================

        ax.plot(

            training_sizes,

            train_scores,

            color="red",

            marker="o",

            markersize=3.5,

            linewidth=1.2,

            label="Training R²"

        )


        # ====================================================
        # CROSS-VALIDATION CURVE
        # ====================================================

        ax.plot(

            training_sizes,

            cv_scores,

            color="green",

            marker="o",

            markersize=3.5,

            linewidth=1.2,

            label="Cross-validation R²"

        )


        # ====================================================
        # AXIS LABELS
        # ====================================================

        ax.set_xlabel(

            "Training examples",

            fontsize=8

        )


        ax.set_ylabel(

            "R² score",

            fontsize=8

        )


        # ====================================================
        # TITLE
        # ====================================================

        ax.set_title(

            f"{target_label} — {model_name}",

            fontsize=10,

            fontweight="bold"

        )


        # ====================================================
        # Y LIMIT
        # ====================================================

        # R² can be negative for poor models.
        # Use actual data range while maintaining
        # a readable graph.

        all_scores = np.concatenate(

            [

                train_scores,

                cv_scores

            ]

        )


        ymin = min(
            -0.1,
            np.min(
                all_scores
            ) - 0.05
        )


        ymax = max(

            1.02,

            np.max(
                all_scores
            ) + 0.03

        )


        ax.set_ylim(

            ymin,

            ymax

        )


        # ====================================================
        # GRID
        # ====================================================

        ax.grid(

            True,

            alpha=0.18,

            linewidth=0.5

        )


        # ====================================================
        # LEGEND
        # ====================================================

        ax.legend(

            fontsize=6.5,

            loc="lower right",

            frameon=True

        )


        # ====================================================
        # PANEL LETTER
        # ====================================================

        panel_number = (

            row * 4
            + col

        )


        panel_letter = chr(

            ord("a")
            + panel_number

        )


        ax.text(

            0.96,

            0.92,

            f"({panel_letter})",

            transform=ax.transAxes,

            fontsize=11,

            fontweight="bold",

            ha="right",

            va="top"

        )


        # ====================================================
        # SPINES
        # ====================================================

        ax.spines[
            "top"
        ].set_linewidth(
            0.7
        )

        ax.spines[
            "right"
        ].set_linewidth(
            0.7
        )

        ax.spines[
            "left"
        ].set_linewidth(
            0.7
        )

        ax.spines[
            "bottom"
        ].set_linewidth(
            0.7
        )


# ============================================================
# 14. OVERALL TITLE
# ============================================================

fig.suptitle(

    "Learning Curves for HMF Electrooxidation Performance Prediction",

    fontsize=16,

    fontweight="bold",

    y=0.995

)


# ============================================================
# 15. LAYOUT
# ============================================================

plt.subplots_adjust(

    left=0.06,

    right=0.98,

    bottom=0.06,

    top=0.94,

    wspace=0.30,

    hspace=0.40

)


# ============================================================
# 16. SAVE MAIN FIGURE
# ============================================================

main_figure = os.path.join(

    OUTPUT_DIR,

    "HMF_Learning_Curves_All_Targets.png"

)


plt.savefig(

    main_figure,

    dpi=600,

    bbox_inches="tight"

)


plt.show()

plt.close()


# ============================================================
# 17. SAVE NUMERICAL RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)


results_file = os.path.join(

    OUTPUT_DIR,

    "HMF_Learning_Curve_Results.csv"

)


results_df.to_csv(

    results_file,

    index=False

)


# ============================================================
# 18. PRINT FINAL INFORMATION
# ============================================================

print("\n")
print("=" * 80)

print(
    "LEARNING CURVE ANALYSIS COMPLETE"
)

print("=" * 80)


print(
    "\nMain figure:"
)

print(
    main_figure
)


print(
    "\nNumerical results:"
)

print(
    results_file
)


print("\nTargets analyzed:")

for target in TARGETS.values():

    print(
        f"- {target}"
    )


print("\nModels analyzed:")

for model in models.keys():

    print(
        f"- {model}"
    )


print(
    "\nFigure contains 16 panels:"
)

print(
    "4 targets × 4 ML models"
)