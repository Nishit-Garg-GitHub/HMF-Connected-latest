# ============================================================
# 08_genetic_algorithms_V2_FAST.py
# ============================================================
#
# FAST / ROBUST INTEGRATED ML + NSGA-II
# Electrochemical HMF oxidation -> FDCA
#
# Purpose:
#   Fix the computational bottleneck in the previous
#   08_genetic algorithms.py implementation.
#
# Main fixes:
#   1. Random Forest uses n_jobs=1 to avoid nested joblib overhead.
#   2. Entire populations are predicted in one batch.
#   3. No per-individual sklearn .predict() calls.
#   4. Population size = 100 and generations = 100 are retained.
#   5. Crossover = 0.90 and mutation = 0.10 are retained.
#   6. The final surrogate is refit on all complete observations
#      before NSGA-II optimization.
#   7. Temperature and Potential are optimized only within the
#      observed dataset ranges.
#   8. Binary features are kept binary.
#   9. A distance-to-training-data diagnostic is reported so that
#      extrapolative GA solutions can be identified.
#
# IMPORTANT:
#   The ACS paper uses conversion, yield, temperature, pressure
#   and reaction time in its integrated GA formulation.
#   The present electrochemical dataset does not contain explicit
#   pressure/time variables, so pressure/time are NOT invented.
#   This is a methodological adaptation, not a literal reproduction.
#
# NSGA-II objectives used here:
#   1. maximize HMF conversion
#   2. maximize FDCA yield
#   3. minimize temperature
#   4. minimize electrode potential
#
# ============================================================

import os
import random
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(ROOT, "HMF_ML_Preprocessed.xlsx")
OUTPUT_DIR = os.path.join(ROOT, "outputs", "KMeans_Oversampling", "NSGA_II_V2")

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

POPULATION_SIZE = 100
MAX_GENERATIONS = 100
CROSSOVER_RATE = 0.90
MUTATION_RATE = 0.10

# RF is deliberately single-threaded.
# The GA evaluates complete populations in batches, so this avoids
# nested joblib process/thread overhead.
RF_TREES = 300
RF_MIN_SAMPLES_LEAF = 2

# Post-hoc compromise ranking only.
# NSGA-II itself remains genuinely multiobjective.
COMPROMISE_WEIGHTS = {
    "Conversion": 0.40,
    "FDCA_Yield": 0.40,
    "Temperature": 0.10,
    "Potential": 0.10,
}

CONVERSION_COLUMN = "Conversion_pct"
YIELD_COLUMN = "Yield_pct"
TEMPERATURE_COLUMN = "Temperature_C"
POTENTIAL_COLUMN = "Potential_V"

# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 82)
print("FAST INTEGRATED ML + NSGA-II")
print("ELECTROCHEMICAL HMF OXIDATION -> FDCA")
print("=" * 82)

print("\nLoading:", DATA_FILE)

df = pd.read_excel(DATA_FILE)

print("Original dataset shape:", df.shape)

required = [CONVERSION_COLUMN, YIELD_COLUMN]
missing_required = [c for c in required if c not in df.columns]
if missing_required:
    raise RuntimeError(
        f"Missing required target columns: {missing_required}"
    )

# Only observations with both targets can train the two surrogate models.
model_df = df.loc[
    df[CONVERSION_COLUMN].notna()
    & df[YIELD_COLUMN].notna()
].reset_index(drop=True)

print(
    "Rows with both Conversion and FDCA Yield:",
    len(model_df)
)

if len(model_df) < 30:
    raise RuntimeError(
        "Too few complete Conversion/Yield observations for surrogate optimization."
    )

# ============================================================
# 3. FEATURE SELECTION
# ============================================================

# Explicitly exclude target/outcome-derived columns.
TARGET_EXCLUDE = {
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

numeric_columns = model_df.select_dtypes(
    include=[np.number]
).columns.tolist()

FEATURE_COLUMNS = [
    c for c in numeric_columns
    if c not in TARGET_EXCLUDE
]

# Remove completely empty / constant features.
FEATURE_COLUMNS = [
    c for c in FEATURE_COLUMNS
    if model_df[c].notna().any()
    and model_df[c].nunique(dropna=True) > 1
]

if not FEATURE_COLUMNS:
    raise RuntimeError("No usable numeric input features were found.")

X_raw = model_df[FEATURE_COLUMNS].copy()
y_conversion = model_df[CONVERSION_COLUMN].astype(float).values
y_yield = model_df[YIELD_COLUMN].astype(float).values

# Remove extremely rare binary descriptors.
# A descriptor appearing in fewer than 2% of the usable observations
# is not a stable free GA decision variable.
drop_rare = []

for col in X_raw.columns:
    vals = X_raw[col].dropna().values
    if len(vals) == 0:
        drop_rare.append(col)
        continue

    unique = np.unique(vals)

    if len(unique) == 2 and set(unique).issubset({0, 1}):
        prevalence = np.mean(vals == 1)
        if prevalence < 0.02 or prevalence > 0.98:
            drop_rare.append(col)

if drop_rare:
    FEATURE_COLUMNS = [
        c for c in FEATURE_COLUMNS if c not in drop_rare
    ]
    X_raw = model_df[FEATURE_COLUMNS].copy()

print("\nNumber of GA input features:", len(FEATURE_COLUMNS))

# ============================================================
# 4. IMPUTATION + BOUNDS
# ============================================================

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X_raw)

X_df = pd.DataFrame(X, columns=FEATURE_COLUMNS)

LOWER_BOUNDS = X_df.min(axis=0).values.astype(float)
UPPER_BOUNDS = X_df.max(axis=0).values.astype(float)

BINARY_FEATURES = []

for i, col in enumerate(FEATURE_COLUMNS):
    original_values = X_raw[col].dropna().values

    if (
        len(original_values) > 0
        and len(np.unique(original_values)) <= 2
        and set(np.unique(original_values)).issubset({0, 1})
    ):
        BINARY_FEATURES.append(col)
        LOWER_BOUNDS[i] = 0.0
        UPPER_BOUNDS[i] = 1.0

print("Binary GA features:", len(BINARY_FEATURES))

if TEMPERATURE_COLUMN in FEATURE_COLUMNS:
    ti = FEATURE_COLUMNS.index(TEMPERATURE_COLUMN)
    print(
        f"Temperature bounds: "
        f"{LOWER_BOUNDS[ti]:.3f} -> {UPPER_BOUNDS[ti]:.3f} °C"
    )
else:
    raise RuntimeError(
        "Temperature_C is not available as a GA feature. "
        "Temperature is required for the adapted objective."
    )

if POTENTIAL_COLUMN in FEATURE_COLUMNS:
    pi = FEATURE_COLUMNS.index(POTENTIAL_COLUMN)
    print(
        f"Potential bounds: "
        f"{LOWER_BOUNDS[pi]:.3f} -> {UPPER_BOUNDS[pi]:.3f} V"
    )
else:
    raise RuntimeError(
        "Potential_V is not available as a GA feature. "
        "Potential is required for the adapted objective."
    )

# ============================================================
# 5. SURROGATE VALIDATION
# ============================================================

print("\n" + "=" * 82)
print("SURROGATE MODEL VALIDATION")
print("=" * 82)

indices = np.arange(len(X))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

def make_rf():
    return RandomForestRegressor(
        n_estimators=RF_TREES,
        max_features="sqrt",
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

conversion_test_model = make_rf()
yield_test_model = make_rf()

conversion_test_model.fit(X[train_idx], y_conversion[train_idx])
yield_test_model.fit(X[train_idx], y_yield[train_idx])

pred_conv_test = conversion_test_model.predict(X[test_idx])
pred_yield_test = yield_test_model.predict(X[test_idx])

conv_rmse = float(
    np.sqrt(mean_squared_error(y_conversion[test_idx], pred_conv_test))
)
conv_r2 = float(
    r2_score(y_conversion[test_idx], pred_conv_test)
)

yield_rmse = float(
    np.sqrt(mean_squared_error(y_yield[test_idx], pred_yield_test))
)
yield_r2 = float(
    r2_score(y_yield[test_idx], pred_yield_test)
)

performance = pd.DataFrame({
    "Target": ["Conversion", "FDCA Yield"],
    "RMSE": [conv_rmse, yield_rmse],
    "R2": [conv_r2, yield_r2],
    "Train_N": [len(train_idx), len(train_idx)],
    "Test_N": [len(test_idx), len(test_idx)],
})

performance.to_csv(
    os.path.join(OUTPUT_DIR, "Surrogate_Model_Performance.csv"),
    index=False,
)

print(f"Conversion surrogate: RMSE={conv_rmse:.4f}, R2={conv_r2:.4f}")
print(f"FDCA yield surrogate: RMSE={yield_rmse:.4f}, R2={yield_r2:.4f}")

# ============================================================
# 6. FINAL SURROGATES
# ============================================================
#
# After validation, refit on all usable observations.
# This maximizes information available to the optimizer.
# ============================================================

print("\nFitting final Random Forest surrogates on all complete observations...")

conversion_model = make_rf()
yield_model = make_rf()

conversion_model.fit(X, y_conversion)
yield_model.fit(X, y_yield)

print("Final surrogate models fitted.")

# ============================================================
# 7. TRAINING-DATA STANDARDIZATION
# ============================================================
#
# Used only as an extrapolation diagnostic.
# A low distance means a GA candidate is close to the experimental
# feature space. A very high distance means the RF is being asked
# to predict far from observed combinations.
# ============================================================

distance_scaler = StandardScaler()
X_scaled = distance_scaler.fit_transform(X)

# ============================================================
# 8. FAST POPULATION PREDICTION
# ============================================================

def repair_population(population):
    """
    Clip all genes to observed ranges and force binary variables
    back to {0,1}.
    """
    pop = np.asarray(population, dtype=float).copy()

    pop = np.clip(
        pop,
        LOWER_BOUNDS,
        UPPER_BOUNDS,
    )

    for feature in BINARY_FEATURES:
        idx = FEATURE_COLUMNS.index(feature)
        pop[:, idx] = (
            pop[:, idx] >= 0.5
        ).astype(float)

    # Explicit process bounds from the paper are only applied where
    # relevant and where the dataset supports the variable.
    if TEMPERATURE_COLUMN in FEATURE_COLUMNS:
        ti = FEATURE_COLUMNS.index(TEMPERATURE_COLUMN)
        pop[:, ti] = np.minimum(pop[:, ti], 200.0)

    return pop


def predict_population(population):
    """
    CRITICAL SPEED FIX:
    Predict the entire population in one call per surrogate.
    """
    pop = repair_population(population)

    conversion = conversion_model.predict(pop)
    fdca_yield = yield_model.predict(pop)

    ti = FEATURE_COLUMNS.index(TEMPERATURE_COLUMN)
    pi = FEATURE_COLUMNS.index(POTENTIAL_COLUMN)

    temperature = pop[:, ti]
    potential = pop[:, pi]

    # NSGA-II is implemented as minimization:
    #
    # maximize conversion -> minimize -conversion
    # maximize yield      -> minimize -yield
    # minimize temperature
    # minimize potential
    objectives = np.column_stack([
        -conversion,
        -fdca_yield,
        temperature,
        potential,
    ])

    return objectives, np.column_stack([
        conversion,
        fdca_yield,
        temperature,
        potential,
    ]), pop


# ============================================================
# 9. DISTANCE-TO-EXPERIMENTAL-SPACE
# ============================================================

def nearest_training_distance(population):
    """
    Calculate the Euclidean distance from each candidate to its
    nearest experimentally observed/imputed training point.

    This is a diagnostic only; it is NOT inserted into the four
    NSGA-II objectives, so it does not alter the optimization.
    """
    pop_scaled = distance_scaler.transform(
        repair_population(population)
    )

    # For only ~100-200 observations this vectorized calculation
    # is inexpensive.
    distances = (
        (pop_scaled[:, None, :] - X_scaled[None, :, :]) ** 2
    ).sum(axis=2)

    return np.sqrt(distances.min(axis=1))


# ============================================================
# 10. NSGA-II CORE
# ============================================================

def dominates(a, b):
    return (
        np.all(a <= b)
        and np.any(a < b)
    )


def fast_non_dominated_sort(objectives):
    n = len(objectives)

    domination_count = np.zeros(n, dtype=int)
    dominated = [[] for _ in range(n)]

    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue

            if dominates(objectives[p], objectives[q]):
                dominated[p].append(q)

            elif dominates(objectives[q], objectives[p]):
                domination_count[p] += 1

        if domination_count[p] == 0:
            fronts[0].append(p)

    front_index = 0

    while front_index < len(fronts):
        next_front = []

        for p in fronts[front_index]:
            for q in dominated[p]:
                domination_count[q] -= 1

                if domination_count[q] == 0:
                    next_front.append(q)

        if next_front:
            fronts.append(next_front)

        front_index += 1

    ranks = np.zeros(n, dtype=int)

    for rank, front in enumerate(fronts):
        for i in front:
            ranks[i] = rank

    return fronts, ranks


def crowding_distance(front, objectives):
    if len(front) == 0:
        return {}

    if len(front) <= 2:
        return {i: np.inf for i in front}

    distance = {i: 0.0 for i in front}

    m = objectives.shape[1]

    for obj_idx in range(m):
        sorted_front = sorted(
            front,
            key=lambda i: objectives[i, obj_idx]
        )

        distance[sorted_front[0]] = np.inf
        distance[sorted_front[-1]] = np.inf

        lo = objectives[sorted_front[0], obj_idx]
        hi = objectives[sorted_front[-1], obj_idx]

        if hi == lo:
            continue

        for j in range(1, len(sorted_front) - 1):
            prev_val = objectives[
                sorted_front[j - 1],
                obj_idx
            ]

            next_val = objectives[
                sorted_front[j + 1],
                obj_idx
            ]

            distance[sorted_front[j]] += (
                next_val - prev_val
            ) / (hi - lo)

    return distance


def rank_population(objectives):
    fronts, ranks = fast_non_dominated_sort(objectives)

    crowding = np.zeros(len(objectives), dtype=float)

    for front in fronts:
        d = crowding_distance(front, objectives)

        for i, value in d.items():
            crowding[i] = value

    return fronts, ranks, crowding


def tournament_select(population, ranks, crowding):
    a, b = np.random.randint(
        0,
        len(population),
        size=2,
    )

    if ranks[a] < ranks[b]:
        return population[a].copy()

    if ranks[b] < ranks[a]:
        return population[b].copy()

    if crowding[a] > crowding[b]:
        return population[a].copy()

    return population[b].copy()


# ============================================================
# 11. GENETIC OPERATORS
# ============================================================

def crossover(parent1, parent2):
    child1 = parent1.copy()
    child2 = parent2.copy()

    if np.random.random() > CROSSOVER_RATE:
        return child1, child2

    for i, feature in enumerate(FEATURE_COLUMNS):

        if feature in BINARY_FEATURES:
            if np.random.random() < 0.5:
                child1[i] = parent2[i]
                child2[i] = parent1[i]

        else:
            alpha = np.random.random()

            child1[i] = (
                alpha * parent1[i]
                + (1.0 - alpha) * parent2[i]
            )

            child2[i] = (
                (1.0 - alpha) * parent1[i]
                + alpha * parent2[i]
            )

    return child1, child2


def mutate(solution):
    child = solution.copy()

    for i, feature in enumerate(FEATURE_COLUMNS):

        if np.random.random() > MUTATION_RATE:
            continue

        if feature in BINARY_FEATURES:
            child[i] = 1.0 - child[i]

        else:
            width = UPPER_BOUNDS[i] - LOWER_BOUNDS[i]

            if width > 0:
                child[i] += np.random.normal(
                    loc=0.0,
                    scale=0.05 * width,
                )

    return repair_population(
        child.reshape(1, -1)
    )[0]


def create_initial_population():
    population = np.random.uniform(
        LOWER_BOUNDS,
        UPPER_BOUNDS,
        size=(POPULATION_SIZE, len(FEATURE_COLUMNS)),
    )

    return repair_population(population)


# ============================================================
# 12. SPACING METRIC
# ============================================================

def calculate_spacing(pareto_objectives):
    """
    Spacing = standard deviation of each solution's nearest-neighbour
    distance after objective normalization.

    Lower values indicate a more uniform Pareto distribution.
    """
    if len(pareto_objectives) < 2:
        return 0.0

    mins = pareto_objectives.min(axis=0)
    maxs = pareto_objectives.max(axis=0)

    ranges = maxs - mins
    ranges[ranges == 0] = 1.0

    normalized = (
        pareto_objectives - mins
    ) / ranges

    diff = (
        normalized[:, None, :]
        - normalized[None, :, :]
    )

    distances = np.sqrt(
        np.sum(diff ** 2, axis=2)
    )

    np.fill_diagonal(
        distances,
        np.inf,
    )

    nearest = distances.min(axis=1)

    if len(nearest) < 2:
        return 0.0

    return float(np.std(nearest, ddof=1))


# ============================================================
# 13. NSGA-II MAIN LOOP
# ============================================================

def run_nsga2():

    print("\n" + "=" * 82)
    print("STARTING FAST NSGA-II")
    print("=" * 82)

    population = create_initial_population()

    spacing_history = []
    best_conversion_history = []
    best_yield_history = []
    pareto_size_history = []

    start_time = time.time()

    final_objectives = None
    final_predictions = None
    final_population = None
    final_front = None

    for generation in range(MAX_GENERATIONS):

        # ----------------------------------------------------
        # Evaluate current population IN BATCH
        # ----------------------------------------------------
        objectives, predictions, population = (
            predict_population(population)
        )

        fronts, ranks, crowding = rank_population(
            objectives
        )

        first_front = fronts[0]

        spacing = calculate_spacing(
            objectives[first_front]
        )

        best_conversion = float(
            np.max(predictions[:, 0])
        )

        best_yield = float(
            np.max(predictions[:, 1])
        )

        spacing_history.append(spacing)
        best_conversion_history.append(best_conversion)
        best_yield_history.append(best_yield)
        pareto_size_history.append(len(first_front))

        print(
            f"Generation {generation + 1:03d}/{MAX_GENERATIONS} | "
            f"Pareto={len(first_front):03d} | "
            f"Best conversion={best_conversion:.2f}% | "
            f"Best yield={best_yield:.2f}% | "
            f"Spacing={spacing:.6f}"
        )

        # ----------------------------------------------------
        # Create offspring
        # ----------------------------------------------------
        offspring = []

        while len(offspring) < POPULATION_SIZE:

            parent1 = tournament_select(
                population,
                ranks,
                crowding,
            )

            parent2 = tournament_select(
                population,
                ranks,
                crowding,
            )

            child1, child2 = crossover(
                parent1,
                parent2,
            )

            child1 = mutate(child1)
            child2 = mutate(child2)

            offspring.append(child1)

            if len(offspring) < POPULATION_SIZE:
                offspring.append(child2)

        offspring = np.asarray(offspring)

        # ----------------------------------------------------
        # Evaluate offspring IN BATCH
        # ----------------------------------------------------
        offspring_objectives, offspring_predictions, offspring = (
            predict_population(offspring)
        )

        # ----------------------------------------------------
        # NSGA-II environmental selection
        # ----------------------------------------------------
        combined_population = np.vstack([
            population,
            offspring,
        ])

        combined_objectives = np.vstack([
            objectives,
            offspring_objectives,
        ])

        combined_predictions = np.vstack([
            predictions,
            offspring_predictions,
        ])

        combined_fronts, _, _ = rank_population(
            combined_objectives
        )

        selected = []

        for front in combined_fronts:

            if len(selected) + len(front) <= POPULATION_SIZE:
                selected.extend(front)

            else:
                remaining = POPULATION_SIZE - len(selected)

                d = crowding_distance(
                    front,
                    combined_objectives,
                )

                ordered = sorted(
                    front,
                    key=lambda i: d[i],
                    reverse=True,
                )

                selected.extend(
                    ordered[:remaining]
                )

                break

        selected = np.asarray(selected, dtype=int)

        population = combined_population[selected]

        # Keep latest evaluated state for final reporting.
        final_objectives = combined_objectives[selected]
        final_predictions = combined_predictions[selected]
        final_population = population.copy()

        final_fronts, _, _ = rank_population(
            final_objectives
        )
        final_front = final_fronts[0]

    elapsed = time.time() - start_time

    print("\nNSGA-II completed.")
    print(f"Runtime: {elapsed / 60.0:.2f} minutes")

    return {
        "population": final_population,
        "objectives": final_objectives,
        "predictions": final_predictions,
        "front": final_front,
        "spacing_history": spacing_history,
        "best_conversion_history": best_conversion_history,
        "best_yield_history": best_yield_history,
        "pareto_size_history": pareto_size_history,
        "runtime_seconds": elapsed,
    }


# ============================================================
# 14. PARETO TABLE
# ============================================================

def build_pareto_table(results):

    population = results["population"]
    objectives = results["objectives"]
    predictions = results["predictions"]
    front = results["front"]

    rows = []

    for rank_in_front, idx in enumerate(front, start=1):

        row = {
            "Pareto_Rank": rank_in_front,
            "Predicted_Conversion_pct": predictions[idx, 0],
            "Predicted_FDCA_Yield_pct": predictions[idx, 1],
            "Temperature_C": predictions[idx, 2],
            "Potential_V": predictions[idx, 3],
            "Nearest_Training_Distance": nearest_training_distance(
                population[idx].reshape(1, -1)
            )[0],
        }

        for j, feature in enumerate(FEATURE_COLUMNS):
            row[feature] = population[idx, j]

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 15. COMPROMISE RANKING
# ============================================================

def minmax(x):
    x = np.asarray(x, dtype=float)

    lo = np.min(x)
    hi = np.max(x)

    if hi == lo:
        return np.ones_like(x)

    return (x - lo) / (hi - lo)


def rank_compromise(pareto_df):

    result = pareto_df.copy()

    # Higher conversion/yield = better.
    conversion_score = minmax(
        result["Predicted_Conversion_pct"].values
    )

    yield_score = minmax(
        result["Predicted_FDCA_Yield_pct"].values
    )

    # Lower temperature/potential = better.
    temperature_score = 1.0 - minmax(
        result["Temperature_C"].values
    )

    potential_score = 1.0 - minmax(
        result["Potential_V"].values
    )

    result["Compromise_Score"] = (
        COMPROMISE_WEIGHTS["Conversion"]
        * conversion_score
        +
        COMPROMISE_WEIGHTS["FDCA_Yield"]
        * yield_score
        +
        COMPROMISE_WEIGHTS["Temperature"]
        * temperature_score
        +
        COMPROMISE_WEIGHTS["Potential"]
        * potential_score
    )

    result = result.sort_values(
        "Compromise_Score",
        ascending=False,
    ).reset_index(drop=True)

    result.insert(
        0,
        "Compromise_Rank",
        np.arange(1, len(result) + 1),
    )

    return result


# ============================================================
# 16. PLOTS
# ============================================================

def save_plots(results, pareto_df):

    # --------------------------------------------------------
    # Pareto conversion vs yield
    # --------------------------------------------------------
    plt.figure(figsize=(8, 6))

    plt.scatter(
        pareto_df["Predicted_Conversion_pct"],
        pareto_df["Predicted_FDCA_Yield_pct"],
        s=45,
    )

    plt.xlabel("Predicted HMF Conversion (%)")
    plt.ylabel("Predicted FDCA Yield (%)")
    plt.title("NSGA-II Pareto Front: HMF Conversion vs FDCA Yield")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "NSGAII_Pareto_Front.png",
        ),
        dpi=300,
    )

    plt.close()

    # --------------------------------------------------------
    # Spacing trajectory
    # --------------------------------------------------------
    plt.figure(figsize=(8, 6))

    plt.plot(
        np.arange(
            1,
            MAX_GENERATIONS + 1,
        ),
        results["spacing_history"],
        linewidth=2,
    )

    plt.xlabel("Generation")
    plt.ylabel("Spacing metric")
    plt.title("NSGA-II Spacing Trajectory")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "NSGAII_Spacing_Trajectory.png",
        ),
        dpi=300,
    )

    plt.close()

    # --------------------------------------------------------
    # Convergence trajectory
    # --------------------------------------------------------
    fig, ax1 = plt.subplots(
        figsize=(8, 6)
    )

    generations = np.arange(
        1,
        MAX_GENERATIONS + 1,
    )

    ax1.plot(
        generations,
        results["best_conversion_history"],
        linewidth=2,
        label="Best conversion",
    )

    ax1.plot(
        generations,
        results["best_yield_history"],
        linewidth=2,
        label="Best FDCA yield",
    )

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Predicted performance (%)")
    ax1.set_title("NSGA-II Performance Convergence")
    ax1.grid(alpha=0.25)
    ax1.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "NSGAII_Convergence.png",
        ),
        dpi=300,
    )

    plt.close()


# ============================================================
# 17. SAVE SUMMARY
# ============================================================

def save_summary(results, ranked_df):

    best = ranked_df.iloc[0]

    summary_path = os.path.join(
        OUTPUT_DIR,
        "NSGAII_Run_Summary.txt",
    )

    with open(summary_path, "w", encoding="utf-8") as f:

        f.write("FAST INTEGRATED NSGA-II RUN SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write(
            "Dataset: "
            + DATA_FILE
            + "\n"
        )

        f.write(
            f"Complete Conversion/Yield observations: {len(model_df)}\n"
        )

        f.write(
            f"Number of GA features: {len(FEATURE_COLUMNS)}\n"
        )

        f.write(
            f"Population size: {POPULATION_SIZE}\n"
        )

        f.write(
            f"Generations: {MAX_GENERATIONS}\n"
        )

        f.write(
            f"Crossover rate: {CROSSOVER_RATE}\n"
        )

        f.write(
            f"Mutation rate: {MUTATION_RATE}\n"
        )

        f.write(
            f"RF trees: {RF_TREES}\n"
        )

        f.write(
            f"Runtime (minutes): "
            f"{results['runtime_seconds'] / 60.0:.3f}\n"
        )

        f.write(
            f"Final Pareto-front size: "
            f"{len(results['front'])}\n\n"
        )

        f.write("SURROGATE VALIDATION\n")
        f.write(
            f"Conversion RMSE: {conv_rmse:.6f}\n"
        )
        f.write(
            f"Conversion R2: {conv_r2:.6f}\n"
        )
        f.write(
            f"Yield RMSE: {yield_rmse:.6f}\n"
        )
        f.write(
            f"Yield R2: {yield_r2:.6f}\n\n"
        )

        f.write(
            "BEST-COMPROMISE SOLUTION "
            "(post-hoc ranking only)\n"
        )

        for col in [
            "Compromise_Rank",
            "Compromise_Score",
            "Predicted_Conversion_pct",
            "Predicted_FDCA_Yield_pct",
            "Temperature_C",
            "Potential_V",
            "Nearest_Training_Distance",
        ]:
            f.write(
                f"{col}: {best[col]}\n"
            )

        f.write("\nCOMPROMISE WEIGHTS\n")

        for key, value in COMPROMISE_WEIGHTS.items():
            f.write(
                f"{key}: {value}\n"
            )

        f.write("\nOBJECTIVE NOTE\n")
        f.write(
            "This electrochemical adaptation optimizes conversion, "
            "FDCA yield, temperature and electrode potential. "
            "Pressure and reaction time were not invented because "
            "they are not explicit variables in the user's dataset.\n"
        )

        f.write(
            "\nEXTRAPOLATION NOTE\n"
            "Nearest_Training_Distance is a diagnostic. "
            "High values indicate that a GA solution is farther "
            "from the experimental feature space and should be "
            "treated cautiously during experimental validation.\n"
        )


# ============================================================
# 18. MAIN
# ============================================================

if __name__ == "__main__":

    results = run_nsga2()

    pareto_df = build_pareto_table(
        results
    )

    ranked_df = rank_compromise(
        pareto_df
    )

    best_df = ranked_df.iloc[
        [0]
    ].copy()

    # --------------------------------------------------------
    # CSV outputs
    # --------------------------------------------------------
    pareto_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "Pareto_Electrochemical_HMF_V2.csv",
        ),
        index=False,
    )

    ranked_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "Ranked_Pareto_Electrochemical_HMF_V2.csv",
        ),
        index=False,
    )

    best_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "Best_Compromise_Electrochemical_HMF_V2.csv",
        ),
        index=False,
    )

    # Also save the full final population for auditing.
    full_population = pd.DataFrame(
        results["population"],
        columns=FEATURE_COLUMNS,
    )

    full_population["Predicted_Conversion_pct"] = (
        results["predictions"][:, 0]
    )

    full_population["Predicted_FDCA_Yield_pct"] = (
        results["predictions"][:, 1]
    )

    full_population["Temperature_C"] = (
        results["predictions"][:, 2]
    )

    full_population["Potential_V"] = (
        results["predictions"][:, 3]
    )

    full_population["Is_Final_Pareto"] = False
    full_population.loc[
        results["front"],
        "Is_Final_Pareto"
    ] = True

    full_population.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "Final_NSGAII_Population_V2.csv",
        ),
        index=False,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    save_plots(
        results,
        pareto_df,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    save_summary(
        results,
        ranked_df,
    )

    print("\n" + "=" * 82)
    print("NSGA-II FINISHED SUCCESSFULLY")
    print("=" * 82)

    print(
        "\nFinal Pareto solutions:",
        len(pareto_df),
    )

    print("\nBest-compromise solution:")

    print(
        ranked_df[
            [
                "Compromise_Rank",
                "Compromise_Score",
                "Predicted_Conversion_pct",
                "Predicted_FDCA_Yield_pct",
                "Temperature_C",
                "Potential_V",
                "Nearest_Training_Distance",
            ]
        ].head(1).to_string(index=False)
    )

    print("\nOutputs saved to:")
    print(OUTPUT_DIR)

    print("\nIMPORTANT:")
    print(
        "The compromise score is a post-hoc ranking of the Pareto "
        "front. It is NOT an NSGA-II objective."
    )
