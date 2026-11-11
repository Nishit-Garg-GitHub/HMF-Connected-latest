"""Stage 08 — validation-gated multiobjective catalyst/condition optimisation.

NSGA-II is executed only if stage 07 shows that the single four-output model
beats the median baseline on every target under publication-grouped CV. The
four predicted responses are maximised jointly. Temperature and potential are
kept within observed ranges and treated as operating-burden objectives. The
algorithm mutates only within the empirical descriptor envelope; its outputs
are computational candidates, never experimentally confirmed results.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from common import (
    RESULTS_DIR, TARGETS, build_feature_matrix, read_processed, stage_status,
    write_json,
)

SEED = 42
POPULATION_SIZE = 48
GENERATIONS = 35
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.12


def load_gate() -> dict:
    """Load the regression gate; missing evidence means optimisation is blocked."""
    path = RESULTS_DIR / "regression_validation_gate.json"
    if not path.exists():
        return {"passed": False, "reason": "Regression validation gate is missing."}
    return json.loads(path.read_text(encoding="utf-8"))


def objective_matrix(model, population: np.ndarray, temp_idx: int | None,
                     potential_idx: int | None) -> np.ndarray:
    """Return minimisation objectives: negative performance plus conditions."""
    predictions = np.clip(model.predict(population), 0, 100)
    columns = [-predictions[:, i] for i in range(4)]
    if temp_idx is not None:
        columns.append(population[:, temp_idx])
    if potential_idx is not None:
        columns.append(population[:, potential_idx])
    return np.column_stack(columns)


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True when a is no worse in all objectives and better in at least one."""
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """Fast enough O(N²) Pareto sorting for the small screening population."""
    count = len(objectives)
    dominated = [[] for _ in range(count)]
    domination_count = np.zeros(count, dtype=int)
    fronts: list[list[int]] = [[]]
    for p in range(count):
        for q in range(count):
            if p == q:
                continue
            if dominates(objectives[p], objectives[q]):
                dominated[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)
    current = 0
    while current < len(fronts) and fronts[current]:
        following = []
        for p in fronts[current]:
            for q in dominated[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    following.append(q)
        if following:
            fronts.append(following)
        current += 1
    return fronts


def crowding_distance(front: list[int], objectives: np.ndarray) -> dict[int, float]:
    """Calculate NSGA-II crowding distance within one Pareto front."""
    distance = {index: 0.0 for index in front}
    if len(front) <= 2:
        return {index: float("inf") for index in front}
    values = objectives[front]
    for objective in range(values.shape[1]):
        order = np.argsort(values[:, objective])
        low = values[order[0], objective]
        high = values[order[-1], objective]
        distance[front[order[0]]] = float("inf")
        distance[front[order[-1]]] = float("inf")
        if np.isclose(high, low):
            continue
        for position in range(1, len(order) - 1):
            previous = values[order[position - 1], objective]
            following = values[order[position + 1], objective]
            index = front[order[position]]
            if np.isfinite(distance[index]):
                distance[index] += (following - previous) / (high - low)
    return distance


def rank_and_crowding(objectives: np.ndarray):
    """Return rank and diversity scores for every candidate."""
    ranks = np.full(len(objectives), 10**6, dtype=int)
    crowding = np.zeros(len(objectives), dtype=float)
    for rank, front in enumerate(non_dominated_sort(objectives)):
        distances = crowding_distance(front, objectives)
        for index in front:
            ranks[index] = rank
            crowding[index] = distances[index]
    return ranks, crowding


def repair(candidate: np.ndarray, lower: np.ndarray, upper: np.ndarray,
           binary_indices: list[int], categorical_groups: list[list[int]] | None = None) -> np.ndarray:
    """Clip descriptors, restore binary flags and keep one-hot groups valid."""
    candidate = np.nan_to_num(candidate, nan=0.0, posinf=0.0, neginf=0.0)
    candidate = np.clip(candidate, lower, upper)
    if binary_indices:
        candidate[binary_indices] = (candidate[binary_indices] >= 0.5).astype(float)
    # One-hot groups (electrolyte cation/anion, reactant) must not become
    # impossible multi-category combinations after crossover and mutation.
    for group in categorical_groups or []:
        if len(group) > 1:
            values = candidate[group].copy()
            candidate[group] = 0.0
            if np.max(values) >= 0.5:
                candidate[group[int(np.argmax(values))]] = 1.0
    return candidate


def tournament(population: np.ndarray, ranks: np.ndarray,
               crowding: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Binary tournament using Pareto rank then crowding distance."""
    a, b = rng.integers(0, len(population), size=2)
    winner = a if (ranks[a] < ranks[b] or
                   (ranks[a] == ranks[b] and crowding[a] > crowding[b])) else b
    return population[winner].copy()


def crossover_mutate(parent_a: np.ndarray, parent_b: np.ndarray,
                     lower: np.ndarray, upper: np.ndarray,
                     binary_indices: list[int], rng: np.random.Generator,
                     categorical_groups: list[list[int]] | None = None):
    """Blend parents and mutate a small subset of descriptors."""
    if rng.random() < CROSSOVER_RATE:
        alpha = rng.random(len(parent_a))
        child = alpha * parent_a + (1 - alpha) * parent_b
    else:
        child = parent_a.copy()
    for index in range(len(child)):
        if rng.random() < MUTATION_RATE:
            span = upper[index] - lower[index]
            child[index] += rng.normal(0, 0.08 * span if span else 0.0)
    return repair(child, lower, upper, binary_indices, categorical_groups)


def environmental_selection(population: np.ndarray, objectives: np.ndarray,
                            size: int):
    """Select the next generation by fronts and descending crowding distance."""
    chosen = []
    for front in non_dominated_sort(objectives):
        if len(chosen) + len(front) <= size:
            chosen.extend(front)
        else:
            distances = crowding_distance(front, objectives)
            ordered = sorted(front, key=lambda i: distances[i], reverse=True)
            chosen.extend(ordered[:size - len(chosen)])
            break
    return population[chosen]


def run_nsga(model, X: pd.DataFrame, feature_names: list[str]):
    """Run a compact NSGA-II search seeded by real observed experiments."""
    rng = np.random.default_rng(SEED)
    # Use the fitted training imputer to initialise complete candidate vectors.
    # The model pipeline accepts raw features, so these imputed values are
    # passed through its imputer a second time without changing their values.
    imputer = model.named_steps.get("imputer")
    matrix = X.to_numpy(dtype=float)
    matrix = imputer.transform(matrix) if imputer is not None else matrix
    lower = np.nanmin(matrix, axis=0)
    upper = np.nanmax(matrix, axis=0)
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix still contains non-finite values after imputation.")
    binary_indices = [
        i for i, name in enumerate(feature_names)
        if X[name].dropna().nunique() <= 2
    ]
    temperature = next((i for i, name in enumerate(feature_names)
                        if name.lower() == "temperature_c"), None)
    potential = next((i for i, name in enumerate(feature_names)
                      if name.lower() == "potential_vs_rhe_v"), None)
    categorical_groups = []
    for prefix in (
        "Electrolyte_Cation_", "Electrolyte_Anion_",
        "Electrolyte_Electrolyte_Cation_", "Electrolyte_Electrolyte_Anion_",
        "Reactant_",
    ):
        group = [i for i, name in enumerate(feature_names) if name.startswith(prefix)]
        if len(group) > 1:
            categorical_groups.append(group)
    # Seed with actual experimental rows, not arbitrary synthetic chemistry.
    if len(matrix) < 2:
        raise ValueError("NSGA-II needs at least two observed experiments.")
    initial_ids = rng.choice(len(matrix), size=POPULATION_SIZE, replace=len(matrix) < POPULATION_SIZE)
    population = matrix[initial_ids].copy()
    population = np.vstack([
        repair(row, lower, upper, binary_indices, categorical_groups)
        for row in population
    ])
    history = []
    for generation in range(GENERATIONS):
        objectives = objective_matrix(model, population, temperature, potential)
        ranks, crowding = rank_and_crowding(objectives)
        children = []
        while len(children) < POPULATION_SIZE:
            parent_a = tournament(population, ranks, crowding, rng)
            parent_b = tournament(population, ranks, crowding, rng)
            children.append(crossover_mutate(
                parent_a, parent_b, lower, upper, binary_indices, rng,
                categorical_groups,
            ))
        combined = np.vstack([population, np.asarray(children)])
        combined_objectives = objective_matrix(model, combined, temperature, potential)
        population = environmental_selection(combined, combined_objectives, POPULATION_SIZE)
        current = objective_matrix(model, population, temperature, potential)
        front = non_dominated_sort(current)[0]
        history.append({
            "generation": generation + 1,
            "pareto_count": len(front),
            "best_conversion": float(-np.min(current[front, 0])),
            "best_yield": float(-np.min(current[front, 1])),
            "best_fe": float(-np.min(current[front, 2])),
            "best_selectivity": float(-np.min(current[front, 3])),
        })
        print(f"NSGA-II generation {generation + 1}/{GENERATIONS}: Pareto={len(front)}")
    final_objectives = objective_matrix(model, population, temperature, potential)
    pareto_indices = non_dominated_sort(final_objectives)[0]
    return population[pareto_indices], final_objectives[pareto_indices], history


def main() -> None:
    """Respect regression validation gate before any optimisation is attempted."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    gate = load_gate()
    if not gate.get("passed", False):
        message = gate.get("reason", "Multi-output regression did not beat baseline on all four targets.")
        write_json(RESULTS_DIR / "nsga2_status.json", {
            "status": "blocked", "reason": message,
            "required_gate": "MAE must beat median baseline for Conversion, Yield, FE and Selectivity.",
        })
        stage_status("08_nsga2", "blocked", {"reason": message})
        print("NSGA-II BLOCKED:", message)
        return
    model_path = RESULTS_DIR / gate["model_path"]
    if not model_path.exists():
        raise FileNotFoundError(f"Validated regression model missing: {model_path}")
    model = load(model_path)
    frame = read_processed()
    X, features, _ = build_feature_matrix(frame)
    if list(gate.get("features", [])) != list(features):
        raise RuntimeError(
            "Feature contract mismatch between validated regression model and current preprocessing. "
            "Re-run stage 07 before NSGA-II."
        )
    pareto_X, pareto_objectives, history = run_nsga(model, X, features)
    # Objectives are stored in minimisation form; first four columns are -predicted metrics.
    output = pd.DataFrame(pareto_X, columns=features)
    for i, target in enumerate(TARGETS):
        output[f"predicted_{target}"] = -pareto_objectives[:, i]
    if pareto_objectives.shape[1] > 4:
        output["temperature_objective"] = pareto_objectives[:, 4]
    if pareto_objectives.shape[1] > 5:
        output["potential_objective"] = pareto_objectives[:, 5]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(RESULTS_DIR / "nsga2_pareto_candidates.csv", index=False)
    pd.DataFrame(history).to_csv(RESULTS_DIR / "nsga2_generation_history.csv", index=False)
    write_json(RESULTS_DIR / "nsga2_status.json", {
        "status": "completed_model_generated_candidates",
        "n_pareto_candidates": int(len(output)), "generations": GENERATIONS,
        "population_size": POPULATION_SIZE,
        "warning": "Candidates require experimental confirmation; model predictions are not measurements.",
    })
    stage_status("08_nsga2", "completed", {"n_pareto_candidates": int(len(output))})


if __name__ == "__main__":
    main()
