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
    write_json, mutual_info_scored,
)

SEED = 42
POPULATION_SIZE = 80
GENERATIONS = 60
TOP_N = 10
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.12


def load_gate() -> dict:
    """Load the regression gate; missing evidence means optimisation is blocked."""
    path = RESULTS_DIR / "regression_validation_gate.json"
    if not path.exists():
        return {"passed": False, "reason": "Regression validation gate is missing."}
    return json.loads(path.read_text(encoding="utf-8"))


def objective_matrix(models, population: np.ndarray, temp_idx: int | None,
                     potential_idx: int | None) -> np.ndarray:
    """Return minimisation objectives: negative performance plus conditions.

    `models` may be a single joint multi-output model or a dict of
    per-target models.  Per-target models are preferred because they use
    more training records.
    """
    if isinstance(models, dict):
        predictions = np.column_stack([
            np.clip(models[target].predict(population), 0, 100)
            for target in TARGETS
        ])
    else:
        predictions = np.clip(models.predict(population), 0, 100)
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
                     categorical_groups: list[list[int]] | None = None,
                     constraint=None):
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
    child = repair(child, lower, upper, binary_indices, categorical_groups)
    return constraint(child) if constraint is not None else child


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


def run_nsga(models, X: pd.DataFrame, feature_names: list[str],
             constraint_factory=None, seed_mask: np.ndarray | None = None,
             return_population: bool = False):
    """Run a compact NSGA-II search seeded by real observed experiments.

    ``constraint_factory(feature_names, lower, upper)`` may return a function
    that maps every candidate onto an allowed chemistry (e.g. Ni or Cu
    hydroxide anodes). ``seed_mask`` prefers matching observed rows when
    building the initial population.
    """
    rng = np.random.default_rng(SEED)
    # Use the fitted training imputer to initialise complete candidate vectors.
    # Per-target models each have their own imputer; use the first one available.
    if isinstance(models, dict):
        first_model = next(iter(models.values()))
        imputer = first_model.named_steps.get("imputer") if hasattr(first_model, "named_steps") else None
    else:
        imputer = models.named_steps.get("imputer") if hasattr(models, "named_steps") else None
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
    constraint = (constraint_factory(feature_names, lower, upper)
                  if constraint_factory is not None else None)
    # Seed with actual experimental rows, not arbitrary synthetic chemistry.
    if len(matrix) < 2:
        raise ValueError("NSGA-II needs at least two observed experiments.")
    pool = np.arange(len(matrix))
    if seed_mask is not None and seed_mask.sum() >= 2:
        preferred = pool[seed_mask]
        n_pref = min(POPULATION_SIZE // 2, len(preferred) * 3)
        initial_ids = np.concatenate([
            rng.choice(preferred, size=n_pref, replace=True),
            rng.choice(pool, size=POPULATION_SIZE - n_pref, replace=False),
        ])
    else:
        initial_ids = rng.choice(len(matrix), size=POPULATION_SIZE,
                                 replace=len(matrix) < POPULATION_SIZE)
    population = matrix[initial_ids].copy()
    population = np.vstack([
        repair(row, lower, upper, binary_indices, categorical_groups)
        for row in population
    ])
    if constraint is not None:
        population = np.vstack([constraint(row) for row in population])
    history = []
    for generation in range(GENERATIONS):
        objectives = objective_matrix(models, population, temperature, potential)
        ranks, crowding = rank_and_crowding(objectives)
        children = []
        while len(children) < POPULATION_SIZE:
            parent_a = tournament(population, ranks, crowding, rng)
            parent_b = tournament(population, ranks, crowding, rng)
            children.append(crossover_mutate(
                parent_a, parent_b, lower, upper, binary_indices, rng,
                categorical_groups, constraint,
            ))
        combined = np.vstack([population, np.asarray(children)])
        combined_objectives = objective_matrix(models, combined, temperature, potential)
        population = environmental_selection(combined, combined_objectives, POPULATION_SIZE)
        current = objective_matrix(models, population, temperature, potential)
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
    final_objectives = objective_matrix(models, population, temperature, potential)
    pareto_indices = non_dominated_sort(final_objectives)[0]
    if return_population:
        return population, final_objectives, history, matrix
    return population[pareto_indices], final_objectives[pareto_indices], history


# ---------------------------------------------------------------------------
# Constrained scenario: Ni(OH)2 or Cu(OH)2 anode
# ---------------------------------------------------------------------------

HYDROXIDE_METALS = ("Ni", "Cu")  # one sub-run per metal
# Descriptors the optimiser may still choose for a hydroxide anode.
FREE_CATALYST_FLAGS = {"Catalyst_Carbon", "Catalyst_Nanoparticle",
                       "Catalyst_Foam_Substrate", "Catalyst_Conductive_Glass"}


def make_hydroxide_constraint(metal: str, envelope: dict[str, tuple[float, float]]):
    """Build a constraint factory forcing a single-metal M(OH)2 anode.

    Catalyst_<metal> and Catalyst_Hydroxide are 1, every other element and
    catalyst-family flag is 0, and derived descriptors (element count, noble,
    bimetallic) are recomputed. ``envelope`` narrows continuous conditions
    (potential, temperature) to the range reported for real Ni/Cu hydroxide
    anodes, so the optimiser cannot extrapolate to e.g. 0.6 V vs RHE where
    these catalysts are not active. For hydroxide electrolytes the pH is tied
    to the concentration (pH = 14 + log10[OH-]); potential is vs RHE.
    """
    def factory(feature_names, lower, upper):
        return _hydroxide_constraint(feature_names, lower, upper, metal, envelope)
    return factory


def _hydroxide_constraint(feature_names, lower, upper, metal, envelope):
    idx = {name: i for i, name in enumerate(feature_names)}
    catalyst_flags = [i for name, i in idx.items()
                      if name.startswith("Catalyst_") and name not in FREE_CATALYST_FLAGS]
    lower = lower.copy()
    upper = upper.copy()
    for name, (lo, hi) in envelope.items():
        if name in idx:
            lower[idx[name]] = max(lower[idx[name]], lo)
            upper[idx[name]] = min(upper[idx[name]], hi)

    def constrain(candidate: np.ndarray) -> np.ndarray:
        candidate = candidate.copy()
        chosen = idx[f"Catalyst_{metal}"]
        candidate[catalyst_flags] = 0.0
        candidate[chosen] = 1.0
        if "Catalyst_Hydroxide" in idx:
            candidate[idx["Catalyst_Hydroxide"]] = 1.0
        if "Catalyst_N_Elements" in idx:
            candidate[idx["Catalyst_N_Elements"]] = 1.0
        if "Potential_vs_RHE" in idx:
            candidate[idx["Potential_vs_RHE"]] = 1.0
        oh = idx.get("Electrolyte_Electrolyte_Anion_OH")
        conc = idx.get("Electrolyte_Concentration_M")
        if oh is not None and candidate[oh] >= 0.5 and conc is not None:
            if "Alkaline_Electrolyte" in idx:
                candidate[idx["Alkaline_Electrolyte"]] = 1.0
            if "Electrolyte_pH" in idx:
                ph = 14 + np.log10(max(candidate[conc], 1e-4))
                candidate[idx["Electrolyte_pH"]] = float(np.clip(
                    ph, lower[idx["Electrolyte_pH"]], upper[idx["Electrolyte_pH"]]))
        return np.clip(candidate, lower, upper)

    return constrain


def _one_hot_label(row: pd.Series, prefix: str) -> str:
    cols = [c for c in row.index if c.startswith(prefix)]
    hits = [c[len(prefix):] for c in cols if row[c] >= 0.5]
    return hits[0] if hits else "nan"


def describe_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Human-readable chemistry/conditions for each candidate row."""
    out = pd.DataFrame(index=frame.index)
    out["catalyst"] = np.where(frame.get("Catalyst_Cu", 0) >= 0.5, "Cu(OH)2", "Ni(OH)2")
    substrate = np.where(frame.get("Catalyst_Foam_Substrate", 0) >= 0.5, "foam",
                np.where(frame.get("Catalyst_Conductive_Glass", 0) >= 0.5, "FTO/ITO/BDD",
                np.where(frame.get("Catalyst_Carbon", 0) >= 0.5, "carbon", "unspecified")))
    out["support"] = substrate
    out["nanostructured"] = (frame.get("Catalyst_Nanoparticle", 0) >= 0.5).astype(int)
    cation = frame.apply(_one_hot_label, axis=1, prefix="Electrolyte_Electrolyte_Cation_")
    anion = frame.apply(_one_hot_label, axis=1, prefix="Electrolyte_Electrolyte_Anion_")
    out["electrolyte"] = [
        f"{c}{a}" if c != "nan" and a != "nan" else "unspecified"
        for c, a in zip(cation, anion)
    ]
    out["electrolyte_M"] = frame["Electrolyte_Concentration_M"].round(2)
    out["electrolyte_pH"] = frame["Electrolyte_pH"].round(2) if "Electrolyte_pH" in frame else np.nan
    out["HMF_mM"] = frame["HMF_Concentration_mM"].round(1)
    out["potential_V_RHE"] = frame["Potential_vs_RHE_V"].round(3)
    out["temperature_C"] = frame["Temperature_C"].round(1)
    out["TEMPO"] = frame["TEMPO_binary"].round().astype(int)
    cell = np.where(frame.get("Flow_Cell", 0) >= 0.5, "flow",
           np.where(frame.get("Divided_Cell", 0) >= 0.5, "divided (H-cell)",
           np.where(frame.get("Undivided_Cell", 0) >= 0.5, "undivided", "unspecified")))
    out["cell"] = cell
    return out


def select_top_candidates(population, objectives, feature_names, n=TOP_N):
    """Best n distinct candidates: by Pareto rank, then mean predicted metric.

    Candidates are de-duplicated on rounded descriptor values so the list is
    not ten copies of the same recipe. Lower potential breaks ties.
    """
    ranks, _ = rank_and_crowding(objectives)
    perf = -objectives[:, :4]
    table = pd.DataFrame(population, columns=feature_names)
    for i, target in enumerate(TARGETS):
        table[f"predicted_{target}"] = perf[:, i]
    table["predicted_mean_4"] = perf.mean(axis=1)
    table["pareto_rank"] = ranks
    key = table[feature_names].round(2).astype(str).agg("|".join, axis=1)
    table = table.loc[~key.duplicated()]
    table = table.sort_values(["pareto_rank", "predicted_mean_4", "Potential_vs_RHE_V"],
                              ascending=[True, False, True])
    return table.head(n).reset_index(drop=True)


def nearest_observed(top: pd.DataFrame, X_imputed: np.ndarray, feature_names,
                     raw: pd.DataFrame, candidate_mask: np.ndarray) -> pd.DataFrame:
    """Nearest real literature record (standardised descriptor distance)."""
    mu = X_imputed.mean(axis=0)
    sd = X_imputed.std(axis=0)
    sd[sd == 0] = 1.0
    reference = (X_imputed[candidate_mask] - mu) / sd
    ref_rows = np.flatnonzero(candidate_mask)
    records = []
    for _, cand in top.iterrows():
        z = (cand[feature_names].to_numpy(dtype=float) - mu) / sd
        d = np.sqrt(((reference - z) ** 2).sum(axis=1))
        j = ref_rows[int(np.argmin(d))]
        records.append({
            "nearest_record_row": int(j),
            "nearest_distance": float(d.min()),
            "nearest_catalyst": str(raw.loc[j, "Catalyst_Anode_Text"]).strip(" ,|;"),
            "nearest_paper": str(raw.loc[j, "Paper Title"])[:90],
            "nearest_conditions": f"{raw.loc[j, 'Electrolyte']} | {raw.loc[j, 'Potential_raw']}",
            **{f"observed_{t}": raw.loc[j, t] for t in TARGETS},
        })
    return pd.DataFrame(records)


def plot_top_candidates(top: pd.DataFrame, labels: pd.DataFrame, path: Path) -> None:
    """Predicted yield vs FE for the top candidates, coloured by potential."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    markers = {"Ni(OH)2": "o", "Cu(OH)2": "s"}
    sc = None
    for cat, marker in markers.items():
        m = labels["catalyst"].eq(cat).to_numpy()
        if m.any():
            sc = ax.scatter(top.loc[m, "predicted_FE_pct"], top.loc[m, "predicted_Yield_pct"],
                            c=top.loc[m, "Potential_vs_RHE_V"], cmap="viridis", marker=marker,
                            s=90, edgecolors="black", linewidths=0.6, label=cat,
                            vmin=top["Potential_vs_RHE_V"].min(), vmax=top["Potential_vs_RHE_V"].max())
    for i, row in top.iterrows():
        ax.annotate(str(i + 1), (row["predicted_FE_pct"], row["predicted_Yield_pct"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    if sc is not None:
        fig.colorbar(sc, ax=ax, label="Potential (V vs RHE)")
    ax.set_xlabel("Predicted faradaic efficiency (%)")
    ax.set_ylabel("Predicted FDCA yield (%)")
    ax.set_title(f"NSGA-II top {len(top)} Ni(OH)$_2$ / Cu(OH)$_2$ candidates")
    ax.grid(alpha=0.25)
    ax.legend(title="Anode")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def condition_envelope(raw: pd.DataFrame, mask: np.ndarray) -> dict:
    """Observed potential/temperature range for real Ni/Cu hydroxide anodes."""
    envelope = {}
    for column in ("Potential_vs_RHE_V", "Temperature_C"):
        values = pd.to_numeric(raw.loc[mask, column], errors="coerce").dropna()
        if len(values) >= 2:
            envelope[column] = (float(values.min()), float(values.max()))
    return envelope


def run_hydroxide_scenario(models, X: pd.DataFrame, features: list[str]) -> dict:
    """Constrained NSGA-II: Ni(OH)2 and Cu(OH)2 sub-runs, report the top 10.

    Each metal is optimised separately (so Cu(OH)2 is not crowded out early),
    the final populations are pooled and re-ranked by Pareto rank.
    """
    from common import read_cleaned, FIGURES_DIR
    raw = read_cleaned()
    hydroxide_rows = (
        (raw["Catalyst_Hydroxide"] == 1)
        & ((raw["Catalyst_Ni"] == 1) | (raw["Catalyst_Cu"] == 1))
    ).to_numpy()
    envelope = condition_envelope(raw, hydroxide_rows)
    print(f"Hydroxide condition envelope: {envelope}")
    populations, histories = [], []
    for metal in HYDROXIDE_METALS:
        population, _, history, matrix = run_nsga(
            models, X, features,
            constraint_factory=make_hydroxide_constraint(metal, envelope),
            seed_mask=hydroxide_rows & (raw[f"Catalyst_{metal}"] == 1).to_numpy(),
            return_population=True,
        )
        populations.append(population)
        histories.extend({**h, "metal": metal} for h in history)
    history = histories
    population = np.vstack(populations)
    temperature = features.index("Temperature_C") if "Temperature_C" in features else None
    potential = features.index("Potential_vs_RHE_V") if "Potential_vs_RHE_V" in features else None
    objectives = objective_matrix(models, population, temperature, potential)
    top = select_top_candidates(population, objectives, features)
    cu_mask = population[:, features.index("Catalyst_Cu")] >= 0.5
    best_cu = select_top_candidates(population[cu_mask], objectives[cu_mask], features, n=3)
    labels = describe_candidates(top)
    nearest = nearest_observed(top, matrix, features, raw, hydroxide_rows)
    report = pd.concat([
        pd.Series(range(1, len(top) + 1), name="rank"),
        labels.reset_index(drop=True),
        top[[f"predicted_{t}" for t in TARGETS] + ["predicted_mean_4", "pareto_rank"]].round(2),
        nearest,
    ], axis=1)
    report.to_csv(RESULTS_DIR / "nsga2_top10_ni_cu_hydroxide.csv", index=False)
    cu_report = pd.concat([
        describe_candidates(best_cu).reset_index(drop=True),
        best_cu[[f"predicted_{t}" for t in TARGETS] + ["predicted_mean_4"]].round(2),
        nearest_observed(best_cu, matrix, features, raw, hydroxide_rows),
    ], axis=1)
    cu_report.to_csv(RESULTS_DIR / "nsga2_best3_cu_hydroxide.csv", index=False)
    top.to_csv(RESULTS_DIR / "nsga2_top10_ni_cu_hydroxide_descriptors.csv", index=False)
    pd.DataFrame(history).to_csv(RESULTS_DIR / "nsga2_hydroxide_generation_history.csv", index=False)
    plot_top_candidates(top, labels, FIGURES_DIR / "nsga2" / "NSGA2_top10_NiCu_hydroxide.png")
    print("\nTop Ni(OH)2 / Cu(OH)2 candidates:")
    print(report[["rank", "catalyst", "support", "electrolyte", "electrolyte_M", "HMF_mM",
                  "potential_V_RHE", "temperature_C", "predicted_Conversion_pct",
                  "predicted_Yield_pct", "predicted_FE_pct", "predicted_Selectivity_pct"]]
          .to_string(index=False))
    print("\nBest Cu(OH)2 candidates:")
    print(cu_report[["catalyst", "support", "electrolyte", "electrolyte_M", "HMF_mM",
                     "potential_V_RHE", "temperature_C", "predicted_Conversion_pct",
                     "predicted_Yield_pct", "predicted_FE_pct", "predicted_Selectivity_pct"]]
          .to_string(index=False))
    return {
        "n_candidates": int(len(report)),
        "catalysts": report["catalyst"].value_counts().to_dict(),
        "observed_hydroxide_seed_rows": int(hydroxide_rows.sum()),
        "condition_envelope": envelope,
    }


def main() -> None:
    """Respect regression validation gate before any optimisation is attempted."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    gate = load_gate()
    if not gate.get("passed", False):
        message = gate.get("reason", "Regression did not beat baseline on all required targets.")
        write_json(RESULTS_DIR / "nsga2_status.json", {
            "status": "blocked", "reason": message,
            "required_gate": "MAE must beat median baseline for Conversion, Yield, FE and Selectivity.",
        })
        stage_status("08_nsga2", "blocked", {"reason": message})
        print("NSGA-II BLOCKED:", message)
        return

    # Load models: per-target models preferred, fallback to joint model
    root_dir = Path(__file__).resolve().parents[1]
    if "per_target_model_paths" in gate and gate["per_target_model_paths"]:
        models = {}
        for target, relpath in gate["per_target_model_paths"].items():
            full_path = root_dir / relpath
            if not full_path.exists():
                raise FileNotFoundError(f"Missing per-target regressor for {target}: {full_path}")
            models[target] = load(full_path)
    else:
        model_key = "joint_model_path" if "joint_model_path" in gate else "model_path"
        model_path = root_dir / gate[model_key]
        if not model_path.exists():
            raise FileNotFoundError(f"Validated regression model missing: {model_path}")
        models = load(model_path)

    frame = read_processed()
    X, features, _ = build_feature_matrix(frame)
    if list(gate.get("features", [])) != list(features):
        raise RuntimeError(
            "Feature contract mismatch between validated regression model and current preprocessing. "
            "Re-run stage 07 before NSGA-II."
        )
    pareto_X, pareto_objectives, history = run_nsga(models, X, features)
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
    hydroxide = run_hydroxide_scenario(models, X, features)
    write_json(RESULTS_DIR / "nsga2_status.json", {
        "status": "completed_model_generated_candidates",
        "n_pareto_candidates": int(len(output)), "generations": GENERATIONS,
        "population_size": POPULATION_SIZE,
        "hydroxide_scenario": hydroxide,
        "warning": "Candidates require experimental confirmation; model predictions are not measurements.",
    })
    stage_status("08_nsga2", "completed", {"n_pareto_candidates": int(len(output))})


if __name__ == "__main__":
    main()
