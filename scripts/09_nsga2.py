"""Stage 08 — validation-gated NSGA-II catalyst/condition optimisation.

The design follows the integrated genetic algorithm of the reference article
(Li et al., ACS Sustainable Chem. Eng. 2025, 13, 13940; sections 2.6 / 3.6),
adapted from thermal aerobic oxidation to HMF electro-oxidation:

* Mixed encoding. Discrete descriptors (metals, catalyst families, cell type,
  electrolyte ions) are binary {0, 1}; one-hot groups carry the hard
  constraint sum(x_i) = 1. Continuous descriptors are min-max normalised to
  [0, 1] for the genetic operators and decoded before prediction.
* Objectives. The article maximises conversion X and yield Y while
  minimising temperature T, |P - 1 atm| and reaction time. This dataset has no
  pressure or time, so the electrochemical operating burdens are temperature
  T and applied potential E (V vs RHE); faradaic efficiency is added as a
  third performance objective. Predictions come from the per-target
  regressors that passed the stage 07 validation gate.
* NSGA-II. Fast non-dominated sorting + crowding distance, binary
  tournament, blend crossover and Gaussian mutation, (mu + lambda) survival.
* Convergence. The spacing metric S = sqrt(1/|H| sum (d_mean - d_i)^2) of the
  first front is recorded every generation (article Figure 14).
* Decision. Pareto solutions are ranked by a composite fitness whose weights
  come from the entropy weight method:
      fitness = w1 X + w2 Y + w3 FE - w4 T - w5 E   (on min-max scaled values)

Two searches are run: an unconstrained search over all catalysts and a
constrained search with a single-metal Ni(OH)2 or Cu(OH)2 anode. All outputs
are model-generated hypotheses, not experimental results.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from common import (
    RESULTS_DIR, FIGURES_DIR, TARGETS, build_feature_matrix, read_processed,
    read_cleaned, stage_status, write_json, mutual_info_scored,  # noqa: F401 (unpickling)
)

SEED = 42
POPULATION_SIZE = 100
GENERATIONS = 1000
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.10
TOP_N = 10

# Objective columns (all maximised quantities are negated -> minimisation).
OBJECTIVES = ["Conversion_pct", "Yield_pct", "FE_pct", "Temperature_C", "Potential_vs_RHE_V"]
BENEFIT = {"Conversion_pct": True, "Yield_pct": True, "FE_pct": True,
           "Temperature_C": False, "Potential_vs_RHE_V": False}
ONE_HOT_PREFIXES = ("Electrolyte_Electrolyte_Cation_", "Electrolyte_Electrolyte_Anion_", "Reactant_")


# ---------------------------------------------------------------------------
# Validation gate and models
# ---------------------------------------------------------------------------

def load_gate() -> dict:
    """Load the regression gate; missing evidence means optimisation is blocked."""
    path = RESULTS_DIR / "regression_validation_gate.json"
    if not path.exists():
        return {"passed": False, "reason": "Regression validation gate is missing."}
    return json.loads(path.read_text(encoding="utf-8"))


def load_regressors(gate: dict):
    """Per-target regressors preferred; fall back to the joint model."""
    root_dir = Path(__file__).resolve().parents[1]
    if gate.get("per_target_model_paths"):
        models = {}
        for target, relpath in gate["per_target_model_paths"].items():
            path = root_dir / relpath
            if not path.exists():
                raise FileNotFoundError(f"Missing per-target regressor for {target}: {path}")
            models[target] = load(path)
        return models
    key = "joint_model_path" if "joint_model_path" in gate else "model_path"
    path = root_dir / gate[key]
    if not path.exists():
        raise FileNotFoundError(f"Validated regression model missing: {path}")
    return load(path)


def predict_targets(models, real: np.ndarray) -> np.ndarray:
    """Predicted Conversion, Yield, FE, Selectivity (clipped to 0-100)."""
    if isinstance(models, dict):
        return np.column_stack([np.clip(models[t].predict(real), 0, 100) for t in TARGETS])
    return np.clip(models.predict(real), 0, 100)


def model_imputer(models):
    first = next(iter(models.values())) if isinstance(models, dict) else models
    return first.named_steps.get("imputer") if hasattr(first, "named_steps") else None


# ---------------------------------------------------------------------------
# Mixed encoding (article: one-hot {0,1} with sum = 1; continuous -> [0, 1])
# ---------------------------------------------------------------------------

class MixedEncoding:
    """Map between real descriptor vectors and the GA's [0, 1] genome."""

    def __init__(self, X: pd.DataFrame, feature_names: list[str], real_matrix: np.ndarray):
        self.names = list(feature_names)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.lower = np.nanmin(real_matrix, axis=0)
        self.upper = np.nanmax(real_matrix, axis=0)
        self.span = np.where(self.upper > self.lower, self.upper - self.lower, 1.0)
        self.binary = [i for i, n in enumerate(self.names) if X[n].dropna().nunique() <= 2]
        self.groups = []
        for prefix in ONE_HOT_PREFIXES:
            group = [i for i, n in enumerate(self.names) if n.startswith(prefix)]
            if len(group) > 1:
                # "_nan" (not reported) is never a valid design choice.
                choosable = [i for i in group if not self.names[i].endswith("_nan")]
                self.groups.append((group, choosable or group))
        self.constraint = None  # optional real-space repair (chemistry rules)

    def set_bounds(self, name: str, lo: float, hi: float) -> None:
        """Narrow a continuous variable's feasible interval (hard constraint)."""
        i = self.idx[name]
        self.lower[i] = max(self.lower[i], lo)
        self.upper[i] = min(self.upper[i], hi)
        self.span[i] = self.upper[i] - self.lower[i] if self.upper[i] > self.lower[i] else 1.0

    def encode(self, real: np.ndarray) -> np.ndarray:
        return np.clip((real - self.lower) / self.span, 0.0, 1.0)

    def decode(self, genome: np.ndarray) -> np.ndarray:
        """Genome -> feasible real vector (binary rounding, sum(one-hot) = 1)."""
        genome = np.clip(np.nan_to_num(genome, nan=0.0), 0.0, 1.0)
        real = self.lower + genome * self.span
        lo_b, hi_b = self.lower[self.binary], self.upper[self.binary]
        real[self.binary] = np.where(genome[self.binary] >= 0.5, hi_b, lo_b)
        for group, choosable in self.groups:
            winner = choosable[int(np.argmax(genome[choosable]))]
            real[group] = 0.0
            real[winner] = 1.0
        if self.constraint is not None:
            real = self.constraint(real)
        return np.clip(real, self.lower, self.upper)

    def canonical(self, genome: np.ndarray) -> np.ndarray:
        """Re-encode the decoded vector so the genome reflects what was evaluated."""
        return self.encode(self.decode(genome))


# ---------------------------------------------------------------------------
# NSGA-II machinery
# ---------------------------------------------------------------------------

def objective_values(models, enc: MixedEncoding, real: np.ndarray):
    """Return (minimisation objectives, predicted targets) for real vectors."""
    pred = predict_targets(models, real)
    t = real[:, enc.idx["Temperature_C"]]
    e = real[:, enc.idx["Potential_vs_RHE_V"]]
    objectives = np.column_stack([-pred[:, 0], -pred[:, 1], -pred[:, 2], t, e])
    return objectives, pred


def non_dominated_sort(F: np.ndarray) -> list[np.ndarray]:
    """Fast non-dominated sorting (vectorised dominance matrix)."""
    le = (F[:, None, :] <= F[None, :, :]).all(axis=2)
    lt = (F[:, None, :] < F[None, :, :]).any(axis=2)
    dom = le & lt                          # dom[i, j]: i dominates j
    count = dom.sum(axis=0)                # how many dominate j
    fronts = []
    remaining = np.ones(len(F), dtype=bool)
    current = np.flatnonzero(count == 0)
    while current.size:
        fronts.append(current)
        remaining[current] = False
        count = count - dom[current].sum(axis=0)
        current = np.flatnonzero((count == 0) & remaining)
    return fronts


def crowding_distance(F: np.ndarray) -> np.ndarray:
    n, m = F.shape
    if n <= 2:
        return np.full(n, np.inf)
    dist = np.zeros(n)
    for j in range(m):
        order = np.argsort(F[:, j])
        lo, hi = F[order[0], j], F[order[-1], j]
        dist[order[0]] = dist[order[-1]] = np.inf
        if hi > lo:
            dist[order[1:-1]] += (F[order[2:], j] - F[order[:-2], j]) / (hi - lo)
    return dist


def rank_and_crowding(F: np.ndarray):
    rank = np.zeros(len(F), dtype=int)
    crowd = np.zeros(len(F))
    for r, front in enumerate(non_dominated_sort(F)):
        rank[front] = r
        crowd[front] = crowding_distance(F[front])
    return rank, crowd


def normalised_objectives(F: np.ndarray, enc: "MixedEncoding") -> np.ndarray:
    """Fixed scaling where 0 is each objective's ideal (article section 3.6).

    Performance: (100 - value) / 100. Temperature and potential: position
    within their feasible interval (lower bound = 0).
    """
    Z = np.empty_like(F, dtype=float)
    Z[:, :3] = (100.0 + F[:, :3]) / 100.0
    for col, name in ((3, "Temperature_C"), (4, "Potential_vs_RHE_V")):
        i = enc.idx[name]
        Z[:, col] = (F[:, col] - enc.lower[i]) / enc.span[i]
    return Z


def spacing_metric(F: np.ndarray, enc: "MixedEncoding") -> float:
    """Schott spacing S = sqrt(1/|H| sum (d_mean - d_i)^2) of a front (eq 12)."""
    if len(F) < 2:
        return 0.0
    Z = normalised_objectives(F, enc)
    d = np.abs(Z[:, None, :] - Z[None, :, :]).sum(axis=2)
    np.fill_diagonal(d, np.inf)
    di = d.min(axis=1)
    return float(np.sqrt(np.mean((di.mean() - di) ** 2)))


def nsga2(models, enc: MixedEncoding, seed_rows: np.ndarray, label: str, rng):
    """Run NSGA-II; returns final genomes, objectives, predictions, history."""
    pick = rng.choice(len(seed_rows), size=POPULATION_SIZE, replace=len(seed_rows) < POPULATION_SIZE)
    pop = np.vstack([enc.canonical(enc.encode(r)) for r in seed_rows[pick]])
    real = np.vstack([enc.decode(g) for g in pop])
    F, P = objective_values(models, enc, real)
    n_var = pop.shape[1]
    history = []
    for gen in range(1, GENERATIONS + 1):
        rank, crowd = rank_and_crowding(F)

        def tournament():
            a, b = rng.integers(0, len(pop), size=2)
            better = rank[a] < rank[b] or (rank[a] == rank[b] and crowd[a] > crowd[b])
            return pop[a if better else b]

        children = []
        while len(children) < POPULATION_SIZE:
            pa, pb = tournament(), tournament()
            if rng.random() < CROSSOVER_RATE:
                alpha = rng.random(n_var)
                child = alpha * pa + (1 - alpha) * pb
            else:
                child = pa.copy()
            mask = rng.random(n_var) < MUTATION_RATE
            child[mask] += rng.normal(0, 0.1, size=mask.sum())
            children.append(enc.canonical(child))
        children = np.vstack(children)
        child_real = np.vstack([enc.decode(g) for g in children])
        Fc, Pc = objective_values(models, enc, child_real)

        allpop = np.vstack([pop, children])
        allF = np.vstack([F, Fc])
        allP = np.vstack([P, Pc])
        chosen = []
        for front in non_dominated_sort(allF):
            if len(chosen) + len(front) <= POPULATION_SIZE:
                chosen.extend(front.tolist())
            else:
                cd = crowding_distance(allF[front])
                chosen.extend(front[np.argsort(-cd)][:POPULATION_SIZE - len(chosen)].tolist())
                break
        pop, F, P = allpop[chosen], allF[chosen], allP[chosen]
        first = non_dominated_sort(F)[0]
        history.append({
            "scenario": label, "generation": gen, "pareto_count": int(len(first)),
            "spacing": spacing_metric(F[first], enc),
            "best_conversion": float(-F[first, 0].min()),
            "best_yield": float(-F[first, 1].min()),
            "best_fe": float(-F[first, 2].min()),
        })
        if gen % 100 == 0 or gen == 1:
            h = history[-1]
            print(f"[{label}] generation {gen}/{GENERATIONS}: Pareto={h['pareto_count']} "
                  f"spacing={h['spacing']:.4f} best Y={h['best_yield']:.2f}")
    return pop, F, P, history


# ---------------------------------------------------------------------------
# Entropy weight method and decision
# ---------------------------------------------------------------------------

def entropy_weights(values: pd.DataFrame) -> dict[str, float]:
    """Entropy weight method over the Pareto decision matrix."""
    n = len(values)
    scaled = pd.DataFrame(index=values.index)
    for col in values:
        lo, hi = values[col].min(), values[col].max()
        span = hi - lo if hi > lo else 1.0
        scaled[col] = (values[col] - lo) / span if BENEFIT[col] else (hi - values[col]) / span
    p = (scaled + 1e-12) / (scaled + 1e-12).sum(axis=0)
    e = -(p * np.log(p)).sum(axis=0) / np.log(max(n, 2))
    d = 1 - e
    w = d / d.sum() if d.sum() > 0 else pd.Series(1 / len(d), index=d.index)
    return {k: float(v) for k, v in w.items()}


def composite_fitness(values: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """fitness = sum_j w_j r_j with benefit/cost min-max scaling (article eq 11)."""
    score = pd.Series(0.0, index=values.index)
    for col, w in weights.items():
        lo, hi = values[col].min(), values[col].max()
        span = hi - lo if hi > lo else 1.0
        r = (values[col] - lo) / span if BENEFIT[col] else (hi - values[col]) / span
        score += w * r
    return score


def decision_table(enc, pop, F, P) -> tuple[pd.DataFrame, dict]:
    """Decode, rank by Pareto front, weight by entropy, de-duplicate."""
    real = np.vstack([enc.decode(g) for g in pop])
    table = pd.DataFrame(real, columns=enc.names)
    for i, t in enumerate(TARGETS):
        table[f"predicted_{t}"] = P[:, i]
    rank, _ = rank_and_crowding(F)
    table["pareto_rank"] = rank
    key = table[enc.names].round(3).astype(str).agg("|".join, axis=1)
    table = table.loc[~key.duplicated()].reset_index(drop=True)
    decision = pd.DataFrame({
        "Conversion_pct": table["predicted_Conversion_pct"],
        "Yield_pct": table["predicted_Yield_pct"],
        "FE_pct": table["predicted_FE_pct"],
        "Temperature_C": table["Temperature_C"],
        "Potential_vs_RHE_V": table["Potential_vs_RHE_V"],
    })
    front = table["pareto_rank"].eq(0)
    weights = entropy_weights(decision[front]) if front.sum() >= 2 else {c: 0.2 for c in OBJECTIVES}
    table["entropy_fitness"] = composite_fitness(decision, weights)
    table = table.sort_values(["pareto_rank", "entropy_fitness"], ascending=[True, False])
    return table.reset_index(drop=True), weights


# ---------------------------------------------------------------------------
# Chemistry constraint: single-metal Ni(OH)2 / Cu(OH)2 anode
# ---------------------------------------------------------------------------

HYDROXIDE_METALS = ("Ni", "Cu")
FREE_CATALYST_FLAGS = {"Catalyst_Carbon", "Catalyst_Nanoparticle",
                       "Catalyst_Foam_Substrate", "Catalyst_Conductive_Glass"}


def hydroxide_constraint(enc: MixedEncoding, metal: str):
    """Catalyst_<metal> = Catalyst_Hydroxide = 1, all other catalyst flags 0;
    derived descriptors recomputed; KOH/NaOH pH tied to concentration."""
    idx = enc.idx
    flags = [i for n, i in idx.items() if n.startswith("Catalyst_") and n not in FREE_CATALYST_FLAGS]

    def constrain(real: np.ndarray) -> np.ndarray:
        real = real.copy()
        real[flags] = 0.0
        real[idx[f"Catalyst_{metal}"]] = 1.0
        real[idx["Catalyst_Hydroxide"]] = 1.0
        if "Catalyst_N_Elements" in idx:
            real[idx["Catalyst_N_Elements"]] = 1.0
        if "Potential_vs_RHE" in idx:
            real[idx["Potential_vs_RHE"]] = 1.0
        oh = idx.get("Electrolyte_Electrolyte_Anion_OH")
        if oh is not None and real[oh] >= 0.5:
            if "Alkaline_Electrolyte" in idx:
                real[idx["Alkaline_Electrolyte"]] = 1.0
            if "Electrolyte_pH" in idx:
                conc = max(real[idx["Electrolyte_Concentration_M"]], 1e-4)
                real[idx["Electrolyte_pH"]] = 14 + np.log10(conc)
        return real

    return constrain


def general_constraint(enc: MixedEncoding):
    """Keep the potential flag consistent for the unconstrained search."""
    idx = enc.idx

    def constrain(real: np.ndarray) -> np.ndarray:
        real = real.copy()
        if "Potential_vs_RHE" in idx:
            real[idx["Potential_vs_RHE"]] = 1.0
        return real

    return constrain


def hydroxide_envelope(raw: pd.DataFrame, mask: np.ndarray) -> dict:
    envelope = {}
    for col in ("Potential_vs_RHE_V", "Temperature_C"):
        vals = pd.to_numeric(raw.loc[mask, col], errors="coerce").dropna()
        if len(vals) >= 2:
            envelope[col] = (float(vals.min()), float(vals.max()))
    return envelope


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _one_hot_label(row: pd.Series, prefix: str) -> str:
    hits = [c[len(prefix):] for c in row.index if c.startswith(prefix) and row[c] >= 0.5]
    return hits[0] if hits else "nan"


def describe(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    elements = ("Ni", "Co", "Fe", "Cu", "Pt", "Pd", "Au", "Ru", "Ir", "Rh", "Mn", "Mo",
                "W", "V", "Bi", "Ce", "Zn", "Ag")
    metals = [f"Catalyst_{e}" for e in elements if f"Catalyst_{e}" in frame]
    out["metals"] = frame[metals].apply(
        lambda r: "+".join(c.split("_")[1] for c in metals if r[c] >= 0.5) or "none", axis=1)
    out["catalyst"] = np.where(
        frame.get("Catalyst_Hydroxide", 0) >= 0.5,
        out["metals"].where(out["metals"].str.contains(r"\+"), out["metals"] + "(OH)2"),
        out["metals"])
    out["support"] = np.where(frame.get("Catalyst_Foam_Substrate", 0) >= 0.5, "foam",
                     np.where(frame.get("Catalyst_Conductive_Glass", 0) >= 0.5, "FTO/ITO/BDD",
                     np.where(frame.get("Catalyst_Carbon", 0) >= 0.5, "carbon", "unspecified")))
    cat = frame.apply(_one_hot_label, axis=1, prefix="Electrolyte_Electrolyte_Cation_")
    an = frame.apply(_one_hot_label, axis=1, prefix="Electrolyte_Electrolyte_Anion_")
    out["electrolyte"] = [f"{c}{a}" if "nan" not in (c, a) else "unspecified" for c, a in zip(cat, an)]
    out["electrolyte_M"] = frame["Electrolyte_Concentration_M"].round(2)
    out["electrolyte_pH"] = frame["Electrolyte_pH"].round(2)
    out["HMF_mM"] = frame["HMF_Concentration_mM"].round(1)
    out["potential_V_RHE"] = frame["Potential_vs_RHE_V"].round(3)
    out["temperature_C"] = frame["Temperature_C"].round(1)
    out["TEMPO"] = frame["TEMPO_binary"].round().astype(int)
    out["cell"] = np.where(frame.get("Flow_Cell", 0) >= 0.5, "flow",
                  np.where(frame.get("Divided_Cell", 0) >= 0.5, "divided (H-cell)",
                  np.where(frame.get("Undivided_Cell", 0) >= 0.5, "undivided", "unspecified")))
    return out


def nearest_observed(top: pd.DataFrame, matrix: np.ndarray, names, raw, mask) -> pd.DataFrame:
    mu, sd = matrix.mean(axis=0), matrix.std(axis=0)
    sd[sd == 0] = 1.0
    ref = (matrix[mask] - mu) / sd
    rows = np.flatnonzero(mask)
    out = []
    for _, cand in top.iterrows():
        z = (cand[names].to_numpy(dtype=float) - mu) / sd
        d = np.sqrt(((ref - z) ** 2).sum(axis=1))
        j = rows[int(np.argmin(d))]
        out.append({
            "nearest_record_row": int(j), "nearest_distance": round(float(d.min()), 3),
            "nearest_catalyst": str(raw.loc[j, "Catalyst_Anode_Text"]).strip(" ,|;"),
            "nearest_paper": str(raw.loc[j, "Paper Title"])[:110],
            "nearest_conditions": f"{raw.loc[j, 'Electrolyte']} | {raw.loc[j, 'Potential_raw']}",
            **{f"observed_{t}": raw.loc[j, t] for t in TARGETS},
        })
    return pd.DataFrame(out)


def high_performance_probability(X: pd.DataFrame, raw: pd.DataFrame, candidates: pd.DataFrame,
                                 names: list[str]) -> np.ndarray | None:
    """P(class >=90%) from the stage 04 selected classifier, refit on all rows."""
    path = RESULTS_DIR / "best_classification_candidate.json"
    if not path.exists():
        return None
    best = json.loads(path.read_text())
    if best.get("target_kind") != "binary_high_performance_90":
        return None
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
    from common import classification_preprocessing_steps, make_classifier
    fdca = raw["Major_Product_Clean"].astype(str).str.contains(
        r"FDCA|FURANDICARBOXYLIC", case=False, regex=True, na=False)
    yld = pd.to_numeric(raw["Yield_pct"], errors="coerce")
    conv = pd.to_numeric(raw["Conversion_pct"], errors="coerce")
    rows = np.flatnonzero((fdca & yld.notna()).to_numpy())
    yl, cv = yld.iloc[rows].clip(0, 100).to_numpy(), conv.iloc[rows].clip(0, 100).to_numpy()
    y = (np.where(np.isfinite(cv), np.minimum(cv, yl), yl) >= 90).astype(int)
    sampler = {"ROS": RandomOverSampler(random_state=SEED),
               "SMOTE": SMOTE(k_neighbors=2, random_state=SEED),
               "ADASYN": ADASYN(n_neighbors=2, random_state=SEED)}.get(best["sampler"])
    steps = classification_preprocessing_steps()
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", make_classifier(best["model"], SEED)))
    pipe = ImbPipeline(steps).fit(X.iloc[rows], y)
    return pipe.predict_proba(candidates[names])[:, 1]


def plot_spacing(history: pd.DataFrame, path: Path) -> None:
    """Article Figure 14 analogue: spacing trace per hydroxide sub-run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = [("NiOH2", "(a)", "Ni(OH)$_2$"), ("CuOH2", "(b)", "Cu(OH)$_2$")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (label, letter, title) in zip(axes, runs):
        h = history[history["scenario"] == label]
        ax.plot(h["generation"], h["spacing"], color="#1f3fbf", linewidth=1.2, label="Spacing")
        ax.set_ylim(0, float(h["spacing"].max()) * 1.15)
        ax.set_title(f"Spacing Trace Plot — {title} anode")
        ax.set_xlabel("Generation Number")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=13, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_top(top: pd.DataFrame, labels: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sc = ax.scatter(top["predicted_FE_pct"], top["predicted_Yield_pct"], c=top["entropy_fitness"],
                    cmap="viridis", s=95, edgecolors="black", linewidths=0.6,
                    marker="o")
    # Several solutions share predicted values; stack their rank labels.
    groups = top.groupby([top["predicted_FE_pct"].round(2), top["predicted_Yield_pct"].round(2)])
    for (fe, yv), g in groups:
        ax.annotate(", ".join(str(i + 1) for i in g.index), (fe, yv),
                    textcoords="offset points", xytext=(7, 4), fontsize=8)
    ax.text(0.02, 0.02, "All ten solutions: " + ", ".join(sorted(labels["catalyst"].unique())),
            transform=ax.transAxes, fontsize=8, color="0.3")
    fig.colorbar(sc, ax=ax, label="Entropy-weighted fitness")
    ax.set_xlabel("Predicted faradaic efficiency (%)")
    ax.set_ylabel("Predicted FDCA yield (%)")
    ax.set_title(f"NSGA-II top {len(top)} Ni(OH)$_2$ / Cu(OH)$_2$ solutions")
    ax.grid(alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Respect the regression validation gate before any optimisation."""
    warnings.filterwarnings("ignore")
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

    models = load_regressors(gate)
    frame = read_processed()
    X, features, _ = build_feature_matrix(frame)
    if list(gate.get("features", [])) != list(features):
        raise RuntimeError("Feature contract mismatch with the validated regressors; re-run stage 07.")
    imputer = model_imputer(models)
    matrix = X.to_numpy(dtype=float)
    matrix = imputer.transform(matrix) if imputer is not None else matrix
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix still contains non-finite values after imputation.")
    raw = read_cleaned()
    rng = np.random.default_rng(SEED)
    history = []

    # ---- 1. Unconstrained search over all catalysts --------------------------
    enc = MixedEncoding(X, features, matrix)
    enc.constraint = general_constraint(enc)
    pop, F, P, hist = nsga2(models, enc, matrix, "general", rng)
    history += hist
    general, general_weights = decision_table(enc, pop, F, P)
    general = pd.concat([describe(general), general], axis=1)
    general[general["pareto_rank"] == 0].to_csv(RESULTS_DIR / "nsga2_pareto_candidates.csv", index=False)

    # ---- 2. Constrained search: Ni(OH)2 / Cu(OH)2 anode ----------------------
    hyd_mask = ((raw["Catalyst_Hydroxide"] == 1)
                & ((raw["Catalyst_Ni"] == 1) | (raw["Catalyst_Cu"] == 1))).to_numpy()
    envelope = hydroxide_envelope(raw, hyd_mask)
    print(f"Hydroxide condition envelope: {envelope}")
    pools = []
    for metal in HYDROXIDE_METALS:
        enc_m = MixedEncoding(X, features, matrix)
        for name, (lo, hi) in envelope.items():
            enc_m.set_bounds(name, lo, hi)
        enc_m.constraint = hydroxide_constraint(enc_m, metal)
        seeds = matrix[hyd_mask & (raw[f"Catalyst_{metal}"] == 1).to_numpy()]
        seeds = np.vstack([seeds, matrix[rng.choice(len(matrix), 40, replace=False)]])
        pop_m, F_m, P_m, hist_m = nsga2(models, enc_m, seeds, f"{metal}OH2", rng)
        history += hist_m
        pools.append((enc_m, pop_m))
    # Pool both sub-runs in real space and re-rank jointly.
    real = np.vstack([np.vstack([e.decode(g) for g in p]) for e, p in pools])
    enc_pool = MixedEncoding(X, features, matrix)
    for name, (lo, hi) in envelope.items():
        enc_pool.set_bounds(name, lo, hi)
    genomes = np.vstack([enc_pool.encode(r) for r in real])
    F_pool, P_pool = objective_values(models, enc_pool, real)
    table, weights = decision_table(enc_pool, genomes, F_pool, P_pool)
    # Pooled rows are already feasible, so re-decoding them is lossless
    # (binary/one-hot values stay exactly 0/1).
    labels = describe(table)
    p_high = high_performance_probability(X, raw, table, features)
    if p_high is not None:
        table["classifier_P_high90"] = p_high

    top = table.head(TOP_N).reset_index(drop=True)
    top_labels = labels.head(TOP_N).reset_index(drop=True)
    cu = table[table["Catalyst_Cu"] >= 0.5].head(3).reset_index(drop=True)
    pred_cols = [f"predicted_{t}" for t in TARGETS] + ["entropy_fitness", "pareto_rank"]
    if p_high is not None:
        pred_cols.append("classifier_P_high90")

    def report(rows: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([
            pd.Series(range(1, len(rows) + 1), name="rank"),
            lab.reset_index(drop=True),
            rows[pred_cols].round(3).reset_index(drop=True),
            nearest_observed(rows, matrix, features, raw, hyd_mask),
        ], axis=1)

    top_report = report(top, top_labels)
    cu_report = report(cu, describe(cu))
    top_report.to_csv(RESULTS_DIR / "nsga2_top10_ni_cu_hydroxide.csv", index=False)
    top.to_csv(RESULTS_DIR / "nsga2_top10_ni_cu_hydroxide_descriptors.csv", index=False)
    cu_report.to_csv(RESULTS_DIR / "nsga2_best3_cu_hydroxide.csv", index=False)
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(RESULTS_DIR / "nsga2_generation_history.csv", index=False)
    write_json(RESULTS_DIR / "nsga2_entropy_weights.json",
               {"general": general_weights, "hydroxide": weights})
    plot_spacing(hist_df, FIGURES_DIR / "nsga2" / "Figure_14_Spacing_Trace.png")
    plot_top(top, top_labels, FIGURES_DIR / "nsga2" / "NSGA2_top10_NiCu_hydroxide.png")

    def converged_at(label: str) -> int | None:
        """First generation whose 20-generation rolling median spacing is
        within 1.5x of the final plateau (median of the last 100)."""
        h = hist_df[hist_df["scenario"] == label]["spacing"]
        plateau = float(h.tail(100).median())
        rolling = h.rolling(20, min_periods=20).median().to_numpy()
        hits = np.flatnonzero(rolling <= 1.5 * plateau)
        return int(hits[0] + 1) if hits.size else None

    status = {
        "status": "completed_model_generated_candidates",
        "algorithm": "NSGA-II, mixed encoding, entropy-weighted decision (Li et al. 2025 design)",
        "objectives": {"maximise": ["Conversion", "FDCA yield", "FE"],
                       "minimise": ["Temperature", "Potential vs RHE"]},
        "generations": GENERATIONS, "population_size": POPULATION_SIZE,
        "n_pareto_candidates": int((general["pareto_rank"] == 0).sum()),
        "general_entropy_weights": general_weights,
        "hydroxide_scenario": {
            "condition_envelope": envelope,
            "entropy_weights": weights,
            "top10_catalysts": top_labels["catalyst"].value_counts().to_dict(),
            "spacing_converged_generation": {m: converged_at(f"{m}OH2") for m in HYDROXIDE_METALS},
        },
        "warning": "Candidates require experimental confirmation; model predictions are not measurements.",
    }
    write_json(RESULTS_DIR / "nsga2_status.json", status)
    stage_status("08_nsga2", "completed", {"n_pareto_candidates": status["n_pareto_candidates"]})

    print("\nEntropy weights (hydroxide):", {k: round(v, 3) for k, v in weights.items()})
    print("\nTop Ni(OH)2 / Cu(OH)2 solutions:")
    show = ["rank", "catalyst", "support", "electrolyte", "electrolyte_M", "HMF_mM",
            "potential_V_RHE", "temperature_C", "predicted_Conversion_pct",
            "predicted_Yield_pct", "predicted_FE_pct", "entropy_fitness"]
    if p_high is not None:
        show.append("classifier_P_high90")
    print(top_report[show].to_string(index=False))
    print("\nBest Cu(OH)2 solutions:")
    print(cu_report[show].to_string(index=False))


if __name__ == "__main__":
    main()
