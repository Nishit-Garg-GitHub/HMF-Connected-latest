"""Stage 08 — SHAP explanation of the validated regressors and literature check.

For each outcome the stage explains the per-target regressor from stage 07
(mean of Huber gradient boosting and extra-trees; Tree SHAP is exact for both
and linear in the average). It reports

* global importance (mean |SHAP|) per descriptor and per input group,
* the direction of each effect (correlation between feature value and SHAP),
* stability: the same analysis repeated on paper-level bootstrap resamples,
  giving how often a descriptor is in the top five and how often its sign
  is the same, and
* a comparison with directions reported in the experimental literature.

SHAP values describe what the fitted model uses; they are associational and
are NOT causal catalyst effects. The literature comparison is deliberately
reported as consistent / opposite / not distinguishable, so a disagreement is
visible rather than tuned away.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, load

from common import (
    FIGURES_DIR, RESULTS_DIR, TARGETS, TARGET_LABELS, build_feature_matrix,
    make_regressor_single, paper_groups, read_processed, save_figure,
    stage_status, write_json, mutual_info_scored,  # noqa: F401 (unpickling)
)

warnings.filterwarnings("ignore")
SEED = 42
N_BOOT = 30
TOP_N_PLOT = 12
SHAP_DIR = FIGURES_DIR / "shap"

GROUPS = [
    ("Catalyst (anode)", lambda n: n.startswith("Catalyst_")),
    ("Electrolyte", lambda n: n.startswith("Electrolyte_") or n == "Alkaline_Electrolyte"),
    ("Potential", lambda n: n.startswith("Potential_")),
    ("Temperature", lambda n: n == "Temperature_C"),
    ("HMF concentration", lambda n: n == "HMF_Concentration_mM"),
    ("TEMPO mediator", lambda n: n == "TEMPO_binary"),
    ("Cell / counter electrode",
     lambda n: n.startswith("Counter_Electrode_") or n in {
         "Divided_Cell", "Undivided_Cell", "Flow_Cell", "Three_Electrode"}),
    ("Reactant", lambda n: n.startswith("Reactant_")),
]

# Literature expectations. Each entry lists the descriptors that represent the
# factor, the direction reported for FDCA yield / faradaic efficiency and the
# source. Directions come from the cited experimental papers (search results
# were abstracts and snippets: verify against full text before submission).
LITERATURE = [
    {"factor": "Ni-containing anode (Ni(OH)2/NiOOH)", "features": ["Catalyst_Ni"],
     "expected": "+", "claim": "Ni(OH)2/NiOOH anodes reach >95% FDCA yield and >90% FE in alkaline HMF "
     "oxidation (beta-Ni(OH)2: 98.7% yield, 98% FE).",
     "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9401862/"},
    {"factor": "Hydroxide / oxyhydroxide phase", "features": ["Catalyst_Hydroxide", "Catalyst_Oxyhydroxide"],
     "expected": "+", "claim": "Metal hydroxide/oxyhydroxide (NiOOH, CuOOH) is the electrochemically "
     "generated active phase.", "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9401862/"},
    {"factor": "Cu-containing anode (Cu(OH)2/CuOOH)", "features": ["Catalyst_Cu"],
     "expected": "+", "claim": "CuO/Cu(OH)2 on Cu foam: 99.6% FDCA yield and 99.3% FE at 1.62 V vs RHE; "
     "Cu-based anodes are active.", "source": "https://pubs.rsc.org/en/content/articlepdf/2023/gc/d3gc01420g"},
    {"factor": "Alkaline electrolyte", "features": ["Alkaline_Electrolyte", "Electrolyte_pH"],
     "expected": "+", "claim": "Ni/Cu hydroxide anodes need OH- to form the active oxyhydroxide; most "
     "high-performance reports use ~1 M KOH.", "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9401862/"},
    {"factor": "HMF concentration", "features": ["HMF_Concentration_mM"],
     "expected": "-", "targets": ("Yield_pct",), "claim": "HMF degrades to humins in alkaline media (lower yield at high HMF) and the "
     "benefit of higher HMF concentration saturates above ~0.3-0.4 M. Faradaic efficiency is not "
     "unambiguous, so only yield is compared.",
     "source": "https://pubs.rsc.org/en/content/articlehtml/2023/gc/d2gc04732b"},
]


def pretty(name: str) -> str:
    """Readable descriptor label."""
    rules = [
        ("Electrolyte_Electrolyte_Cation_", "cation: "), ("Electrolyte_Electrolyte_Anion_", "anion: "),
        ("Catalyst_", "anode: "), ("Counter_Electrode_", "counter electrode: "),
    ]
    for prefix, label in rules:
        if name.startswith(prefix):
            return (label + name[len(prefix):]).replace("_", " ").replace(": nan", ": not reported")
    return {
        "HMF_Concentration_mM": "HMF concentration (mM)", "Potential_vs_RHE_V": "potential (V vs RHE)",
        "Potential_vs_RHE": "potential reported vs RHE", "Temperature_C": "temperature (C)",
        "TEMPO_binary": "TEMPO mediator", "Electrolyte_Concentration_M": "electrolyte concentration (M)",
        "Electrolyte_pH": "electrolyte pH", "Alkaline_Electrolyte": "alkaline electrolyte",
        "Divided_Cell": "divided cell", "Undivided_Cell": "undivided cell", "Flow_Cell": "flow cell",
        "Three_Electrode": "three-electrode cell",
    }.get(name, name.replace("_", " "))


def group_of(name: str) -> str:
    for label, test in GROUPS:
        if test(name):
            return label
    return "Other"


def shap_matrix(pipe, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Tree SHAP averaged over the ensemble members; returns (values, imputed X)."""
    import shap
    Xi = pipe.named_steps["imputer"].transform(X)
    values = []
    for member in pipe.named_steps["model"].estimators_:
        values.append(np.asarray(shap.TreeExplainer(member).shap_values(Xi)))
    return np.mean(values, axis=0), Xi


def summarise(values: np.ndarray, Xi: np.ndarray, names: list[str]) -> pd.DataFrame:
    mean_abs = np.abs(values).mean(axis=0)
    direction = np.zeros(len(names))
    for j in range(len(names)):
        if np.std(Xi[:, j]) > 0 and np.std(values[:, j]) > 0:
            direction[j] = np.corrcoef(Xi[:, j], values[:, j])[0, 1]
    return pd.DataFrame({"feature": names, "label": [pretty(n) for n in names],
                         "group": [group_of(n) for n in names],
                         "mean_abs_shap": mean_abs, "direction": direction})


def _bootstrap_once(X, y, papers, names, seed):
    rng = np.random.default_rng(seed)
    unique = np.unique(papers)
    picked = rng.choice(unique, size=len(unique), replace=True)
    rows = np.concatenate([np.flatnonzero(papers == p) for p in picked])
    model = make_regressor_single(seed).fit(X.iloc[rows], y[rows])
    values, Xi = shap_matrix(model, X.iloc[rows])
    return summarise(values, Xi, names).set_index("feature")


def explain_target(target, frame, X_all, groups_all, model_path):
    v = pd.to_numeric(frame[target], errors="coerce")
    ids = np.flatnonzero((v.notna() & v.between(0, 100)).to_numpy())
    y = v.iloc[ids].to_numpy(float)
    X = X_all.iloc[ids].reset_index(drop=True)
    papers = groups_all[ids]
    names = list(X.columns)
    model = load(model_path)
    values, Xi = shap_matrix(model, X)
    table = summarise(values, Xi, names)

    boots = Parallel(n_jobs=4)(
        delayed(_bootstrap_once)(X, y, papers, names, SEED + b) for b in range(N_BOOT))
    imp = pd.concat([b["mean_abs_shap"] for b in boots], axis=1)
    dirs = pd.concat([b["direction"] for b in boots], axis=1)
    top5 = (imp.rank(ascending=False, method="first") <= 5).mean(axis=1)
    table["top5_freq"] = table["feature"].map(top5)
    table["sign_stable"] = table["feature"].map(
        (np.sign(dirs).mean(axis=1).abs()))          # 1 = same sign in every bootstrap
    table["boot_mean_abs_shap_lo"] = table["feature"].map(imp.quantile(0.05, axis=1))
    table["boot_mean_abs_shap_hi"] = table["feature"].map(imp.quantile(0.95, axis=1))
    table = table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return table, values, Xi, names, len(y), len(np.unique(papers))


def plot_bars(tables: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = {"Catalyst (anode)": "#c0392b", "Electrolyte": "#2980b9", "Potential": "#8e44ad",
              "Temperature": "#e67e22", "HMF concentration": "#16a085", "TEMPO mediator": "#7f8c8d",
              "Cell / counter electrode": "#34495e", "Reactant": "#95a5a6", "Other": "#bdc3c7"}
    for ax, target, letter in zip(axes, ("Yield_pct", "FE_pct"), "ab"):
        t = tables[target].head(TOP_N_PLOT)[::-1]
        ax.barh(t["label"], t["mean_abs_shap"], color=[colors[g] for g in t["group"]],
                xerr=[(t["mean_abs_shap"] - t["boot_mean_abs_shap_lo"]).clip(lower=0),
                      (t["boot_mean_abs_shap_hi"] - t["mean_abs_shap"]).clip(lower=0)],
                error_kw={"elinewidth": 0.8, "capsize": 2})
        ax.set_xlabel("Mean |SHAP| (percentage points)")
        ax.set_title(f"({letter}) {TARGET_LABELS[target]}", loc="left", fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    fig.legend(handles, colors.keys(), ncol=5, loc="lower center", fontsize=8, frameon=False)
    fig.suptitle("Regression SHAP importance (bars: 5-95% over paper bootstraps)", fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(SHAP_DIR / f"Regression_SHAP_Bar.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)


def plot_beeswarm(values, Xi, names, target) -> None:
    import shap
    order = np.argsort(-np.abs(values).mean(axis=0))[:TOP_N_PLOT]
    fig = plt.figure(figsize=(8, 5.5))
    shap.summary_plot(values[:, order], Xi[:, order], feature_names=[pretty(names[i]) for i in order],
                      show=False, plot_size=None, max_display=TOP_N_PLOT)
    plt.title(f"SHAP beeswarm — {TARGET_LABELS[target]}", fontweight="bold")
    plt.xlabel("SHAP value (percentage points)")
    for ext in ("png", "pdf"):
        plt.savefig(SHAP_DIR / f"Regression_SHAP_Beeswarm_{target}.{ext}",
                    dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_groups(tables: dict) -> pd.DataFrame:
    rows = []
    for target, t in tables.items():
        g = t.groupby("group")["mean_abs_shap"].sum()
        for name, val in g.items():
            rows.append({"target": target, "group": name, "share": val / g.sum()})
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="group", columns="target", values="share").fillna(0)
    pivot = pivot.reindex(columns=[t for t in TARGETS if t in pivot.columns])
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ypos = np.arange(len(pivot))
    width = 0.8 / len(pivot.columns)
    for i, target in enumerate(pivot.columns):
        ax.barh(ypos + i * width, pivot[target], height=width, label=TARGET_LABELS[target])
    ax.set_yticks(ypos + 0.4 - width / 2, pivot.index)
    ax.set_xlabel("Share of total mean |SHAP|")
    ax.set_title("Which input groups the models rely on", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(SHAP_DIR / f"Regression_SHAP_Groups.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    return df


def literature_table(tables: dict) -> pd.DataFrame:
    rows = []
    for entry in LITERATURE:
        for target in entry.get("targets", ("Yield_pct", "FE_pct")):
            t = tables[target].set_index("feature")
            present = [f for f in entry["features"] if f in t.index]
            if not present:
                rows.append({**{k: entry[k] for k in ("factor", "expected", "claim", "source")},
                             "target": target, "verdict": "not testable (no variation in data)"})
                continue
            best = max(present, key=lambda f: t.loc[f, "mean_abs_shap"])
            r = t.loc[best]
            rank = int((tables[target]["mean_abs_shap"] > r["mean_abs_shap"]).sum()) + 1
            data_dir = "+" if r["direction"] > 0.1 else "-" if r["direction"] < -0.1 else "0"
            # A-priori rule: the descriptor must be among the ten most important
            # AND either recur in the bootstrap top five or keep a stable sign.
            distinguishable = rank <= 10 and (r["top5_freq"] >= 0.2 or r["sign_stable"] >= 0.8)
            if data_dir == "0" or not distinguishable:
                verdict = "not distinguishable"
            elif data_dir == entry["expected"]:
                verdict = "consistent"
            else:
                verdict = "opposite"
            rows.append({**{k: entry[k] for k in ("factor", "expected", "claim", "source")},
                         "target": target, "feature": best, "data_direction": data_dir,
                         "corr_value_vs_shap": round(float(r["direction"]), 3),
                         "mean_abs_shap": round(float(r["mean_abs_shap"]), 3),
                         "importance_rank": rank, "of_features": len(t),
                         "top5_freq": round(float(r["top5_freq"]), 2),
                         "sign_stability": round(float(r["sign_stable"]), 2),
                         "verdict": verdict})
    return pd.DataFrame(rows)


def main() -> None:
    import json
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    gate = json.loads((RESULTS_DIR / "regression_validation_gate.json").read_text())
    frame = read_processed()
    X_all, features, _ = build_feature_matrix(frame)
    groups_all = paper_groups()
    root = RESULTS_DIR.parent

    tables, payload = {}, {}
    for target in TARGETS:
        print(f"SHAP for {target} ({N_BOOT} paper bootstraps) ...", flush=True)
        table, values, Xi, names, n, n_papers = explain_target(
            target, frame, X_all, groups_all, root / gate["per_target_model_paths"][target])
        table.to_csv(RESULTS_DIR / f"shap_regression_{target}.csv", index=False)
        tables[target] = table
        payload[target] = (values, Xi, names)
        print(table.head(8)[["label", "mean_abs_shap", "direction", "top5_freq"]].round(3).to_string(index=False))

    plot_bars(tables)
    plot_beeswarm(*payload["Yield_pct"], "Yield_pct")
    plot_beeswarm(*payload["FE_pct"], "FE_pct")
    group_df = plot_groups(tables)
    group_df.to_csv(RESULTS_DIR / "shap_regression_group_share.csv", index=False)
    lit = literature_table(tables)
    lit.to_csv(RESULTS_DIR / "literature_check.csv", index=False)
    print("\nLiterature comparison:")
    print(lit[["factor", "target", "expected", "data_direction", "importance_rank",
               "top5_freq", "sign_stability", "verdict"]].to_string(index=False))
    write_json(RESULTS_DIR / "shap_regression_audit.json", {
        "method": "Tree SHAP on the mean of Huber GB and ExtraTrees; fitted on all valid records",
        "bootstraps": N_BOOT, "bootstrap_unit": "publication",
        "interpretation": "Associational; not causal. Direction = correlation of feature value with SHAP.",
        "literature_verdicts": lit.groupby("verdict").size().to_dict(),
    })
    stage_status("08_regression_shap", "completed", {"targets": TARGETS})


if __name__ == "__main__":
    main()
