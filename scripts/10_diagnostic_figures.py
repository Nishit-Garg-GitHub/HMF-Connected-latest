"""Stage 10 — schematic and diagnostic figures for the report.

Everything is drawn from files already written by stages 01-09; no model is
retrained. Figures (PNG + PDF) go to figures/diagnostics/:

  Fig_Workflow                 pipeline schematic (analogue of the reference Fig. 1-2)
  Fig_Data_Overview            record availability, outcome distributions, catalyst families
  Fig_KMeans_Diagnostics       inertia / silhouette and cluster sizes for K=3-6
  Fig_Classification_Heatmap   pooled macro-F1 by model x sampler, binary and K=3
  Fig_Regression_Parity        out-of-fold observed vs predicted, four outcomes
  Fig_Regression_Skill         R2 and MAE versus the training-fold median baseline
  Fig_NSGA2_Pareto             Pareto projections for Ni / Cu / Ni-Cu hydroxide
  Fig_Literature_Check         SHAP direction versus published direction
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from common import FIGURES_DIR, RESULTS_DIR, TARGETS, TARGET_LABELS, read_processed, stage_status

warnings.filterwarnings("ignore")
OUT = FIGURES_DIR / "diagnostics"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREY, GREEN, ORANGE = "#2c6fbb", "#c0392b", "#8a8f98", "#2e8b57", "#e08e0b"


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def box(ax, xy, w, h, text, color):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                fc=color, ec="#333", lw=1, alpha=0.9))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=7.4)


def fig_workflow():
    gate = json.loads((RESULTS_DIR / "regression_validation_gate.json").read_text())
    audit = json.loads((RESULTS_DIR / "kmeans_audit.json").read_text())
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.4); ax.axis("off")
    steps = [
        (0.1, 3.7, "154 literature records\n99 publications", "#dfe9f5"),
        (2.5, 3.7, "Cleaning + descriptors\nanode-only catalyst parsing,\nelectrolyte, T, E, cell", "#dfe9f5"),
        (5.0, 3.7, f"K-means on (Conversion, Yield)\n{audit['clustered_records']} records, K = 3-6", "#e6f2e1"),
        (7.5, 3.7, "ROS / SMOTE / ADASYN\ninside training folds only", "#e6f2e1"),
        (10.0, 3.7, "7 classifiers\npaper-grouped CV", "#e6f2e1"),
        (0.1, 1.5, "Per-target regressors\n(Huber GB + Extra Trees)", "#fbeed6"),
        (2.5, 1.5, "Validation gate\nbeats median baseline,\nR2 >= 0, majority of folds", "#fbeed6"),
        (5.0, 1.5, "Tree SHAP +\npaper bootstraps +\nliterature check", "#f6dede"),
        (7.5, 1.5, "NSGA-II (pop 100, 1000 gen)\nmax yield, FE; min T, E\nNi / Cu / Ni-Cu hydroxide", "#f6dede"),
        (10.0, 1.5, "Entropy-weight ranking\nmember spread, nearest\nobserved record", "#f6dede"),
    ]
    for x, y, t, c in steps:
        box(ax, (x, y), 2.15, 1.15, t, c)
    for i in range(4):
        ax.annotate("", (steps[i + 1][0], 4.27), (steps[i][0] + 2.15, 4.27), arrowprops=dict(arrowstyle="->"))
    for i in range(5, 9):
        ax.annotate("", (steps[i + 1][0], 2.07), (steps[i][0] + 2.15, 2.07), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", (1.05, 2.65), (1.05, 3.7), arrowprops=dict(arrowstyle="->"))
    ax.text(6, 0.65, f"Required targets for the gate: {', '.join(gate['required_targets'])}  |  "
            f"gate passed: {gate['passed']}", ha="center", fontsize=9)
    ax.text(6, 0.2, "Predictions are associational and must be confirmed experimentally.", ha="center",
            fontsize=8.5, style="italic", color=GREY)
    ax.set_title("Workflow of the data-driven HMF-to-FDCA electro-oxidation study", fontweight="bold")
    save(fig, "Fig_Workflow")


def fig_data():
    frame = read_processed()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    counts = {TARGET_LABELS[t]: int((pd.to_numeric(frame[t], errors="coerce").between(0, 100)).sum())
              for t in TARGETS}
    counts["Conversion and yield"] = int((pd.to_numeric(frame["Conversion_pct"], errors="coerce").between(0, 100)
                                          & pd.to_numeric(frame["Yield_pct"], errors="coerce").between(0, 100)).sum())
    axes[0].barh(list(counts), list(counts.values()), color=BLUE)
    axes[0].axvline(len(frame), color=RED, ls="--"); axes[0].text(len(frame), -0.55, f" all {len(frame)}", color=RED)
    for i, v in enumerate(counts.values()):
        axes[0].text(v + 1, i, str(v), va="center")
    axes[0].set_title("(a) Valid records per outcome", loc="left", fontweight="bold")
    axes[0].set_xlabel("Records")
    for t, c in (("Conversion_pct", BLUE), ("Yield_pct", RED)):
        v = pd.to_numeric(frame[t], errors="coerce"); v = v[v.between(0, 100)]
        axes[1].hist(v, bins=np.arange(0, 105, 5), alpha=0.6, color=c, label=TARGET_LABELS[t])
    axes[1].set_title("(b) Outcome distributions", loc="left", fontweight="bold")
    axes[1].set_xlabel("%"); axes[1].set_ylabel("Records"); axes[1].legend(frameon=False)
    fam = {c.replace("Catalyst_", ""): int(pd.to_numeric(frame[c], errors="coerce").fillna(0).gt(0).sum())
           for c in ["Catalyst_Ni", "Catalyst_Co", "Catalyst_Fe", "Catalyst_Cu", "Catalyst_Pt", "Catalyst_Au",
                     "Catalyst_Hydroxide", "Catalyst_Oxide", "Catalyst_LDH", "Catalyst_Phosphide",
                     "Catalyst_Conductive_Glass", "Catalyst_Foam_Substrate"] if c in frame}
    fam = dict(sorted(fam.items(), key=lambda kv: kv[1]))
    axes[2].barh(list(fam), list(fam.values()), color=GREEN)
    axes[2].set_title("(c) Anode descriptors (records)", loc="left", fontweight="bold")
    save(fig, "Fig_Data_Overview")


def fig_kmeans():
    rows = []
    for k in (3, 4, 5, 6):
        lab = pd.read_csv(RESULTS_DIR / f"kmeans_labels_K{k}.csv")
        X = lab[["conversion_used", "yield_used"]].to_numpy()
        km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_),
                     "sizes": sorted(np.bincount(lab["cluster"]).tolist(), reverse=True)})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(df.k, df.inertia, "o-", color=BLUE); axes[0].set_title("(a) Elbow", loc="left", fontweight="bold")
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")
    axes[1].plot(df.k, df.silhouette, "o-", color=RED); axes[1].set_title("(b) Silhouette", loc="left", fontweight="bold")
    axes[1].set_xlabel("K")
    for i, r in df.iterrows():
        axes[2].bar([r.k] * len(r.sizes), r.sizes, bottom=[0] + list(np.cumsum(r.sizes)[:-1]), width=0.7,
                    color=plt.cm.tab10(np.arange(len(r.sizes))), edgecolor="white")
    axes[2].set_xlabel("K"); axes[2].set_ylabel("Records")
    axes[2].set_title("(c) Cluster sizes (singletons block oversampling)", loc="left", fontweight="bold", fontsize=9)
    save(fig, "Fig_KMeans_Diagnostics")


def fig_classification():
    d = pd.read_csv(RESULTS_DIR / "classification_summary.csv")
    panels = [("binary_high_performance_90", 0, "Binary, performance >= 90%"),
              ("kmeans_cluster", 3, "K-means K = 3")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (kind, k, title) in zip(axes, panels):
        s = d[(d.target_kind == kind) & (d.k == k)]
        pv = s.pivot(index="model", columns="sampler", values="macro_f1_mean")
        dummy = s.dummy_macro_f1_mean.iloc[0]
        im = ax.imshow(pv.values, cmap="YlGnBu", vmin=dummy, vmax=max(0.8, np.nanmax(pv.values)))
        ax.set_xticks(range(pv.shape[1]), pv.columns); ax.set_yticks(range(pv.shape[0]), pv.index)
        for (i, j), v in np.ndenumerate(pv.values):
            ax.text(j, i, "" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(f"{title}\nstratified dummy = {dummy:.2f}", fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8, label="pooled macro-F1")
    save(fig, "Fig_Classification_Heatmap")


def fig_parity():
    o = pd.read_csv(RESULTS_DIR / "per_target_oof_predictions.csv")
    s = pd.read_csv(RESULTS_DIR / "per_target_regression_summary.csv").set_index("target")
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.5))
    for ax, t in zip(axes.ravel(), TARGETS):
        d = o[o.target == t]
        ax.scatter(d.observed, d.predicted, s=18, alpha=0.6, color=BLUE, edgecolor="none")
        ax.plot([0, 100], [0, 100], "k--", lw=0.8)
        ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
        ax.set_xlabel("Observed (%)"); ax.set_ylabel("Out-of-fold predicted (%)")
        r = s.loc[t]
        ax.set_title(f"{TARGET_LABELS[t]}\nR2 = {r.model_r2_mean:.2f}, MAE = {r.model_mae_mean:.1f} "
                     f"(median baseline {r.baseline_mae_mean:.1f}), n = {int(r.n_records)}", fontsize=9)
    fig.tight_layout()
    save(fig, "Fig_Regression_Parity")


def fig_skill():
    s = pd.read_csv(RESULTS_DIR / "per_target_regression_summary.csv").set_index("target").loc[TARGETS]
    nested_path = RESULTS_DIR / "per_target_regression_nested.csv"
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    x = np.arange(len(TARGETS))
    axes[0].bar(x - 0.2, s.model_r2_mean, 0.4, yerr=s.model_r2_sd, color=BLUE, label="pooled out-of-fold R2")
    if nested_path.exists():
        n = pd.read_csv(nested_path)
        col = next((c for c in n.columns if "r2" in c.lower()), None)
        if col:
            nr = n.groupby("target")[col].mean().reindex(TARGETS)
            axes[0].bar(x + 0.2, nr.values, 0.4, color=ORANGE, label="nested (selection-corrected)")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xticks(x, [t.replace("_pct", "") for t in TARGETS]); axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("(a) R2 under paper-grouped CV", loc="left", fontweight="bold")
    axes[1].bar(x - 0.2, s.model_mae_mean, 0.4, color=BLUE, label="model")
    axes[1].bar(x + 0.2, s.baseline_mae_mean, 0.4, color=GREY, label="training-fold median")
    axes[1].set_xticks(x, [t.replace("_pct", "") for t in TARGETS]); axes[1].set_ylabel("MAE (points)")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("(b) MAE versus median baseline", loc="left", fontweight="bold")
    save(fig, "Fig_Regression_Skill")


def fig_nsga():
    p = pd.read_csv(RESULTS_DIR / "nsga2_pareto_hydroxide_all.csv")
    top = pd.read_csv(RESULTS_DIR / "nsga2_top10_ni_cu_hydroxide.csv")
    rob_path = RESULTS_DIR / "nsga2_top10_robust_potential.csv"
    rob = pd.read_csv(rob_path) if rob_path.exists() else None
    col = {"Ni": BLUE, "Cu": ORANGE, "Ni+Cu": GREEN}
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    pairs = [("potential_V_RHE", "predicted_Yield_pct", "Potential (V vs RHE)", "Predicted yield (%)"),
             ("temperature_C", "predicted_Yield_pct", "Temperature (C)", "Predicted yield (%)"),
             ("predicted_Yield_pct", "predicted_FE_pct", "Predicted yield (%)", "Predicted FE (%)")]
    for ax, (xc, yc, xl, yl), letter in zip(axes, pairs, "abc"):
        for m, g in p.groupby("metals"):
            ax.scatter(g[xc], g[yc], s=18, alpha=0.5, color=col.get(m, GREY), label=m)
        ax.scatter(top[xc], top[yc], s=70, facecolor="none", edgecolor="k", lw=1.2, label="top 10")
        if rob is not None:
            ax.scatter(rob[xc], rob[yc], marker="*", s=90, color=RED, label="robust (E >= 1.45 V)")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f"({letter})", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("NSGA-II Pareto solutions (hydroxide scenarios; model predictions, not measurements)",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "Fig_NSGA2_Pareto")


def fig_literature():
    lit = pd.read_csv(RESULTS_DIR / "literature_check.csv")
    colors = {"consistent": GREEN, "opposite": RED, "not distinguishable": GREY}
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = [f"{r.factor} [{r.target.replace('_pct', '')}]" for r in lit.itertuples()]
    ranks = lit["importance_rank"].fillna(0).astype(float)
    ax.barh(labels, ranks, color=[colors.get(v, GREY) for v in lit.verdict])
    ax.invert_yaxis(); ax.set_xlabel("SHAP importance rank (1 = most important)")
    for i, (v, r) in enumerate(zip(lit.verdict, ranks)):
        ax.text(r + 0.3, i, v, va="center", fontsize=8)
    ax.set_title("Regression SHAP versus published trends", fontweight="bold")
    save(fig, "Fig_Literature_Check")


def main() -> None:
    for fn in (fig_workflow, fig_data, fig_kmeans, fig_classification, fig_parity, fig_skill,
               fig_nsga, fig_literature):
        try:
            fn()
        except Exception as exc:  # keep going so one missing input does not block the rest
            print(f"FAILED {fn.__name__}: {type(exc).__name__}: {exc}")
            raise
    stage_status("10_diagnostic_figures", "completed", {"directory": str(OUT)})


if __name__ == "__main__":
    main()
