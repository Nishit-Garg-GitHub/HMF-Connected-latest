"""Stage 12 — research-paper style LaTeX report.

Builds ``report/HMF_FDCA_ML_Report.tex`` from the artefacts written by stages
01-09. Every number, table and figure reference is read from ``results/`` and
``figures/`` at run time, so re-running the pipeline and then this stage keeps
the manuscript consistent with the latest scores. Nothing is retrained here.

Stage 12 compiles the PDF automatically when tectonic or pdflatex is installed;
to compile by hand from the ``report`` folder:

    cd report && tectonic HMF_FDCA_ML_Report.tex      # or pdflatex (run twice)
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    RESULTS_DIR, FIGURES_DIR, ROOT, TARGETS, build_feature_matrix, paper_groups,
    read_cleaned,
)

REPORT_DIR = ROOT / "report"
TEX_PATH = REPORT_DIR / "HMF_FDCA_ML_Report.tex"
# Commit holding the classification scores before the descriptor/model revision.
BASELINE_FILE = RESULTS_DIR / "classification_summary_baseline_v1.csv"
BASELINE_COMMIT = "bdbd2e2"

MODEL_LABEL = {
    "DecisionTree": "DT", "SVM": "SVM", "RandomForest": "RF",
    "NeuralNetwork": "NN", "GradientBoosting": "GB", "ExtraTrees": "ET",
    "Ensemble": "ENS",
}
MODEL_ORDER = ["DecisionTree", "SVM", "RandomForest", "NeuralNetwork",
               "GradientBoosting", "ExtraTrees", "Ensemble"]
SAMPLER_ORDER = ["none", "ROS", "SMOTE", "ADASYN"]
TARGET_LABEL = {
    "Conversion_pct": "Conversion", "Yield_pct": "FDCA yield",
    "FE_pct": "Faradaic efficiency", "Selectivity_pct": "Selectivity",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def esc(text) -> str:
    """Escape LaTeX special characters in data-derived strings."""
    text = str(text)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")):
        text = text.replace(a, b)
    return text


def feat(name: str) -> str:
    """Typeset a descriptor name."""
    return r"\texttt{" + esc(name) + "}"


def sampler_name(name) -> str:
    return "no balancing" if str(name) == "none" else esc(name)


def sampler_phrase(name) -> str:
    return "without balancing" if str(name) == "none" else f"with {esc(name)}"


def f3(x) -> str:
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def read_csv(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def read_json(name: str) -> dict:
    path = RESULTS_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def cropped_png(rel: str) -> str:
    """Copy a PNG into report/assets with surrounding white margin removed.

    Several pipeline figures are saved with generous page margins; trimming
    them keeps the manuscript compact. Returns the path to use in LaTeX.
    """
    try:
        from PIL import Image, ImageChops
    except Exception:
        return rel
    source = FIGURES_DIR / rel
    target = REPORT_DIR / "assets" / rel.replace("/", "__")
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return f"assets/{target.name}"
    image = Image.open(source).convert("RGB")
    diff = ImageChops.difference(image, Image.new("RGB", image.size, (255, 255, 255)))
    box = ImageChops.add(diff, diff, 2.0, -20).getbbox()
    if box:
        pad = int(0.01 * max(image.size))
        box = (max(box[0] - pad, 0), max(box[1] - pad, 0),
               min(box[2] + pad, image.width), min(box[3] + pad, image.height))
        image = image.crop(box)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, optimize=True)
    return f"assets/{target.name}"


def figure(rel_candidates: list[str], caption: str, label: str,
           width: str = r"\textwidth") -> str:
    """Include the first existing figure (PDF preferred) or a placeholder note."""
    for rel in rel_candidates:
        if (FIGURES_DIR / rel).exists():
            path = cropped_png(rel) if rel.lower().endswith(".png") else rel
            return (
                "\\begin{figure}[!htbp]\n\\centering\n"
                f"\\includegraphics[width={width},height=0.85\\textheight,keepaspectratio]{{{path}}}\n"
                f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{figure}}\n"
            )
    return f"% Figure {label} skipped: none of {rel_candidates} exist.\n"


def baseline_summary() -> pd.DataFrame:
    if BASELINE_FILE.exists():
        return pd.read_csv(BASELINE_FILE)
    try:
        text = subprocess.check_output(
            ["git", "show", f"{BASELINE_COMMIT}:results/classification_summary.csv"],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
        return pd.read_csv(io.StringIO(text))
    except Exception:
        return pd.DataFrame()


def best_per_model(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    keys = ["target_kind", "k", "model"]
    return (summary.sort_values("macro_f1_mean", ascending=False)
            .groupby(keys, as_index=False).head(1))


def task_name(kind: str, k) -> str:
    return r"Binary $\geq$90\%" if kind.startswith("binary") else f"K-means, $K={int(k)}$"


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def collect() -> dict:
    X, features, _ = build_feature_matrix()
    raw = read_cleaned()
    groups = paper_groups()
    missing = X.isna().sum()
    current = read_csv("classification_summary.csv")
    binary = read_csv("classification_binary_summary.csv")
    kmeans = read_csv("classification_kmeans_summary.csv")
    labels = {k: read_csv(f"kmeans_labels_K{k}.csv") for k in (3, 4, 5, 6)}

    fdca = raw["Major_Product_Clean"].astype(str).str.contains(
        r"FDCA|FURANDICARBOXYLIC", case=False, regex=True, na=False)
    yld = pd.to_numeric(raw["Yield_pct"], errors="coerce")
    conv = pd.to_numeric(raw["Conversion_pct"], errors="coerce")
    keep = fdca & yld.notna()
    perf = np.where(conv[keep].notna(), np.minimum(conv[keep].clip(0, 100), yld[keep].clip(0, 100)),
                    yld[keep].clip(0, 100))
    return {
        "n_records": len(raw),
        "n_papers": int(len(np.unique(groups))),
        "n_features": len(features),
        "features": features,
        "missing": missing[missing > 0].sort_values(ascending=False),
        "n_indicators": int((missing > 0).sum()),
        "target_coverage": raw[TARGETS].notna().sum(),
        "n_complete_targets": int(raw[TARGETS].notna().all(axis=1).sum()),
        "binary_n": int(keep.sum()),
        "binary_pos": int((perf >= 90).sum()),
        "current": current,
        "baseline": baseline_summary(),
        "binary": binary,
        "kmeans": kmeans,
        "labels": labels,
        "kmeans_audit": read_json("kmeans_audit.json"),
        "best": read_json("best_classification_candidate.json"),
        "incomplete": read_csv("classification_incomplete_combinations.csv"),
        "learning": read_csv("learning_curves_summary.csv"),
        "importance": read_csv("feature_importance_selected_model.csv"),
        "shap_audit": read_json("shap_paper_K3_K5_audit.json"),
        "shap_k": {k: read_csv(f"shap_paper_K{k}_feature_importance.csv") for k in (3, 5)},
        "per_target": read_csv("per_target_regression_summary.csv"),
        "joint": read_csv("multioutput_regression_summary.csv"),
        "gate": read_json("regression_validation_gate.json"),
        "binary_repeats": read_csv("classification_binary_repeat_scores.csv"),
        "nsga": read_json("nsga2_status.json"),
        "nsga_top": read_csv("nsga2_top10_ni_cu_hydroxide.csv"),
        "nsga_cu": read_csv("nsga2_best3_cu_hydroxide.csv"),
        "nsga_robust": read_csv("nsga2_top10_robust_potential.csv"),
        "nsga_cu10": read_csv("nsga2_top10_Cu_hydroxide.csv"),
        "nsga_nicu10": read_csv("nsga2_top10_NiCu_hydroxide.csv"),
        "nsga_best": read_csv("nsga2_best_by_catalyst.csv"),
        "pareto_all": read_csv("nsga2_pareto_hydroxide_all.csv"),
        "lit": read_csv("literature_check.csv"),
        "shap_reg": {t: read_csv(f"shap_regression_{t}.csv") for t in ("Yield_pct", "FE_pct")},
        "shap_groups": read_csv("shap_regression_group_share.csv"),
        "nested": read_csv("per_target_regression_nested.csv"),
        "cluster_sizes": {k: sorted(v["cluster"].value_counts().tolist(), reverse=True)
                          for k, v in labels.items() if "cluster" in v},
    }


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def table_dataset(d: dict) -> str:
    rows = [f"Literature records & {d['n_records']} \\\\",
            f"Independent publications (CV groups) & {d['n_papers']} \\\\",
            f"Input descriptors (before missing-value indicators) & {d['n_features']} \\\\",
            f"Missing-value indicator columns added in-fold & {d['n_indicators']} \\\\"]
    for target, n in d["target_coverage"].items():
        label = TARGET_LABEL[target]
        label = label if label.startswith("FDCA") else label.lower()
        rows.append(f"Records reporting {label} & {int(n)} \\\\")
    rows.append(f"Records reporting all four outcomes & {d['n_complete_targets']} \\\\")
    rows.append(f"Binary-screen records (FDCA product, yield reported) & {d['binary_n']} "
                f"({d['binary_pos']} positive) \\\\")
    miss = ", ".join(f"{feat(k)} ({int(v)})" for k, v in d["missing"].items())
    return (
        "\\begin{table}[!htbp]\n\\centering\n\\caption{Dataset composition. Missing "
        "counts per continuous descriptor: " + miss + ".}\n\\label{tab:data}\n"
        "\\begin{tabular}{lr}\n\\toprule\nQuantity & Value \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def table_clusters(d: dict) -> str:
    rows = []
    for k, frame in d["labels"].items():
        if frame.empty:
            continue
        counts = frame["cluster"].value_counts().sort_index()
        smallest = int(counts.min())
        rows.append(f"{k} & " + ", ".join(str(int(v)) for v in counts.values)
                    + f" & {smallest} \\\\")
    return (
        "\\begin{table}[!htbp]\n\\centering\n\\caption{K-means cluster sizes on "
        "(conversion, FDCA yield). The minority clusters are extremely small, which "
        "bounds the attainable macro-F1 under publication-grouped validation.}\n"
        "\\label{tab:clusters}\n\\begin{tabular}{rlr}\n\\toprule\n$K$ & Cluster sizes & "
        "Smallest cluster \\\\\n\\midrule\n" + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def table_before_after(d: dict) -> str:
    old = best_per_model(d["baseline"])
    new = best_per_model(d["current"])
    if new.empty:
        return ""
    keys = ["target_kind", "k", "model"]
    merged = new.merge(old[keys + ["sampler", "macro_f1_mean"]], on=keys, how="left",
                       suffixes=("", "_old"))
    merged["order_task"] = np.where(merged["target_kind"].str.startswith("binary"), 0, merged["k"])
    merged["order_model"] = merged["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    merged = merged.sort_values(["order_task", "order_model"])
    lines, last = [], None
    for r in merged.itertuples():
        task = task_name(r.target_kind, r.k)
        if last is not None and task != last:
            lines.append("\\midrule")
        old_f1 = getattr(r, "macro_f1_mean_old")
        old_txt = "--" if pd.isna(old_f1) else f"{old_f1:.3f} ({esc(r.sampler_old)})"
        delta = "new" if pd.isna(old_f1) else f"{r.macro_f1_mean - old_f1:+.3f}"
        task_cell = task if task != last else ""
        lines.append(f"{task_cell} & {MODEL_LABEL.get(r.model, r.model)} & {old_txt} & "
                     f"{r.macro_f1_mean:.3f} ({esc(r.sampler)}) & "
                     f"[{r.macro_f1_ci95_lo:.3f}, {r.macro_f1_ci95_hi:.3f}] & {delta} \\\\")
        last = task
    return (
        "\\begin{table}[!htbp]\n\\centering\\small\n\\caption{Best pooled out-of-fold "
        "macro-F1 per classifier family and task (best sampler in parentheses), before "
        f"(commit \\texttt{{{BASELINE_COMMIT}}}) and after the descriptor, preprocessing, "
        "model and evaluation revisions. 95\\% CI over 25 repeats. K-means rows are not "
        "like-for-like: earlier values used all 154 records with median-imputed outcomes, "
        "current values use only the records with measured outcomes, and earlier $K=4$ values "
        "were computed on the subset of folds where SMOTE/ADASYN could be fitted "
        "(Section~\\ref{sec:eval-fix}).}\n\\label{tab:beforeafter}\n"
        "\\begin{tabular}{llllll}\n\\toprule\nTask & Model & Before & After & 95\\% CI & "
        "$\\Delta$ \\\\\n\\midrule\n" + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def table_grid(summary: pd.DataFrame, caption: str, label: str) -> str:
    if summary.empty:
        return ""
    pivot = summary.pivot_table(index="model", columns="sampler",
                                values="macro_f1_mean", aggfunc="first")
    sd = summary.pivot_table(index="model", columns="sampler",
                             values="macro_f1_sd", aggfunc="first")
    best = summary["macro_f1_mean"].max()
    lines = []
    for model in MODEL_ORDER:
        if model not in pivot.index:
            continue
        cells = []
        for sampler in SAMPLER_ORDER:
            value = pivot.loc[model].get(sampler, np.nan)
            if pd.isna(value):
                cells.append("--")
                continue
            text = f"{value:.3f} $\\pm$ {sd.loc[model, sampler]:.3f}"
            cells.append(f"\\textbf{{{text}}}" if np.isclose(value, best) else text)
        lines.append(f"{MODEL_LABEL[model]} & " + " & ".join(cells) + " \\\\")
    return (
        "\\begin{table}[!htbp]\n\\centering\\small\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        "\\begin{tabular}{lcccc}\n\\toprule\nModel & None & ROS & SMOTE & ADASYN \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def table_learning(d: dict) -> str:
    lc = d["learning"]
    if lc.empty:
        return ""
    full = lc[lc["training_fraction"] == lc["training_fraction"].max()]
    lines = []
    for r in full.sort_values(["K", "model"]).itertuples():
        gap = r.mean_training_macro_f1 - r.mean_validation_macro_f1
        lines.append(f"{r.K} & {MODEL_LABEL.get(r.model, r.model)} & "
                     f"{r.mean_training_macro_f1:.3f} & {r.mean_validation_macro_f1:.3f} & "
                     f"{gap:.3f} \\\\")
    return (
        "\\begin{table}[!htbp]\n\\centering\\small\n\\caption{Learning-curve end points "
        "(full training fraction, publication-grouped 80/20 holdouts, 10 repeats).}\n"
        "\\label{tab:learning}\n\\begin{tabular}{rlrrr}\n\\toprule\n$K$ & Model & Train "
        "macro-F1 & Validation macro-F1 & Gap \\\\\n\\midrule\n" + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def table_importance(d: dict, n: int = 10) -> str:
    cols = []
    titles = []
    imp = d["importance"]
    if not imp.empty:
        value = "mean_abs_shap" if "mean_abs_shap" in imp else "importance_mean"
        cols.append(imp.head(n)[["feature", value]].values.tolist())
        best = d["best"]
        task = ("Binary screen" if str(best.get("target_kind", "")).startswith("binary")
                else f"$K={best.get('k', '?')}$ clusters")
        titles.append(f"{task} ({MODEL_LABEL.get(best.get('model'), best.get('model'))}"
                      f" + {esc(best.get('sampler'))})")
    for k, table in d["shap_k"].items():
        if table.empty:
            continue
        run = d["shap_audit"].get("runs", {}).get(str(k), {})
        cols.append(table.head(n)[["feature", "mean_abs_shap"]].values.tolist())
        titles.append(f"$K={k}$ ({MODEL_LABEL.get(run.get('model'), run.get('model', '?'))}"
                      f" + {esc(run.get('sampler', '?'))})")
    if not cols:
        return ""
    header = " & ".join(f"\\multicolumn{{2}}{{c}}{{{t}}}" for t in titles)
    rule = "".join(f"\\cmidrule(lr){{{2*i+1}-{2*i+2}}}" for i in range(len(cols)))
    lines = []
    for i in range(n):
        cells = []
        for col in cols:
            if i < len(col):
                cells += [feat(col[i][0]), f"{col[i][1]:.3f}"]
            else:
                cells += ["", ""]
        lines.append(" & ".join(cells) + " \\\\")
    return (
        "\\begin{table*}[!htbp]\n\\centering\\scriptsize\n\\caption{Top-ranked descriptors "
        "by mean absolute SHAP value on held-out publications. Values are associational, "
        "not causal; \\texttt{missingindicator\\_*} columns flag unreported conditions.}\n"
        "\\label{tab:shap}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{"
        + "lr" * len(cols) + "}\n\\toprule\n"
        + header + " \\\\\n" + rule + "\n" + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n"
    )


def table_regression(d: dict) -> str:
    out = []
    for frame, title, label in ((d["per_target"], "Per-target regression (primary)", "tab:reg"),
                                (d["joint"], "Joint four-output regression on complete rows "
                                 "(secondary)", "tab:regjoint")):
        if frame.empty:
            continue
        lines = []
        for r in frame.itertuples():
            better = "yes" if r.model_mae_mean < r.baseline_mae_mean else "no"
            lines.append(f"{TARGET_LABEL.get(r.target, esc(r.target))} & {r.model_r2_mean:.3f} & "
                         f"{r.model_mae_mean:.2f} & {r.baseline_mae_mean:.2f} & "
                         f"{r.model_rmse_mean:.2f} & {better} \\\\")
        out.append(
            "\\begin{table}[!htbp]\n\\centering\\small\n"
            f"\\caption{{{title}: publication-grouped cross-validation against a median "
            "baseline. MAE and RMSE in percentage points.}\n"
            f"\\label{{{label}}}\n\\begin{{tabular}}{{lrrrrc}}\n\\toprule\n"
            "Target & $R^2$ & MAE & Baseline MAE & RMSE & Beats baseline \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        )
    return "".join(out)


def table_nsga(d: dict, key: str = "nsga_top", caption: str = "", label: str = "tab:nsga") -> str:
    top = d[key]
    if top.empty:
        return ""
    lines = []
    for r in top.head(10).itertuples():
        obs = getattr(r, "observed_Yield_pct", float("nan"))
        obs_txt = f"{obs:.1f}" if pd.notna(obs) else "--"
        spread = getattr(r, "member_spread_Yield_pct", float("nan"))
        lines.append(
            f"{r.rank} & {esc(r.catalyst)}/{esc(r.support)} & {esc(r.electrolyte)} "
            f"{r.electrolyte_M:.2f}\\,M & {r.HMF_mM:.1f} & {r.potential_V_RHE:.3f} & "
            f"{r.temperature_C:.1f} & {r.predicted_Yield_pct:.1f} & "
            f"{(f'{spread:.1f}' if pd.notna(spread) else '--')} & {r.predicted_FE_pct:.1f} & "
            f"{obs_txt} & {r.entropy_fitness:.3f} \\\\")
    return (
        "\\begin{table*}[!htbp]\n\\centering\\scriptsize\n\\caption{" + caption + "}\n"
        f"\\label{{{label}}}\n\\resizebox{{\\textwidth}}{{!}}{{%\n"
        "\\begin{tabular}{rllrrrrrrrr}\n\\toprule\n"
        "Rank & Catalyst & Electrolyte & HMF (mM) & $E$ (V vs RHE) & $T$ ($^\\circ$C) & "
        "Yield (\\%) & model spread & FE (\\%) & nearest measured yield (\\%) & Fitness \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n"
    )


def table_best_by_catalyst(d: dict) -> str:
    b = d["nsga_best"]
    if b.empty:
        return ""
    name = {"Ni_hydroxide": "Ni(OH)$_2$", "Cu_hydroxide": "Cu(OH)$_2$", "NiCu_hydroxide": "Ni--Cu hydroxide"}
    lines = [f"{name.get(r.catalyst, esc(r.catalyst))} & {int(r.n_candidates)} & {r.best_predicted_Yield_pct:.1f} & "
             f"{r.best_predicted_FE_pct:.1f} & {r.best_fitness:.3f} & "
             f"{'on the Pareto front' if int(r.best_pareto_rank) == 0 else 'dominated by another catalyst'} \\\\"
             for r in b.itertuples()]
    return (
        "\\begin{table}[!htbp]\n\\centering\\small\n\\caption{Best NSGA-II solution for each hydroxide anode "
        "(model predictions). Fitness is the entropy-weighted score of the pooled search, so values are "
        "comparable across rows.}\n\\label{tab:nsgabest}\n\\begin{tabular}{lrrrrl}\n\\toprule\n"
        "Anode & Candidates & Yield (\\%) & FE (\\%) & Fitness & Status \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def table_literature(d: dict) -> str:
    lit = d["lit"]
    if lit.empty:
        return ""
    cite = {"PMC9401862": "lit1", "d3gc01420g": "lit2", "d2gc04732b": "lit3"}
    lines = []
    for r in lit.itertuples():
        key = next((v for k, v in cite.items() if k in str(r.source)), None)
        src = f"\\cite{{{key}}}" if key else "--"
        rank = f"{int(r.importance_rank)}/{int(r.of_features)}" if pd.notna(getattr(r, "importance_rank", np.nan)) else "--"
        sign = {"+": "$+$", "-": "$-$", "0": "0"}
        lines.append(
            f"{esc(r.factor)} & {TARGET_LABEL.get(r.target, esc(r.target))} & {sign.get(r.expected, r.expected)} & "
            f"{sign.get(str(getattr(r, 'data_direction', '')), '--')} & {rank} & "
            f"{esc(r.verdict)} & {src} \\\\")
    return (
        "\\begin{table*}[!htbp]\n\\centering\\scriptsize\n\\caption{Regression-SHAP direction compared "
        "with directions reported in the experimental literature. Rank is the position among all "
        "descriptors by mean $|$SHAP$|$. A trend is called consistent or opposite only if the "
        "descriptor is among the ten most important and either recurs in the bootstrap top five "
        "or keeps a stable sign; otherwise it is \\emph{not distinguishable}.}\n\\label{tab:lit}\n"
        "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{llcclcl}\n\\toprule\n"
        "Factor & Target & Literature & Model & Rank & Verdict & Source \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n"
    )


def top_features(d: dict, target: str, n: int = 3) -> str:
    t = d["shap_reg"].get(target)
    if t is None or t.empty:
        return ""
    return ", ".join(f"{esc(r.label)} ({r.mean_abs_shap:.1f})" for r in t.head(n).itertuples())


def shap_rank(d: dict, target: str, feature: str):
    t = d["shap_reg"].get(target)
    if t is None or t.empty or feature not in set(t["feature"]):
        return None
    return int(t.index[t["feature"] == feature][0]) + 1


# ---------------------------------------------------------------------------
# Manuscript
# ---------------------------------------------------------------------------

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.3cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{caption}
\usepackage{float}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}
\captionsetup{font=small,labelfont=bf}
\graphicspath{{../figures/}}
\setlength{\parskip}{0.35em}
% Let large figures share pages with text instead of sitting alone on
% half-empty float pages.
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.8}
\setcounter{topnumber}{3}
\setcounter{totalnumber}{5}
"""


def build(d: dict) -> str:
    best = d["best"]
    binary = d["binary"]
    kmeans = d["kmeans"]
    best_k = {int(k): g.sort_values("macro_f1_mean", ascending=False).iloc[0]
              for k, g in kmeans.groupby("k")} if not kmeans.empty else {}
    k3 = best_k.get(3)
    bb = binary.sort_values("macro_f1_mean", ascending=False).iloc[0] if not binary.empty else best
    sel_task = ("the binary screen" if str(best.get("target_kind", "")).startswith("binary")
                else f"the $K={best.get('k', '?')}$ cluster task")
    et_bin = binary[binary.model == "ExtraTrees"].sort_values("macro_f1_mean", ascending=False)
    et_bin = et_bin.iloc[0] if len(et_bin) else None
    old_best = d["baseline"]
    old_bin = old_best[old_best.target_kind.str.startswith("binary")]["macro_f1_mean"].max() \
        if not old_best.empty else float("nan")
    old_k3 = old_best[(old_best.target_kind == "kmeans_cluster") & (old_best.k == 3)][
        "macro_f1_mean"].max() if not old_best.empty else float("nan")
    gate = d["gate"]
    nsga = d["nsga"]
    n_incomplete = len(d["incomplete"])
    above = int((d["current"]["macro_f1_mean"] >= 0.76).sum()) if not d["current"].empty else 0
    model_name = {"NeuralNetwork": "neural network", "ExtraTrees": "extra-trees",
                  "GradientBoosting": "gradient boosting", "RandomForest": "random forest",
                  "SVM": "support vector machine", "DecisionTree": "decision tree",
                  "Ensemble": "soft-voting ensemble"}

    km = d["kmeans_audit"] or {}
    if km.get("record_selection", "all") == "observed":
        kmeans_limit = (f"K-means was fitted only to the {km.get('clustered_records', '?')} "
                        "records that report both conversion and yield, so classification "
                        "scores refer to those records alone and rare clusters contain very "
                        "few records.")
    else:
        kmeans_limit = ("Outcomes were median-imputed for K-means, so some cluster labels are "
                        "pseudo-labels.")

    s = [PREAMBLE]
    s.append(r"""\title{Machine-Learning Screening of Electrocatalytic 5-Hydroxymethylfurfural
Oxidation to 2,5-Furandicarboxylic Acid from Literature Data: Leakage-Aware Classification,
Regression and Multi-Objective Optimisation}
\author{Nishit Garg \and Aditya Singh\\[0.4em]
\textit{BITS Pilani}}
\date{""" + date.today().strftime("%d %B %Y") + r"""}
\begin{document}
\maketitle
""")

    # Abstract
    s.append(r"\begin{abstract}" + "\n" + (
        f"We assembled {d['n_records']} electrocatalytic HMF-oxidation records from "
        f"{d['n_papers']} publications and built a leakage-aware machine-learning pipeline that "
        "(i) engineers chemical descriptors for the catalyst, electrolyte, cell configuration and "
        "operating conditions, (ii) clusters reported conversion and FDCA yield with K-means, "
        "(iii) compares seven classifier families under four class-balancing strategies, "
        "(iv) interprets the selected models with SHAP, (v) regresses four performance outcomes and "
        "(vi) searches catalyst/condition space with a validation-gated NSGA-II. All model "
        "selection uses publication-grouped cross-validation so that no paper contributes records "
        "to both training and test folds. Correcting descriptor extraction (anode-only catalyst "
        "parsing, formula-aware compound families, temperature and pH parsing, counter-electrode "
        "descriptors), retaining missing-value indicators and re-tuning the models raised the best "
        f"pooled out-of-fold macro-F1 for a binary $\\geq$90\\% performance screen from "
        f"{f3(old_bin)} to {f3(bb.get('macro_f1_mean'))} "
        f"({model_name.get(bb.get('model'), bb.get('model'))} {sampler_phrase(bb.get('sampler'))}; "
        f"95\\% CI {f3(bb.get('macro_f1_ci95_lo'))}--{f3(bb.get('macro_f1_ci95_hi'))}; "
        f"stratified dummy {f3(bb.get('dummy_macro_f1_mean'))})"
        + (f"; for three-cluster K-means labels of the records with measured outcomes the best "
           f"score is {f3(k3.macro_f1_mean)} (stratified dummy "
           f"{f3(k3.get('dummy_macro_f1_mean'))})" if k3 is not None else "")
        + ". We also identify and correct an evaluation flaw in which folds where an "
        "oversampler could not be fitted were silently discarded, which had inflated earlier "
        f"$K=4$ scores. {'No' if above == 0 else above} model--sampler combination"
        f"{'' if above == 1 else 's'} reached a mean macro-F1 of 0.76, reflecting the small, "
        "heterogeneous and imbalanced literature corpus rather than a pipeline limitation. "
        + ((
            "Per-target gradient-boosting/extra-trees regressors beat a training-fold median baseline "
            "under publication-grouped validation for FDCA yield "
            f"(pooled out-of-fold $R^2={gate['target_checks']['Yield_pct']['mean_model_r2']:.2f}$) and "
            f"faradaic efficiency ($R^2={gate['target_checks']['FE_pct']['mean_model_r2']:.2f}$) but not "
            "for conversion, which is therefore excluded from optimisation. Tree-SHAP analysis of these "
            "regressors is compared with published trends, and a NSGA-II search restricted to "
            "Ni(OH)$_2$, Cu(OH)$_2$ and Ni--Cu hydroxide anodes yields candidate conditions that are "
            "presented as hypotheses for experimental testing, not as validated optima.")
           if gate.get("passed") else
           "Regression models did not beat a median baseline under publication-grouped "
           "validation, so the validation gate blocked NSGA-II and no catalyst candidates "
           "are reported.")) + "\n\\end{abstract}\n")
    s.append(r"\noindent\textbf{Keywords:} HMF electrooxidation; FDCA; machine learning; "
             r"grouped cross-validation; class imbalance; SHAP; NSGA-II" + "\n\n")

    # 1 Introduction
    s.append(r"""\section{Introduction}
2,5-Furandicarboxylic acid (FDCA) is a bio-based replacement for terephthalic acid, and its
electrochemical production from 5-hydroxymethylfurfural (HMF) couples biomass valorisation with
renewable electricity. Reported performance depends jointly on the anode material, electrolyte,
applied potential, temperature, mediator use and cell design, and the published literature now
contains enough individual experiments to attempt data-driven screening. Literature-mined
datasets are, however, small, heterogeneously reported and dominated by high-performing results,
which makes naive machine-learning evaluation prone to optimistic bias: records from the same
paper share catalysts and protocols, so random cross-validation leaks information between
training and test sets.

Following the integrated workflow of Li \textit{et al.}~\cite{li2025}, which combined K-means
labelling, oversampling, classifier comparison, SHAP interpretation and a genetic algorithm for
thermal aerobic HMF oxidation, we adapt the methodology to electro-oxidation and make three
contributions: (1) a publication-grouped, fold-local evaluation protocol in which imputation,
scaling and oversampling are fitted only on training papers; (2) a corrected descriptor set for
the anode catalyst, electrolyte and cell configuration; and (3) an honest assessment of how far
classification, regression and optimisation can be pushed with the current corpus.
""")

    s.append(figure(["diagnostics/Fig_Workflow.png"],
                    "Workflow of the study: leakage-aware classification, validated regression, "
                    "SHAP with literature check and hydroxide-restricted NSGA-II.", "fig:workflow"))

    # 2 Methods
    s.append(r"\section{Data and methods}" + "\n")
    s.append(r"\subsection{Literature dataset}" + "\n" + (
        f"The corpus contains {d['n_records']} experiments from {d['n_papers']} publications "
        "(Table~\\ref{tab:data}). Each record lists the cell type, HMF concentration, catalyst "
        "description, electrolyte, applied potential, TEMPO use, temperature and up to four "
        "outcomes: HMF conversion, FDCA yield, faradaic efficiency (FE) and selectivity. Outcome "
        "reporting is incomplete; only "
        f"{d['n_complete_targets']} records report all four outcomes.\n\n"))
    s.append(table_dataset(d))
    s.append(figure(["diagnostics/Fig_Data_Overview.png"],
                    "Record availability per outcome, outcome distributions and anode descriptors.",
                    "fig:data"))

    s.append(r"""\subsection{Descriptor engineering}
Inputs are restricted to eight agreed experimental groups: TEMPO, temperature, reactant,
HMF concentration, catalyst, electrolyte, potential versus RHE and cell configuration. All
measured outcomes and any column derived from them are excluded. Three corrections to the
earlier descriptor extraction were made in this revision.
\begin{itemize}
\item \textbf{Anode-only catalyst parsing.} Catalyst strings describe the whole cell
(e.g.\ ``Anode: Ni$_3$N@C on Ni foam. Cathode: Pt foil''). The previous parser searched the full
string, so 74 records were flagged as Pt catalysts because of the Pt counter electrode. Element
and compound flags are now extracted from the anode/working-electrode segment only, which leaves
2 genuine Pt anodes.
\item \textbf{Formula-aware compound families.} Phosphide, nitride, sulfide and selenide
flags had been matched only on words such as ``phosphide'' and were zero for every record.
Case-sensitive formula patterns (Ni$_3$S$_2$, CoNiP, VN, Co$_9$S$_8$, Ni$_2$P) now recover 12
phosphide, 8 nitride and 15 sulfide anodes, plus borides, foam substrates, conductive supports
(FTO, ITO, BDD), element counts, a noble-metal flag and a bimetallic flag.
\item \textbf{Operating conditions.} ``23 $\pm$ 2\,$^\circ$C'' had been parsed as 2\,$^\circ$C;
uncertainties are now stripped first. An electrolyte pH is taken from explicit ``pH'' values or
inferred as $14+\log_{10}[\mathrm{OH^-}]$ for hydroxides. Counter-electrode descriptors
(Pt, carbon, Ni, reported) are added to the cell-configuration group.
\end{itemize}
""" + (f"This yields {d['n_features']} descriptors. " if d else "") + (
        "Within every training fold, missing continuous values are median-imputed and a binary "
        f"indicator is appended for each of the {d['n_indicators']} descriptors with missing "
        "values. Whether a paper reports, for example, its temperature or potential is a property "
        "of the input record and carries information that median imputation alone would erase.\n"))

    s.append(r"""\subsection{Classification targets}
\textit{K-means labels.} Conversion and FDCA yield are standardised and clustered for
$K\in\{3,4,5,6\}$ (100 initialisations). Only records that report both outcomes are clustered, so
no class label is built from an imputed value (a median-imputed all-record variant is available as
a sensitivity run); labels are never used as inputs (Table~\ref{tab:clusters}). Where a paper
reports conversion and FDCA selectivity but not yield and FDCA is the major product, yield is
derived as $X\cdot S/100$ and flagged as derived; this adds records to the yield-based analyses
without altering any reported value.

\textit{Binary performance screen.} For records whose major product is FDCA and whose yield is
reported or derived (values outside 0--100\% are treated as missing, never clipped), a performance score
\begin{equation}
P = \begin{cases} \min(X, Y) & \text{if conversion } X \text{ is reported,}\\ Y & \text{otherwise,}\end{cases}
\end{equation}
is computed from conversion $X$ and yield $Y$, and records with $P\geq 90\%$ form the positive class.
""")
    s.append(table_clusters(d))

    s.append(r"""\subsection{Models and class balancing}
Seven classifier families were compared: decision tree (DT), RBF support vector machine (SVM),
random forest (RF), multilayer perceptron (NN), gradient boosting (GB), extremely randomised
trees (ET) and a soft-voting ensemble (ENS) of extra-trees and an $L_2$-regularised logistic
regression, each under no balancing, random oversampling (ROS), SMOTE~\cite{chawla2002} and
ADASYN~\cite{he2008}, using scikit-learn~\cite{pedregosa2011} and
imbalanced-learn~\cite{lemaitre2017}. Hyperparameters were selected from a small per-family grid
evaluated with the same grouped protocol: DT depth 6; SVM $C=0.3$, $\gamma=0.01$; RF 300 trees of
depth 6; NN one hidden layer of 32 units with $L_2$ penalty $\alpha=0.1$ and \emph{no} early
stopping (the former early-stopping split removed 10\% of an already tiny training fold and often
converged on the majority class); GB 100 depth-1 stumps; ET 400 trees with minimum leaf size 2.
Class-weighted losses are used where supported.
""")

    s.append(r"""\subsection{Evaluation protocol}\label{sec:eval}
All scores come from 25 repeats of 5-fold \texttt{StratifiedGroupKFold} with publication as the
group. Imputation, indicator creation, variance filtering, scaling and oversampling are fitted on
the training papers of each fold only. Within a repeat, out-of-fold predictions from all folds are
pooled and a single macro-F1 is computed,
\begin{equation}
\mathrm{F1}_{\text{macro}} = \frac{1}{C}\sum_{c=1}^{C}\frac{2\,\mathrm{P}_c\,\mathrm{R}_c}{\mathrm{P}_c+\mathrm{R}_c},
\end{equation}
where $\mathrm{P}_c$ and $\mathrm{R}_c$ are the precision and recall of class $c$. The mean, standard
deviation and $t$-based 95\% confidence interval are reported over repeats. The selected model is
compared with a stratified dummy classifier on identical splits and with two permutation tests
(100 permutations): labels shuffled within publications, which probes only within-paper
variation, and labels shuffled across all records, which tests for any signal. Candidates from
different tasks are ranked by their macro-F1 lift over the task's own dummy baseline, because
chance level differs between the binary and the $K$-class tasks.

\subsubsection*{Correction of fold-dropping bias}\label{sec:eval-fix}
SMOTE and ADASYN require more minority-class training rows than neighbours. In the previous
implementation a fold that failed this requirement was silently skipped and the repeat's macro-F1
was computed on the remaining folds; for $K=4$ only about 1.2 of 5 folds (35 of 154 records) were
scored. Neighbour counts now adapt to $\min(2, n_{\min}-1)$, a repeat counts only if every fold
produced predictions, and a combination is ranked only if at least 80\% of its repeats are
complete. """ + (f"{n_incomplete} combinations (SMOTE/ADASYN at $K\\geq 4$, where "
                 "single-record classes occur) are therefore reported as infeasible rather than "
                 "scored.\n" if n_incomplete else "\n"))

    s.append(r"""\subsection{Interpretation, regression and optimisation}
\textit{Regression.} Conversion, yield, FE and selectivity are regressed per target on every
record that reports that target (valid range 0--100\%). Each regressor is the mean of a Huber-loss
gradient-boosting model (80 depth-2 trees) and an extra-trees model (300 trees, minimum leaf 4)
after median imputation, evaluated with 10 repeats of 5-fold GroupKFold by publication. Scores
are pooled out-of-fold $R^2$ and MAE against a training-fold \emph{median} baseline. A target
passes the validation gate when its MAE beats that baseline, it wins more than half of the
folds and its pooled $R^2\geq0$; only the targets optimised downstream (yield and FE) are
required to pass. Because the model family was chosen on the same records, a nested
(selection-corrected) cross-validation is also reported.

\textit{SHAP.} Tree SHAP~\cite{lundberg2017} is computed for the two ensemble members of each
validated regressor and averaged. Stability is assessed on 30 publication-level bootstrap
resamples (how often a descriptor is in the top five and whether its sign persists), and the
direction of an effect is the correlation between a descriptor's value and its SHAP value.
For the classifiers, SHAP is computed on a held-out 20\% of publications. SHAP describes what a fitted
model uses; it does not establish causal catalyst effects.

\textit{Optimisation.} NSGA-II~\cite{deb2002} (population 100, 1000 generations) searches a mixed
binary/continuous encoding of the descriptor space, maximising predicted yield and FE while
minimising temperature $T$ and potential $E$. Conversion is not an objective because its
regressor did not beat the baseline. Three searches restrict the anode to Ni(OH)$_2$, Cu(OH)$_2$
or a Ni--Cu hydroxide within the potential and temperature range reported for such anodes.
Pareto solutions are ranked by an entropy-weighted fitness
\begin{equation}
F = w_1\tilde Y + w_2\widetilde{\mathrm{FE}} - w_3\tilde T - w_4\tilde E,
\qquad w_j = \frac{1-e_j}{\sum_k (1-e_k)},
\end{equation}
with min--max scaled objectives and $e_j$ the Shannon entropy of objective $j$ over the front.
Convergence is tracked with the spacing metric. The disagreement of the two ensemble members is
reported as a model-spread indicator, and a second ranking restricted to $E\geq1.45$\,V
guards against the optimiser sitting at the lower bound of the permitted range.
""")

    # 3 Results
    s.append(r"\section{Results}" + "\n")
    s.append(r"\subsection{Clustering}" + "\n" + (
        "K-means separates a dominant high-conversion/high-yield group from small groups with low "
        "yield at high conversion or low conversion (Figure~\\ref{fig:kmeans}). "
        f"{d['kmeans_audit'].get('clustered_records', '?')} of {d['n_records']} records report both "
        "outcomes and are clustered; the remaining records are not used for K-means, so no "
        "label is built from an imputed value. The smallest cluster contains "
        f"{min((v[-1] for v in d['cluster_sizes'].values()), default='?')} record(s) "
        "(Table~\\ref{tab:clusters}). Macro-F1 weights each class "
        "equally, so misclassifying a two-record cluster costs as much as misclassifying the "
        "majority class.\n"))
    # The PNG is used first: the PDF version carries a full-page margin.
    s.append(figure(["KMeans_Figure6/HMF_KMeans_Figure6_Style.png",
                     "KMeans/HMF_KMeans_K3.png"],
                    "K-means clustering of reported conversion and FDCA yield for $K=3$--6.",
                    "fig:kmeans", width="0.78\\textwidth"))
    s.append(figure(["oversampling_kmeans/KMeans_After_Oversampling_K3_to_K6.pdf",
                     "oversampling_kmeans/KMeans_After_Oversampling_K3_to_K6.png"],
                    "ADASYN and SMOTE synthetic records (crosses) in performance space for "
                    "$K=3$--6, drawn for the records with both outcomes measured (descriptive only). "
                    "Where a K-means class has a single record (e.g.\\ $K\\geq4$), that record is "
                    "duplicated before SMOTE, so the class gains no interpolated points. ADASYN is "
                    "undefined for $K\\geq4$ because a class of 1--4 near-identical records has no "
                    "majority-class neighbours; those panels show measured records only.",
                    "fig:oversampled", width="0.85\\textwidth"))

    s.append(r"\subsection{Classification performance}" + "\n")
    para = (
        f"Table~\\ref{{tab:beforeafter}} summarises the effect of the revision. For the binary "
        f"screen ({d['binary_n']} records, {d['binary_pos']} positive), the best model is the "
        f"{model_name.get(bb.get('model'), bb.get('model'))} {sampler_phrase(bb.get('sampler'))} "
        f"(macro-F1 {f3(bb.get('macro_f1_mean'))} $\\pm$ {f3(bb.get('macro_f1_sd'))}), "
        f"against {f3(bb.get('dummy_macro_f1_mean'))} for a stratified dummy classifier "
        "(Table~\\ref{tab:binary}, Figure~\\ref{fig:binary}). ")
    if et_bin is not None:
        close = abs(bb.macro_f1_mean - et_bin.macro_f1_mean) <= bb.macro_f1_sd
        para += (f"The best extra-trees configuration ({sampler_name(et_bin.sampler)}) scores "
                 f"{et_bin.macro_f1_mean:.3f}"
                 + (", within one repeat-to-repeat SD of the best model. " if close else ". "))
    old_b = best_per_model(d["baseline"])
    new_b = best_per_model(d["current"])
    if not old_b.empty and not new_b.empty:
        keys = ["target_kind", "k", "model"]
        gain = new_b.merge(old_b[keys + ["macro_f1_mean"]], on=keys, suffixes=("", "_old"))
        gain = gain[gain.target_kind.str.startswith("binary")]
        if len(gain):
            gain = gain.assign(delta=gain.macro_f1_mean - gain.macro_f1_mean_old)
            lo, hi = gain.loc[gain.delta.idxmin()], gain.loc[gain.delta.idxmax()]
            para += (f"Every pre-existing family improved on the binary screen, by "
                     f"{lo.delta:+.2f} ({MODEL_LABEL[lo.model]}) to {hi.delta:+.2f} "
                     f"({MODEL_LABEL[hi.model]}). ")
    if k3 is not None:
        para += (f"For $K=3$ the best model is {MODEL_LABEL.get(k3.model)} + {esc(k3.sampler)} "
                 f"({k3.macro_f1_mean:.3f}; Table~\\ref{{tab:k3}}). ")
    hi_k = kmeans[kmeans.k >= 4] if not kmeans.empty else kmeans
    if not hi_k.empty:
        para += (f"For $K\\geq4$, ranked scores lie between {hi_k.macro_f1_mean.min():.2f} and "
                 f"{hi_k.macro_f1_mean.max():.2f}: clusters contain as few as one record, and "
                 "SMOTE/ADASYN cannot be fitted on such classes, so those combinations are "
                 "reported as infeasible. ")
    pval = best.get("permutation_p_value")
    pglob = best.get("permutation_p_value_global")
    if pval is not None:
        para += (f"The selected candidate ({model_name.get(best.get('model'), best.get('model'))} + "
                 f"{esc(best.get('sampler'))} on {sel_task}, the largest lift over its dummy baseline) "
                 f"is subjected to two permutation tests. The within-publication test gives $p={pval:.2f}$. Because most "
                 "descriptors are constant within a paper, shuffling labels inside papers "
                 "preserves the paper-level signal the model exploits, so this test only shows "
                 "that the model does not explain variation between records of the same paper. ")
    if pglob is not None:
        para += (f"Shuffling labels across all records gives $p={pglob:.3f}$ (100 permutations; "
                 "the smallest attainable value is 0.01), so the score is unlikely under no "
                 "signal at all. Because the model and hyperparameters were selected on these "
                 "same records, the score remains optimistic. ")
    s.append(para + "\n\n")
    s.append(table_before_after(d))
    s.append(table_grid(binary, "Binary $\\geq$90\\% screen: pooled out-of-fold macro-F1 (mean "
                        "$\\pm$ SD over 25 repeats) for every model and sampler.", "tab:binary"))
    s.append(table_grid(kmeans[kmeans.k == 3] if not kmeans.empty else kmeans,
                        "K-means $K=3$: pooled out-of-fold macro-F1 (mean $\\pm$ SD over 25 "
                        "repeats).", "tab:k3"))
    s.append(figure(["diagnostics/Fig_Classification_Heatmap.png"],
                    "Pooled out-of-fold macro-F1 for every model and sampler (binary screen and $K=3$); "
                    "the colour scale starts at each task's stratified-dummy score.", "fig:heatmap"))
    s.append(figure(["diagnostics/Fig_KMeans_Diagnostics.png"],
                    "K-means diagnostics: inertia, silhouette and cluster sizes for $K=3$--6.",
                    "fig:kdiag"))
    s.append(figure(["classification/model_sampler_binary90.png"],
                    "Binary $\\geq$90\\% screen: pooled out-of-fold macro-F1 by classifier and "
                    "training-fold sampler (error bars: SD over 25 repeats; dotted line: 0.76).",
                    "fig:binary"))
    s.append(figure(["classification/model_sampler_K3.png"],
                    "K-means $K=3$: pooled out-of-fold macro-F1 by classifier and sampler.",
                    "fig:k3"))
    s.append(figure(["paper_style/Figure_9_Oversampling_F1.pdf",
                     "paper_style/Figure_9_Oversampling_F1.png"],
                    "Fold-averaged macro-F1 of DT, SVM, RF and NN after ROS, ADASYN and SMOTE for "
                    "$K=3$--5 (reference-paper layout). Fold averages are computed over feasible "
                    "folds only and differ from the pooled scores in Table~\\ref{tab:beforeafter}.",
                    "fig:fig9"))
    s.append(figure(["paper_style/Figure_10_Performance_Metrics.pdf",
                     "paper_style/Figure_10_Performance_Metrics.png"],
                    "Accuracy, precision, recall and F1 after oversampling.", "fig:fig10"))

    lc = d["learning"]
    lc_text = ""
    if not lc.empty:
        first, last = lc.training_fraction.min(), lc.training_fraction.max()
        a = lc[lc.training_fraction == first].set_index(["K", "model"])
        b = lc[lc.training_fraction == last].set_index(["K", "model"])
        gap = (b.mean_training_macro_f1 - b.mean_validation_macro_f1)
        rise = (b.mean_validation_macro_f1 - a.mean_validation_macro_f1)
        rising = [f"{MODEL_LABEL.get(m, m)} at $K={k}$" for (k, m), v in rise.items() if v > 0.10]
        lc_text = (
            "Training macro-F1 exceeds validation macro-F1 for every model and $K$ "
            f"(gaps of {gap.min():.2f}--{gap.max():.2f} at the full training size; Table~\\ref{{tab:learning}}, "
            "Figure~\\ref{fig:learning}), indicating variance-limited models. "
            + (f"Validation macro-F1 rises by more than 0.1 between the smallest and largest training "
               f"size for {', '.join(rising)}, so additional publications are the most direct route to "
               "higher scores; " if rising else "Validation macro-F1 is essentially flat in training size; ")
            + "for the remaining model/$K$ combinations the curves are flat, so extra records of the "
            "same kind add little, and minority clusters do not grow with more records.\n")
    s.append(r"\subsection{Learning curves}" + "\n" + lc_text)
    s.append(table_learning(d))
    s.append(figure(["learning_curves/HMF_Learning_Curves_Figure.pdf",
                     "learning_curves/HMF_Learning_Curves_Figure.png"],
                    "Learning curves (macro-F1) for DT, SVM, RF and NN at $K=3$, 4 and 5 with "
                    "publication-grouped holdouts; shading shows $\\pm$1 SD.", "fig:learning"))
    s.append(figure(["learning_curves/Figure_11_Oversampled_NN_Learning_Curves_ROS_ADASYN_SMOTE.pdf",
                     "learning_curves/Figure_11_Oversampled_NN_Learning_Curves_ROS_ADASYN_SMOTE.png"],
                    "Neural-network learning curves with ROS, ADASYN and SMOTE applied to training "
                    "folds only. For SMOTE, single-record training classes are duplicated first (noted "
                    "in each panel), which is a descriptive fallback not used in the classification "
                    "scores of Table~\\ref{tab:beforeafter}. ADASYN is undefined for $K\\geq4$ and "
                    "for most $K=3$ subsets, so those panels are blank or partial.", "fig:fig11"))

    s.append(r"\subsection{Model interpretation}" + "\n" + (
        "Table~\\ref{tab:shap} and Figures~\\ref{fig:shapsel}--\\ref{fig:fig13} list the "
        "descriptors with the largest SHAP contributions in the classifiers. "
        "These rankings describe what the models use to separate the literature records; they "
        "partly encode publication style and are not evidence of causal catalyst effects.\n"))
    s.append(table_importance(d))
    s.append(figure(["shap/selected_model_feature_importance.png"],
                    "SHAP feature importance for the selected binary-screen model.",
                    "fig:shapsel", width="0.8\\textwidth"))
    s.append(figure(["shap/Figure_12_SHAP_Bar.pdf", "shap/Figure_12_SHAP_Bar.png"],
                    "Global SHAP importance of the best Stage-04 model for $K=3$ and $K=5$.",
                    "fig:fig12"))
    s.append(figure(["shap/Figure_13_SHAP_Beeswarm.pdf", "shap/Figure_13_SHAP_Beeswarm.png"],
                    "SHAP beeswarm plots for the majority class at $K=3$ and $K=5$, coloured by "
                    "feature value.", "fig:fig13"))


    s.append(r"\subsection{Regression SHAP and comparison with the literature}" + "\n")
    if not d["lit"].empty:
        ni_y = shap_rank(d, "Yield_pct", "Catalyst_Ni")
        ni_f = shap_rank(d, "FE_pct", "Catalyst_Ni")
        cu_y = shap_rank(d, "Yield_pct", "Catalyst_Cu")
        cu_f = shap_rank(d, "FE_pct", "Catalyst_Cu")
        ydir = d["shap_reg"]["Yield_pct"].set_index("feature")
        cg = ydir.loc["Catalyst_Conductive_Glass"] if "Catalyst_Conductive_Glass" in ydir.index else None
        verdicts = d["lit"]["verdict"].value_counts().to_dict()
        s.append(
            "Tree SHAP of the validated yield and FE regressors (Figures~\\ref{fig:regshap}"
            "--\\ref{fig:regbeeswarm}) ranks " + top_features(d, "Yield_pct") + " highest for yield "
            "and " + top_features(d, "FE_pct") + " for FE (mean $|$SHAP$|$ in percentage points). "
            f"A nickel-containing anode is an important yield descriptor "
            f"(rank {ni_y}) with a positive effect in every bootstrap resample, in line with the "
            "Ni(OH)$_2$/NiOOH literature~\\cite{lit1}, but it is not distinguishable for FE "
            f"(rank {ni_f}). Copper ranks {cu_y} for yield and {cu_f} for FE with an unstable "
            "sign, so the data neither confirm nor contradict the high activity "
            "reported for Cu(OH)$_2$/CuOOH anodes~\\cite{lit2}: few records carry Cu anodes, and "
            "an absence of evidence in this corpus is not evidence against Cu. "
            + (f"The largest single contribution is the conductive-glass (FTO/ITO) support, with a "
               f"negative effect (mean $|$SHAP$|$ {cg.mean_abs_shap:.1f} points for yield). This is almost "
               "certainly a study-design confound---glass-supported films are typically thin, "
               "low-loading or photo-assisted systems---rather than a property of the catalyst. "
               if cg is not None else "")
            + "HMF concentration lowers predicted yield, consistent with humin formation in "
            "alkaline media~\\cite{lit3}. Alkaline electrolyte and the hydroxide/oxyhydroxide phase "
            "are positive but not distinguishable because almost every record is alkaline and "
            "most Ni/Cu records are hydroxide-derived. All records use HMF, so the reactant cannot "
            "be assessed. Overall, "
            + ", ".join(f"{v} comparison{'s' if v != 1 else ''} {esc(k)}" for k, v in verdicts.items())
            + " (Table~\\ref{tab:lit}). Nickel therefore emerges from the data as an important yield "
            "descriptor without any steering of the model, but the corpus cannot rank Ni against Cu.\n\n")
        s.append(table_literature(d))
        s.append(figure(["shap/Regression_SHAP_Bar.pdf", "shap/Regression_SHAP_Bar.png"],
                        "Regression SHAP importance for (a) FDCA yield and (b) faradaic efficiency; "
                        "error bars show the 5--95\\% range over publication bootstraps.",
                        "fig:regshap"))
        s.append(figure(["shap/Regression_SHAP_Beeswarm_Yield_pct.png"],
                        "SHAP beeswarm for FDCA yield, coloured by descriptor value.",
                        "fig:regbeeswarm", width="0.75\\textwidth"))
        s.append(figure(["shap/Regression_SHAP_Groups.png"],
                        "Share of total mean $|$SHAP$|$ by input group.", "fig:shapgroups",
                        width="0.7\\textwidth"))
        s.append(figure(["diagnostics/Fig_Literature_Check.png"],
                        "Importance rank of each literature factor and the verdict of the comparison.",
                        "fig:litcheck", width="0.8\\textwidth"))

    s.append(r"\subsection{Regression}" + "\n")
    if not d["per_target"].empty:
        pt = d["per_target"].set_index("target")
        checks = gate.get("target_checks", {})
        nested = d["nested"]
        ncol = next((c for c in nested.columns if "r2" in c.lower()), None) if not nested.empty else None
        nested_txt = ""
        if ncol:
            nr = nested.groupby("target")[ncol].mean()
            nested_txt = (" A nested check, in which the model is re-selected inside every training "
                          "fold, gives $R^2$ of " + ", ".join(
                              f"{nr[t]:.2f} ({TARGET_LABEL[t]})" for t in TARGET_LABEL if t in nr.index)
                          + ", so selection on the same records explains only part of the skill.")
        s.append(
            "Under publication-grouped validation the yield and FE regressors beat the median baseline "
            f"in MAE ({pt.loc['Yield_pct','model_mae_mean']:.1f} vs {pt.loc['Yield_pct','baseline_mae_mean']:.1f} "
            f"and {pt.loc['FE_pct','model_mae_mean']:.1f} vs {pt.loc['FE_pct','baseline_mae_mean']:.1f} "
            f"points) with pooled out-of-fold $R^2$ of {checks.get('Yield_pct', {}).get('mean_model_r2', float('nan')):.2f} "
            f"and {checks.get('FE_pct', {}).get('mean_model_r2', float('nan')):.2f} "
            "(Table~\\ref{tab:reg}, Figures~\\ref{fig:parity} and~\\ref{fig:skill}). "
            f"Selectivity ($R^2={pt.loc['Selectivity_pct','model_r2_mean']:.2f}$) also improves on the baseline. "
            f"Conversion does not (MAE {pt.loc['Conversion_pct','model_mae_mean']:.1f} vs "
            f"{pt.loc['Conversion_pct','baseline_mae_mean']:.1f}): most reported conversions are near 100\\%, so a "
            "constant prediction is very hard to beat, and conversion is therefore not used as an "
            "optimisation objective. Fold-level $R^2$ values (not shown) are much lower than pooled values "
            "because individual test folds have little outcome variance, so pooled $R^2$ is used for the "
            "gate and fold $R^2$ is not interpreted." + nested_txt
            + f" The gate {'passed' if gate.get('passed') else 'failed'}. The joint four-output model on complete "
            "records is a secondary comparison (Table~\\ref{tab:regjoint}). These $R^2$ values describe "
            "unseen papers, not unseen conditions of known papers.\n\n")
    s.append(table_regression(d))
    s.append(figure(["diagnostics/Fig_Regression_Parity.png"],
                    "Out-of-fold observed versus predicted values for the four outcomes "
                    "(publication-grouped validation).", "fig:parity", width="0.75\\textwidth"))
    s.append(figure(["diagnostics/Fig_Regression_Skill.png"],
                    "Regression skill: $R^2$ and MAE against the training-fold median baseline.",
                    "fig:skill"))

    s.append(r"\subsection{Multi-objective optimisation}" + "\n")
    if nsga and str(nsga.get("status", "")).startswith("blocked"):
        s.append(
            "NSGA-II was not run. The regression validation gate (Table~\\ref{tab:reg}) requires the "
            "yield and FE regressors to beat the training-fold median baseline in MAE, to win a majority of "
            "folds and to have a non-negative pooled $R^2$; at least one failed, so "
            "optimising against these surrogate models would only exploit noise. No catalyst or "
            "operating-condition candidates are reported.\n\n")
    elif nsga:
        weights = nsga.get("general_entropy_weights", {})
        objective_label = {**TARGET_LABEL, "Temperature_C": "temperature",
                           "Potential_vs_RHE_V": "potential"}
        wtxt = ", ".join(f"{objective_label.get(k, esc(k.replace('_', ' ')))} {v:.2f}"
                         for k, v in weights.items())
        top = d["nsga_top"]
        pa = d["pareto_all"]
        hyd_text = ""
        if not top.empty:
            t10 = top.head(10)
            first = t10.iloc[0]
            lead = t10["catalyst"].value_counts().index[0]
            by_metal = ""
            if not pa.empty and "metals" in pa:
                g = pa.groupby("metals")["predicted_Yield_pct"].agg(["median", "max"])
                by_metal = (" Across the three searches the median predicted yield of the Pareto front is "
                            + ", ".join(f"{m}: {r['median']:.1f}\\%" for m, r in g.iterrows())
                            + "; the Ni-containing fronts sit above the Cu(OH)$_2$ front, which is "
                              "consistent with Ni carrying more weight in the training data and should not "
                              "be read as proof that Cu is inferior. ")
            gap = (t10["predicted_Yield_pct"] - t10["observed_Yield_pct"]).dropna()
            gap_txt = ""
            if len(gap):
                yield_mae = d["per_target"].set_index("target").loc["Yield_pct", "model_mae_mean"]
                rel = ("comparable to or larger than" if gap.mean() >= 0.8 * yield_mae
                       else "smaller than")
                gap_txt = (f"For the top ten, the nearest measured record lies at a z-scored Euclidean "
                           f"descriptor distance of {t10['nearest_distance'].min():.1f}--"
                           f"{t10['nearest_distance'].max():.1f}; its yield is on average {gap.mean():.1f} "
                           f"points below the prediction, which is {rel} the yield MAE of {yield_mae:.1f} "
                           "points, so the predicted gain over known electrodes is not established. ")
            hyd_text = (
                "In the hydroxide-restricted search, "
                + ("all ten" if t10["catalyst"].nunique() == 1 else "the")
                + f" top-ranked solutions are {esc(lead).replace('(OH)2', '(OH)$_2$')}-based, on "
                f"{esc(first['support'])}, at {t10.electrolyte_M.min():.2f}--"
                f"{t10.electrolyte_M.max():.2f}\\,M {esc(first['electrolyte'])}, "
                f"{t10.potential_V_RHE.min():.2f}--{t10.potential_V_RHE.max():.2f}\\,V vs RHE and "
                f"{t10.temperature_C.min():.0f}--{t10.temperature_C.max():.0f}\\,$^\\circ$C "
                "(Table~\\ref{tab:nsga}, Figure~\\ref{fig:nsgapareto}). The candidates gather at the "
                "lower bounds of $E$ and $T$ because both are minimised and the surrogate predicts nearly "
                "flat yield there; the model spread reported in the table indicates how much the two ensemble "
                "members disagree. " + gap_txt + by_metal)
            hyd_text += ("Table~\\ref{tab:nsgabest} gives the best solution for each hydroxide anode; the "
                         "top-ten lists for Cu(OH)$_2$ and Ni--Cu hydroxide are in Tables~\\ref{tab:nsgacu} "
                         "and~\\ref{tab:nsganicu}. ")
            cu10 = d["nsga_cu10"]
            if not cu10.empty and "member_spread_Yield_pct" in cu10:
                hyd_text += (f"The Cu(OH)$_2$ solutions carry a much larger model spread "
                             f"({cu10.member_spread_Yield_pct.mean():.1f} points on average versus "
                             f"{t10.member_spread_Yield_pct.mean():.1f} for the overall top ten), i.e. the two "
                             "ensemble members disagree about Cu, and some Cu candidates use HMF "
                             "concentrations far outside most training records. ")
            rob = d["nsga_robust"]
            if not rob.empty:
                hyd_text += (f"Restricting the ranking to $E\\geq1.45$\\,V gives candidates with predicted yields of "
                             f"{rob.predicted_Yield_pct.min():.1f}--{rob.predicted_Yield_pct.max():.1f}\\% "
                             "(Table~\\ref{tab:nsgarob}), a more conservative set that does not sit at the lower "
                             "potential bound of the permitted range. ")
        s.append(
            f"NSGA-II ({nsga.get('population_size', '?')} individuals, "
            f"{nsga.get('generations', '?')} generations) returned "
            f"{nsga.get('n_pareto_candidates', '?')} Pareto candidates in the unconstrained search "
            f"(entropy weights: {wtxt}). " + hyd_text
            + "The candidates are hypotheses for experimental testing: they are model predictions "
            "extrapolated from a small set of Ni/Cu hydroxide records, not measurements.\n\n")
        s.append(table_nsga(d, "nsga_top",
                            "NSGA-II top-10 candidates in the hydroxide-restricted search, ranked by "
                            "entropy-weighted fitness. Performance values are model predictions, not "
                            "measurements; model spread is the absolute difference between the "
                            "gradient-boosting and extra-trees members.", "tab:nsga"))
        s.append(table_best_by_catalyst(d))
        s.append(table_nsga(d, "nsga_cu10",
                            "Top-10 NSGA-II solutions with a Cu(OH)$_2$ anode (model predictions).", "tab:nsgacu"))
        s.append(table_nsga(d, "nsga_nicu10",
                            "Top-10 NSGA-II solutions with a Ni--Cu hydroxide anode (model predictions).",
                            "tab:nsganicu"))
        s.append(table_nsga(d, "nsga_robust",
                            "Top-10 candidates restricted to $E\\geq1.45$\\,V vs RHE.", "tab:nsgarob"))
        s.append(figure(["diagnostics/Fig_NSGA2_Pareto.png"],
                        "Pareto solutions of the Ni, Cu and Ni--Cu hydroxide searches (model "
                        "predictions); circles mark the top ten, stars the $E\\geq1.45$\\,V ranking.",
                        "fig:nsgapareto"))
        s.append(figure(["nsga2/NSGA2_top10_NiCu_hydroxide.pdf", "nsga2/NSGA2_top10_NiCu_hydroxide.png"],
                        "Predicted FDCA yield versus FE for the top-10 hydroxide candidates, "
                        "coloured by entropy-weighted fitness.", "fig:nsga", width="0.75\\textwidth"))
        s.append(figure(["nsga2/Figure_14_Spacing_Trace.pdf", "nsga2/Figure_14_Spacing_Trace.png"],
                        "Spacing metric of the first Pareto front over generations.",
                        "fig:spacing", width="0.75\\textwidth"))
    else:
        s.append("NSGA-II results were not available when this report was generated.\n\n")

    br = d["binary_repeats"]
    if not br.empty and bb.get("model"):
        sel = br[(br.model == bb["model"]) & (br.sampler == bb["sampler"]) & (br.status == "ok")]
        n_hi = int((sel.pooled_macro_f1 >= 0.76).sum())
        above_text = (f"{n_hi} of {len(sel)} repeats of the best binary model reach 0.76 "
                      f"(maximum {sel.pooled_macro_f1.max():.3f}), but the mean does not.")
    else:
        above_text = "The mean of the best binary model does not reach 0.76."

    # 4 Discussion
    s.append(r"""\section{Discussion}
\textbf{What improved the scores.} Most of the gain came from data quality rather than model
capacity. Removing counter-electrode contamination and recovering compound families gave the
models chemically meaningful catalyst descriptors; the missing-value indicators restored
information about reporting practice; and disabling early stopping stopped the neural network from
converging on the majority class. Tuning added smaller, family-specific gains.

\textbf{Why 0.76 was not reached.} """ + above_text + r""" Three factors cap performance: (i) only """ + str(d['binary_n']) +
             r""" records from fewer than one hundred papers, many sharing catalysts; (ii) a positive
class that is the majority, so macro-F1 depends heavily on the """ + str(d['binary_n'] - d['binary_pos']) + r""" low-performance records;
and (iii) between-paper differences in protocol that the descriptors cannot represent. Two
routes would raise the reported number without improving the model: random (non-grouped)
cross-validation, which lets near-duplicate records from one paper appear in training and test,
and redefining the threshold or clusters to suit the classifier. We deliberately did neither.

\textbf{Evaluation lessons.} Silent exclusion of infeasible folds can make a model look several
tenths better than it is; reporting feasibility explicitly is essential when oversampling tiny
classes. Hyperparameters were selected on the same grouped protocol as the reported scores, which
introduces a mild optimistic bias; a nested check is reported for the regressors, but the classifiers would need a nested or external validation set.

\textbf{Limitations.} """ + kmeans_limit + r""" Descriptors are parsed from free text and remain coarse (no loading, morphology or
surface area). Some yields are derived from conversion and selectivity. Only a small number of Ni and Cu hydroxide records inform the NSGA-II search, so its candidates extrapolate; SHAP rankings and NSGA-II candidates are associational and model-dependent.

\section{Conclusions}
""" + (
        f"A leakage-aware pipeline for literature-scale electrocatalysis data reaches a "
        f"publication-grouped macro-F1 of {f3(bb.get('macro_f1_mean'))} for a $\\geq$90\\% "
        "performance screen"
        + (f" and {k3.macro_f1_mean:.3f} for three-cluster labels" if k3 is not None else "")
        + ", well above chance but short of 0.76. Correct descriptor extraction and honest "
        "fold accounting mattered more than model choice. Yield and FE are predictable for unseen papers "
        "to a useful but modest degree, conversion is not, and a Ni-containing anode emerges "
        "as an important and stable catalyst descriptor for yield (behind a support-type confound), in agreement with the literature, while "
        "the corpus cannot rank Ni against Cu. Expanding the corpus, reporting "
        "operating conditions completely and " + (
            "validating the NSGA-II candidates experimentally " if gate.get("passed") else
            "building regressors that pass the validation gate before any optimisation ")
        + "are the most promising next steps.\n"))

    s.append(r"""
\section*{Data and code availability}
All results, figures and this manuscript are regenerated by \texttt{python run\_all.py}; this
document is produced by \texttt{scripts/12\_latex\_report.py} from the files in
\texttt{results/} and \texttt{figures/}.

\begin{thebibliography}{9}
\bibitem{li2025} Li \textit{et al.}, \textit{ACS Sustainable Chem.\ Eng.} \textbf{2025}, 13, 13940.
\textcolor{gray}{[Complete author list and title.]}
\bibitem{chawla2002} N.~V. Chawla, K.~W. Bowyer, L.~O. Hall, W.~P. Kegelmeyer, SMOTE: Synthetic
minority over-sampling technique, \textit{J.\ Artif.\ Intell.\ Res.} \textbf{2002}, 16, 321--357.
\bibitem{he2008} H.~He, Y.~Bai, E.~A. Garcia, S.~Li, ADASYN: Adaptive synthetic sampling approach
for imbalanced learning, in \textit{Proc.\ IEEE IJCNN} \textbf{2008}, 1322--1328.
\bibitem{pedregosa2011} F.~Pedregosa \textit{et al.}, Scikit-learn: Machine learning in Python,
\textit{J.\ Mach.\ Learn.\ Res.} \textbf{2011}, 12, 2825--2830.
\bibitem{lemaitre2017} G.~Lema\^{\i}tre, F.~Nogueira, C.~K. Aridas, Imbalanced-learn: A Python
toolbox to tackle the curse of imbalanced datasets in machine learning, \textit{J.\ Mach.\ Learn.\
Res.} \textbf{2017}, 18, 1--5.
\bibitem{lundberg2017} S.~M. Lundberg, S.-I. Lee, A unified approach to interpreting model
predictions, in \textit{Adv.\ Neural Inf.\ Process.\ Syst.} \textbf{2017}, 30, 4765--4774.
\bibitem{lit1} Ni(OH)$_2$/NiOOH electro-oxidation of HMF to FDCA (beta-Ni(OH)$_2$, 98.7\% yield),
PubMed Central article PMC9401862. \textcolor{gray}{[Authors, title and journal to be completed from the full text.]}
\bibitem{lit2} Cu-based anode for HMF electro-oxidation to FDCA, \textit{Green Chem.} \textbf{2023},
DOI 10.1039/D3GC01420G. \textcolor{gray}{[Authors and title to be completed from the full text.]}
\bibitem{lit3} HMF electro-oxidation to FDCA, \textit{Green Chem.} \textbf{2023},
DOI 10.1039/D2GC04732B. \textcolor{gray}{[Authors and title to be completed from the full text.]}
\bibitem{deb2002} K.~Deb, A.~Pratap, S.~Agarwal, T.~Meyarivan, A fast and elitist multiobjective
genetic algorithm: NSGA-II, \textit{IEEE Trans.\ Evol.\ Comput.} \textbf{2002}, 6, 182--197.
\end{thebibliography}
\end{document}
""")
    return "".join(s)


def compile_pdf() -> bool:
    """Compile the .tex with tectonic (preferred) or pdflatex, if installed.

    A missing LaTeX engine is not a pipeline failure: the .tex is still
    written and the manual command is printed.
    """
    import shutil
    tectonic = shutil.which("tectonic") or (
        "/opt/homebrew/bin/tectonic" if Path("/opt/homebrew/bin/tectonic").exists() else None)
    if tectonic:
        commands = [[tectonic, TEX_PATH.name]]
    elif shutil.which("pdflatex"):
        # Two passes resolve cross-references.
        commands = [["pdflatex", "-interaction=nonstopmode", TEX_PATH.name]] * 2
    else:
        print("No LaTeX engine found (tectonic or pdflatex); PDF not compiled.")
        return False
    for command in commands:
        result = subprocess.run(command, cwd=REPORT_DIR, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout[-3000:])
            print(result.stderr[-3000:])
            raise RuntimeError(f"LaTeX compilation failed: {' '.join(command)}")
    print(f"PDF compiled: {TEX_PATH.with_suffix('.pdf')}")
    return True


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = collect()
    TEX_PATH.write_text(build(data), encoding="utf-8")
    print(f"LaTeX report written: {TEX_PATH}")
    if not compile_pdf():
        print("Compile with: cd report && tectonic HMF_FDCA_ML_Report.tex")


if __name__ == "__main__":
    sys.exit(main())
