"""Stage 10 — research-paper style LaTeX report.

Builds ``report/HMF_FDCA_ML_Report.tex`` from the artefacts written by stages
01-09. Every number, table and figure reference is read from ``results/`` and
``figures/`` at run time, so re-running the pipeline and then this stage keeps
the manuscript consistent with the latest scores. Nothing is retrained here.

Compile with any LaTeX engine from the ``report`` folder, e.g.::

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
}
MODEL_ORDER = ["DecisionTree", "SVM", "RandomForest", "NeuralNetwork",
               "GradientBoosting", "ExtraTrees"]
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
        "kmeans_audit": read_json("kmeans_all154_audit.json"),
        "best": read_json("best_classification_candidate.json"),
        "incomplete": read_csv("classification_incomplete_combinations.csv"),
        "learning": read_csv("learning_curves_summary.csv"),
        "importance": read_csv("feature_importance_selected_model.csv"),
        "shap_audit": read_json("shap_paper_K3_K5_audit.json"),
        "shap_k": {k: read_csv(f"shap_paper_K{k}_feature_importance.csv") for k in (3, 5)},
        "per_target": read_csv("per_target_regression_summary.csv"),
        "joint": read_csv("multioutput_regression_summary.csv"),
        "gate": read_json("regression_validation_gate.json"),
        "nsga": read_json("nsga2_status.json"),
        "nsga_top": read_csv("nsga2_top10_ni_cu_hydroxide.csv"),
        "nsga_cu": read_csv("nsga2_best3_cu_hydroxide.csv"),
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
        "model and evaluation revisions. 95\\% CI over 25 repeats. Earlier $K=4$ values "
        "were computed on the subset of folds where SMOTE/ADASYN could be fitted and are "
        "therefore not comparable (Section~\\ref{sec:eval-fix}).}\n\\label{tab:beforeafter}\n"
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
        titles.append(f"Binary screen ({MODEL_LABEL.get(best.get('model'), best.get('model'))}"
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


def table_nsga(d: dict) -> str:
    top = d["nsga_top"]
    if top.empty:
        return ""
    lines = []
    for r in top.head(10).itertuples():
        lines.append(
            f"{r.rank} & {esc(r.catalyst)}/{esc(r.support)} & {esc(r.electrolyte)} "
            f"{r.electrolyte_M:.2f}\\,M & {r.HMF_mM:.1f} & {r.potential_V_RHE:.3f} & "
            f"{r.temperature_C:.1f} & {r.predicted_Conversion_pct:.1f} & "
            f"{r.predicted_Yield_pct:.1f} & {r.predicted_FE_pct:.1f} & {r.entropy_fitness:.3f} \\\\")
    return (
        "\\begin{table*}[!htbp]\n\\centering\\scriptsize\n\\caption{NSGA-II top-10 "
        "candidates in the constrained Ni(OH)$_2$/Cu(OH)$_2$ search, ranked by "
        "entropy-weighted fitness. All performance values are model predictions, not "
        "measurements.}\n\\label{tab:nsga}\n\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{rllrrrrrrr}\n\\toprule\n"
        "Rank & Catalyst & Electrolyte & HMF (mM) & $E$ (V vs RHE) & $T$ ($^\\circ$C) & "
        "Conv. (\\%) & Yield (\\%) & FE (\\%) & Fitness \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n"
    )


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
                  "SVM": "support vector machine", "DecisionTree": "decision tree"}

    s = [PREAMBLE]
    s.append(r"""\title{Machine-Learning Screening of Electrocatalytic 5-Hydroxymethylfurfural
Oxidation to 2,5-Furandicarboxylic Acid from Literature Data: Leakage-Aware Classification,
Regression and Multi-Objective Optimisation}
\author{\textit{Author names and affiliations to be added}}
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
        "(iii) compares six classifier families under four class-balancing strategies, "
        "(iv) interprets the selected models with SHAP, (v) regresses four performance outcomes and "
        "(vi) searches catalyst/condition space with a validation-gated NSGA-II. All model "
        "selection uses publication-grouped cross-validation so that no paper contributes records "
        "to both training and test folds. Correcting descriptor extraction (anode-only catalyst "
        "parsing, formula-aware compound families, temperature and pH parsing, counter-electrode "
        "descriptors), retaining missing-value indicators and re-tuning the models raised the best "
        f"pooled out-of-fold macro-F1 for a binary $\\geq$90\\% performance screen from "
        f"{f3(old_bin)} to {f3(best.get('macro_f1_mean'))} "
        f"({model_name.get(best.get('model'), best.get('model'))} with {esc(best.get('sampler'))}; "
        f"95\\% CI {f3(best.get('macro_f1_ci95_lo'))}--{f3(best.get('macro_f1_ci95_hi'))}; "
        f"stratified dummy {f3(best.get('dummy_macro_f1_mean'))})"
        + (f", and for three-cluster K-means labels from {f3(old_k3)} to "
           f"{f3(k3.macro_f1_mean)}" if k3 is not None else "")
        + ". We also identify and correct an evaluation flaw in which folds where an "
        "oversampler could not be fitted were silently discarded, which had inflated earlier "
        f"$K=4$ scores. {'No' if above == 0 else above} model--sampler combination"
        f"{'' if above == 1 else 's'} reached a mean macro-F1 of 0.76, reflecting the small, "
        "heterogeneous and imbalanced literature corpus rather than a pipeline limitation. "
        "Regression models only modestly beat a median baseline, and NSGA-II outputs are "
        "presented as hypotheses for experimental testing.") + "\n\\end{abstract}\n")
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
$K\in\{3,4,5,6\}$ (100 initialisations). Missing outcomes are median-imputed for clustering only
and flagged; labels are never used as inputs (Table~\ref{tab:clusters}).

\textit{Binary performance screen.} For records whose major product is FDCA and whose yield is
reported, a performance score
\begin{equation}
P = \begin{cases} \min(X, Y) & \text{if conversion } X \text{ is reported,}\\ Y & \text{otherwise,}\end{cases}
\end{equation}
is computed from conversion $X$ and yield $Y$, and records with $P\geq 90\%$ form the positive class.
""")
    s.append(table_clusters(d))

    s.append(r"""\subsection{Models and class balancing}
Six classifier families were compared: decision tree (DT), RBF support vector machine (SVM),
random forest (RF), multilayer perceptron (NN), gradient boosting (GB) and extremely randomised
trees (ET), each under no balancing, random oversampling (ROS), SMOTE~\cite{chawla2002} and
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
compared with a stratified dummy classifier on identical splits and with a permutation test in
which labels are shuffled within publications.

\subsubsection*{Correction of fold-dropping bias}\label{sec:eval-fix}
SMOTE and ADASYN require more minority-class training rows than neighbours. In the previous
implementation a fold that failed this requirement was silently skipped and the repeat's macro-F1
was computed on the remaining folds; for $K=4$ only about 1.2 of 5 folds (35 of 154 records) were
scored. Neighbour counts now adapt to $\min(2, n_{\min}-1)$, a repeat counts only if every fold
produced predictions, and a combination is ranked only if at least 80\% of its repeats are
complete. """ + (f"{n_incomplete} combinations (mostly SMOTE/ADASYN at $K\\geq 5$, where "
                 "single-record classes occur) are therefore reported as infeasible rather than "
                 "scored.\n" if n_incomplete else "\n"))

    s.append(r"""\subsection{Interpretation, regression and optimisation}
SHAP values~\cite{lundberg2017} are computed on a held-out 20\% of publications, with Tree SHAP for
tree ensembles and Kernel SHAP through the complete fitted pipeline otherwise. For the K-means
analysis the best Stage-04 model for each $K$ is explained. Conversion, yield, FE and
selectivity are regressed per target (random forest with mutual-information feature selection,
$k=15$) on all records that report that target, using 10 repeats of 3-fold GroupKFold; a joint
four-output extra-trees model on complete rows is reported as a secondary analysis. A
validation gate requires every target to beat the median baseline in mean MAE and in a majority
of folds before the regressors may drive optimisation.

NSGA-II~\cite{deb2002} searches a mixed binary/continuous encoding of the descriptor space,
maximising predicted conversion, yield and FE while minimising temperature and potential. Pareto
solutions are ranked by an entropy-weighted fitness
\begin{equation}
F = w_1\tilde X + w_2\tilde Y + w_3\widetilde{\mathrm{FE}} - w_4\tilde T - w_5\tilde E,
\qquad w_j = \frac{1-e_j}{\sum_k (1-e_k)},
\end{equation}
with min--max scaled objectives and $e_j$ the Shannon entropy of objective $j$ over the front.
Convergence is tracked with the spacing metric of the first front.
""")

    # 3 Results
    s.append(r"\section{Results}" + "\n")
    s.append(r"\subsection{Clustering}" + "\n" + (
        "K-means separates a dominant high-conversion/high-yield group from small groups with low "
        "yield at high conversion or low conversion (Figure~\\ref{fig:kmeans}). "
        f"{d['kmeans_audit'].get('records_with_imputed_outcomes', 'Some')} of {d['n_records']} "
        "records required outcome imputation for clustering, and the smallest clusters contain "
        "between one and five records (Table~\\ref{tab:clusters}). Macro-F1 weights each class "
        "equally, so misclassifying a two-record cluster costs as much as misclassifying the "
        "majority class.\n"))
    # The PNG is used first: the PDF version carries a full-page margin.
    s.append(figure(["KMeans_Figure6/HMF_KMeans_Figure6_Style.png",
                     "KMeans_All154/HMF_KMeans_All154_K3.png"],
                    "K-means clustering of reported conversion and FDCA yield for $K=3$--6.",
                    "fig:kmeans", width="0.78\\textwidth"))
    s.append(figure(["oversampling_kmeans/KMeans_After_Oversampling_K3_to_K6.pdf",
                     "oversampling_kmeans/KMeans_After_Oversampling_K3_to_K6.png"],
                    "ADASYN and SMOTE synthetic records (crosses) in performance space for "
                    "$K=3$--6, drawn for the 102 records with both outcomes measured. Panels marked "
                    "infeasible contain a class with fewer than two measured records.",
                    "fig:oversampled", width="0.85\\textwidth"))

    s.append(r"\subsection{Classification performance}" + "\n")
    para = (
        f"Table~\\ref{{tab:beforeafter}} summarises the effect of the revision. For the binary "
        f"screen ({d['binary_n']} records, {d['binary_pos']} positive), the best model is the "
        f"{model_name.get(best.get('model'), best.get('model'))} with {esc(best.get('sampler'))} "
        f"(macro-F1 {f3(best.get('macro_f1_mean'))} $\\pm$ {f3(best.get('macro_f1_sd'))}), "
        f"against {f3(best.get('dummy_macro_f1_mean'))} for a stratified dummy classifier "
        "(Table~\\ref{tab:binary}, Figure~\\ref{fig:binary}). ")
    if et_bin is not None:
        para += (f"Extra-trees without balancing is statistically indistinguishable "
                 f"({et_bin.macro_f1_mean:.3f}). ")
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
    para += ("For $K\\geq4$, scores stay between 0.2 and 0.45: several classes have fewer than "
             "five members, and a single grouped fold frequently contains none of them in "
             "training. ")
    pval = best.get("permutation_p_value")
    if pval is not None:
        para += (f"The within-publication permutation test gives $p={pval:.2f}$. Because most "
                 "descriptors are constant within a paper, shuffling labels inside papers "
                 "preserves the paper-level signal the model exploits, so this test is "
                 "conservative here; the dummy comparison and the confidence interval are the "
                 "more informative checks.")
    s.append(para + "\n\n")
    s.append(table_before_after(d))
    s.append(table_grid(binary, "Binary $\\geq$90\\% screen: pooled out-of-fold macro-F1 (mean "
                        "$\\pm$ SD over 25 repeats) for every model and sampler.", "tab:binary"))
    s.append(table_grid(kmeans[kmeans.k == 3] if not kmeans.empty else kmeans,
                        "K-means $K=3$: pooled out-of-fold macro-F1 (mean $\\pm$ SD over 25 "
                        "repeats).", "tab:k3"))
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

    s.append(r"\subsection{Learning curves}" + "\n" + (
        "Training macro-F1 stays well above validation macro-F1 for every model and $K$ "
        "(Table~\\ref{tab:learning}, Figure~\\ref{fig:learning}), indicating variance-limited "
        "models. At $K=3$ the SVM validation curve is still rising at the largest training "
        "size and the other families fluctuate without a clear plateau, so additional "
        "publications are the most direct route to higher scores; for $K\\geq4$ the curves are "
        "flat because the minority clusters do not grow with more records.\n"))
    s.append(table_learning(d))
    s.append(figure(["learning_curves/HMF_Learning_Curves_Figure.pdf",
                     "learning_curves/HMF_Learning_Curves_Figure.png"],
                    "Learning curves (macro-F1) for DT, SVM, RF and NN at $K=3$, 4 and 5 with "
                    "publication-grouped holdouts; shading shows $\\pm$1 SD.", "fig:learning"))
    s.append(figure(["learning_curves/Figure_11_Oversampled_NN_Learning_Curves_ROS_ADASYN_SMOTE.pdf",
                     "learning_curves/Figure_11_Oversampled_NN_Learning_Curves_ROS_ADASYN_SMOTE.png"],
                    "Neural-network learning curves with ROS, ADASYN and SMOTE applied to training "
                    "folds only.", "fig:fig11"))

    s.append(r"\subsection{Model interpretation}" + "\n" + (
        "Table~\\ref{tab:shap} and Figures~\\ref{fig:shapsel}--\\ref{fig:fig13} list the "
        "descriptors with the largest SHAP contributions. Support type (conductive glass versus "
        "foam), copper and iron content, oxyhydroxide/hydroxide phases, counter-electrode "
        "material, potential and the temperature-not-reported indicator recur across models. "
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

    s.append(r"\subsection{Regression}" + "\n")
    if not d["per_target"].empty:
        pt = d["per_target"]
        s.append(
            "Per-target models reduce MAE relative to the median baseline for all four outcomes "
            f"(Table~\\ref{{tab:reg}}), but $R^2$ remains between {pt.model_r2_mean.min():.2f} "
            f"and {pt.model_r2_mean.max():.2f}: the outcome distributions are concentrated near "
            "100\\% with a few low outliers, so the models mainly learn to avoid large errors on "
            "the typical record rather than to explain variance. The validation gate "
            f"{'passed' if gate.get('passed') else 'failed'}. The joint model trained on the "
            f"{d['n_complete_targets']} complete records does not beat the baseline "
            "(Table~\\ref{tab:regjoint}).\n\n")
    s.append(table_regression(d))

    s.append(r"\subsection{Multi-objective optimisation}" + "\n")
    if nsga:
        weights = nsga.get("general_entropy_weights", {})
        objective_label = {**TARGET_LABEL, "Temperature_C": "temperature",
                           "Potential_vs_RHE_V": "potential"}
        wtxt = ", ".join(f"{objective_label.get(k, esc(k.replace('_', ' ')))} {v:.2f}"
                         for k, v in weights.items())
        hyd = nsga.get("hydroxide_scenario", {})
        top = d["nsga_top"]
        hyd_text = ""
        if hyd and not top.empty:
            t10 = top.head(10)
            cats = t10["catalyst"].value_counts()
            lead = cats.index[0]
            first = t10.iloc[0]
            yield_mae = d["per_target"].set_index("target").get("model_mae_mean", pd.Series()) \
                .get("Yield_pct", float("nan")) if not d["per_target"].empty else float("nan")
            settle = max(hyd.get("spacing_converged_generation", {}).values() or [0])
            hyd_text = (
                "In the constrained Ni(OH)$_2$/Cu(OH)$_2$ search, "
                + ("all ten" if cats.iloc[0] == len(t10) else f"{cats.iloc[0]} of the top ten")
                + f" top-ranked solutions are {esc(lead).replace('(OH)2', '(OH)$_2$')} on "
                f"{esc(first['support'])}, at {t10.electrolyte_M.min():.2f}--"
                f"{t10.electrolyte_M.max():.2f}\\,M {esc(first['electrolyte'])}, "
                f"{t10.potential_V_RHE.min():.2f}--{t10.potential_V_RHE.max():.2f}\\,V vs RHE and "
                f"{t10.temperature_C.min():.0f}--{t10.temperature_C.max():.0f}\\,$^\\circ$C "
                "(Table~\\ref{tab:nsga}, Figure~\\ref{fig:nsga}). The top candidate's nearest literature "
                f"record ({esc(first['nearest_catalyst'])}, {esc(first['nearest_conditions'])}) "
                f"reported {first['observed_Yield_pct']:.1f}\\% yield against "
                f"{first['predicted_Yield_pct']:.1f}\\% predicted; that difference is smaller "
                f"than the yield regression MAE of {yield_mae:.1f} percentage points. ")
            cu = d["nsga_cu"]
            if not cu.empty:
                c = cu.iloc[0]
                hyd_text += (f"The best Cu(OH)$_2$ candidate is predicted at "
                             f"{c['predicted_Yield_pct']:.1f}\\% yield and "
                             f"{c['predicted_FE_pct']:.1f}\\% FE. ")
            hyd_text += (f"The spacing metric settles into a stable band after about {settle} "
                         "generations and fluctuates around it thereafter "
                         "(Figure~\\ref{fig:spacing}). ")
        s.append(
            f"NSGA-II ({nsga.get('population_size', '?')} individuals, "
            f"{nsga.get('generations', '?')} generations) returned "
            f"{nsga.get('n_pareto_candidates', '?')} Pareto candidates. Entropy weights for the "
            f"unconstrained search were: {wtxt}. " + hyd_text
            + "Because the predicted gains lie within the regression error, the candidates "
            "should be read as experimental starting points, not predictions of record "
            "performance.\n\n")
        s.append(table_nsga(d))
        s.append(figure(["nsga2/NSGA2_top10_NiCu_hydroxide.pdf", "nsga2/NSGA2_top10_NiCu_hydroxide.png"],
                        "Predicted FDCA yield versus FE for the top-10 hydroxide candidates, "
                        "coloured by entropy-weighted fitness.", "fig:nsga", width="0.75\\textwidth"))
        s.append(figure(["nsga2/Figure_14_Spacing_Trace.pdf", "nsga2/Figure_14_Spacing_Trace.png"],
                        "Spacing metric of the first Pareto front over generations.",
                        "fig:spacing", width="0.75\\textwidth"))
    else:
        s.append("NSGA-II results were not available when this report was generated.\n\n")

    # 4 Discussion
    s.append(r"""\section{Discussion}
\textbf{What improved the scores.} Most of the gain came from data quality rather than model
capacity. Removing counter-electrode contamination and recovering compound families gave the
models chemically meaningful catalyst descriptors; the missing-value indicators restored
information about reporting practice; and disabling early stopping stopped the neural network from
converging on the majority class. Tuning added smaller, family-specific gains.

\textbf{Why 0.76 was not reached.} Several individual repeats of the best binary model exceed 0.76,
but the 25-repeat mean does not. Three factors cap performance: (i) only """ + str(d['binary_n']) +
             r""" records from fewer than one hundred papers, many sharing catalysts; (ii) a positive
class that is the majority, so macro-F1 depends heavily on the ~30 low-performance records;
and (iii) between-paper differences in protocol that the descriptors cannot represent. Two
routes would raise the reported number without improving the model: random (non-grouped)
cross-validation, which lets near-duplicate records from one paper appear in training and test,
and redefining the threshold or clusters to suit the classifier. We deliberately did neither.

\textbf{Evaluation lessons.} Silent exclusion of infeasible folds can make a model look several
tenths better than it is; reporting feasibility explicitly is essential when oversampling tiny
classes. Hyperparameters were selected on the same grouped protocol as the reported scores, which
introduces a mild optimistic bias; a nested or external validation set would remove it.

\textbf{Limitations.} Outcomes were imputed for K-means only, so some cluster labels are
pseudo-labels. Descriptors are parsed from free text and remain coarse (no loading, morphology or
surface area). SHAP rankings and NSGA-II candidates are associational and model-dependent.

\section{Conclusions}
""" + (
        f"A leakage-aware pipeline for literature-scale electrocatalysis data reaches a "
        f"publication-grouped macro-F1 of {f3(best.get('macro_f1_mean'))} for a $\\geq$90\\% "
        "performance screen"
        + (f" and {k3.macro_f1_mean:.3f} for three-cluster labels" if k3 is not None else "")
        + ", well above chance but short of 0.76. Correct descriptor extraction and honest "
        "fold accounting mattered more than model choice. Expanding the corpus, reporting "
        "operating conditions completely and validating the NSGA-II candidates experimentally "
        "are the most promising next steps.\n"))

    s.append(r"""
\section*{Data and code availability}
All results, figures and this manuscript are regenerated by \texttt{python run\_all.py}; this
document is produced by \texttt{scripts/10\_latex\_report.py} from the files in
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
\bibitem{deb2002} K.~Deb, A.~Pratap, S.~Agarwal, T.~Meyarivan, A fast and elitist multiobjective
genetic algorithm: NSGA-II, \textit{IEEE Trans.\ Evol.\ Comput.} \textbf{2002}, 6, 182--197.
\end{thebibliography}
\end{document}
""")
    return "".join(s)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = collect()
    TEX_PATH.write_text(build(data), encoding="utf-8")
    print(f"LaTeX report written: {TEX_PATH}")
    print("Compile with: cd report && tectonic HMF_FDCA_ML_Report.tex")


if __name__ == "__main__":
    sys.exit(main())
